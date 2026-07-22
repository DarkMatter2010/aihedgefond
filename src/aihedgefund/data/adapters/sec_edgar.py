"""Official SEC EDGAR Form 4 adapter (data.sec.gov + Archives XML).

No third-party wrappers. HTTP is confined here. Filings are immutable once
accepted, so responses are aggressively disk-cached. Rate-limited to
``edgar.max_rps`` (hard ceiling 10 req/s per SEC fair-access policy).

PIT: only ``filed_at`` (= acceptanceDateTime) is used as availability time.
``transaction_date`` is retained for audit and must not drive features.
"""

from __future__ import annotations

import json
import threading
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.config import EdgarSettings
from aihedgefund.core.ports import InsiderFilingPort
from aihedgefund.core.runtime import Clock, SystemClock
from aihedgefund.core.schemas import (
    Form4Frame,
    Form4Ingested,
    Form4Record,
    Form4Request,
    IngestedForm4Data,
)
from aihedgefund.data.form4_quality import Form4QualityGate
from aihedgefund.data.provider import DataUnavailableError

DATA_SEC_BASE: Final[str] = "https://data.sec.gov"
TICKERS_URL: Final[str] = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_BASE: Final[str] = "https://www.sec.gov/Archives/edgar/data"
FORM4_TYPES: Final[frozenset[str]] = frozenset({"4", "4/A"})
_XML_VALUE_RE: Final[re.Pattern[str]] = re.compile(r"\{.*\}")


HttpGet = Callable[[str], bytes]


class _TokenBucket:
    """Simple rate limiter: at most `rate` tokens per second (thread-safe)."""

    def __init__(self, rate: float) -> None:
        if rate <= 0.0 or rate > 10.0:
            msg = "rate must be in (0, 10]"
            raise ValueError(msg)
        self._rate = float(rate)
        self._tokens = float(rate)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated
                self._updated = now
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)
def _local_name(tag: str) -> str:
    return _XML_VALUE_RE.sub("", tag)


def _child_text(node: ET.Element, *path: str) -> str | None:
    current: ET.Element | None = node
    for name in path:
        if current is None:
            return None
        nxt: ET.Element | None = None
        for child in current:
            if _local_name(child.tag) == name:
                nxt = child
                break
        current = nxt
    if current is None:
        return None
    text = (current.text or "").strip()
    if text:
        return text
    for child in current:
        if _local_name(child.tag) == "value" and child.text:
            return child.text.strip()
    return None


def _parse_acceptance(raw: str) -> datetime:
    # SEC uses "2023-01-15 16:05:12" or ISO with Z.
    cleaned = raw.strip().replace("Z", "+00:00")
    if "T" not in cleaned and " " in cleaned:
        cleaned = cleaned.replace(" ", "T", 1)
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_date(raw: str | None) -> date | None:
    if raw is None or not raw.strip():
        return None
    return date.fromisoformat(raw.strip()[:10])


def parse_form4_xml(
    xml_bytes: bytes,
    *,
    symbol: str,
    cik: str,
    accession: str,
    filed_at: datetime,
) -> tuple[Form4Record, ...]:
    """Parse non-derivative (and derivative) transactions from ownership XML."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        msg = f"form4 XML parse failed for {accession}: {exc}"
        raise DataUnavailableError(msg) from exc

    owner = _child_text(root, "reportingOwner", "reportingOwnerId", "rptOwnerName")
    records: list[Form4Record] = []
    for table_name in ("nonDerivativeTable", "derivativeTable"):
        table = None
        for child in root.iter():
            if _local_name(child.tag) == table_name:
                table = child
                break
        if table is None:
            continue
        for node in table:
            local = _local_name(node.tag)
            if local not in {"nonDerivativeTransaction", "derivativeTransaction"}:
                continue
            code = _child_text(node, "transactionCoding", "transactionCode")
            shares_raw = _child_text(node, "transactionAmounts", "transactionShares")
            ad_raw = _child_text(
                node,
                "transactionAmounts",
                "transactionAcquiredDisposedCode",
            )
            price_raw = _child_text(node, "transactionAmounts", "transactionPricePerShare")
            txn_date = _parse_date(_child_text(node, "transactionDate"))
            if code is None or shares_raw is None or ad_raw is None:
                continue
            ad = ad_raw.strip().upper()
            if ad not in {"A", "D"}:
                continue
            try:
                shares = float(shares_raw.replace(",", ""))
            except ValueError:
                continue
            price: float | None
            try:
                price = float(price_raw.replace(",", "")) if price_raw else None
            except ValueError:
                price = None
            records.append(
                Form4Record(
                    symbol=symbol,
                    cik=cik,
                    accession=accession,
                    filed_at=filed_at,
                    transaction_date=txn_date,
                    transaction_code=code.strip().upper(),
                    shares=shares,
                    price=price,
                    acquired_disposed=ad,  # type: ignore[arg-type]
                    reporting_owner=owner,
                )
            )
    return tuple(records)


class SecEdgarForm4Provider(InsiderFilingPort):
    """Fetch Form 4 filings from official SEC endpoints with disk cache."""

    def __init__(
        self,
        settings: EdgarSettings,
        bus: MessageBus,
        quality_gate: Form4QualityGate,
        *,
        clock: Clock | None = None,
        symbol_aliases: Mapping[str, str] | None = None,
        http_get: HttpGet | None = None,
        include_historical_files: bool = False,
        cache_only: bool = False,
        skip_uncached_filings: bool = False,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._quality_gate = quality_gate
        self._clock = clock or SystemClock()
        self._symbol_aliases = dict(symbol_aliases or {})
        self._bucket = _TokenBucket(settings.max_rps)
        self._http_get = http_get or self._default_http_get
        self._cache_dir = Path(settings.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._include_historical_files = bool(include_historical_files)
        self._cache_only = bool(cache_only)
        self._skip_uncached_filings = bool(skip_uncached_filings)

    def get_form4(self, request: Form4Request) -> Form4Frame:
        """Return Form 4 rows with `filed_at` inside `[start, end]`."""
        tickers = self._load_ticker_map()
        n_symbols = len(request.symbols)
        workers = max(1, min(8, int(self._settings.max_rps)))

        def _one(symbol: str) -> tuple[str, tuple[Form4Record, ...], bool]:
            vendor = self._symbol_aliases.get(symbol, symbol)
            cik = tickers.get(vendor.upper())
            if cik is None:
                return symbol, (), True
            symbol_records = self._records_for_symbol(
                symbol=symbol,
                cik=cik,
                start=request.start,
                end=request.end,
            )
            return symbol, symbol_records, len(symbol_records) == 0

        records: list[Form4Record] = []
        without: list[str] = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, symbol): symbol for symbol in request.symbols}
            for fut in as_completed(futures):
                symbol, symbol_records, missing = fut.result()
                done += 1
                if missing:
                    without.append(symbol)
                records.extend(symbol_records)
                print(
                    f"form4 progress {done}/{n_symbols} {symbol}: "
                    f"+{len(symbol_records)} rows (total {len(records)})",
                    flush=True,
                )

        # Deduplicate identical transaction rows (same filing can repeat lines).
        deduped: list[Form4Record] = []
        seen_rows: set[tuple[object, ...]] = set()
        for record in records:
            key = (
                record.accession,
                record.symbol,
                record.transaction_date,
                record.reporting_owner,
                record.transaction_code,
                float(record.shares),
                record.acquired_disposed,
            )
            if key in seen_rows:
                continue
            seen_rows.add(key)
            deduped.append(record)
        frame = Form4Frame(
            records=tuple(deduped),
            symbols_queried=request.symbols,
            symbols_without_filings=tuple(sorted(set(without))),
        )
        self._quality_gate.validate(frame, now=self._clock.now())
        self._bus.publish_event(
            Form4Ingested(
                timestamp=self._clock.now(),
                payload=IngestedForm4Data(request=request, data=frame),
            )
        )
        return frame


    def _records_for_symbol(
        self,
        *,
        symbol: str,
        cik: str,
        start: datetime,
        end: datetime,
    ) -> tuple[Form4Record, ...]:
        try:
            submissions = self._load_submissions(cik)
        except DataUnavailableError:
            return ()
        filings_block = submissions.get("filings", {})
        if not isinstance(filings_block, dict):
            msg = f"submissions filings block invalid for CIK {cik}"
            raise DataUnavailableError(msg)
        chunks: list[dict[str, object]] = []
        recent = filings_block.get("recent", {})
        if isinstance(recent, dict):
            chunks.append(recent)
        if self._include_historical_files:
            files = filings_block.get("files", [])
            if isinstance(files, list):
                for meta in files:
                    if not isinstance(meta, dict):
                        continue
                    name = str(meta.get("name", ""))
                    if not name:
                        continue
                    chunks.append(self._load_submission_chunk(cik, name))

        out: list[Form4Record] = []
        for chunk in chunks:
            out.extend(
                self._records_from_filing_chunk(
                    chunk,
                    symbol=symbol,
                    cik=cik,
                    start=start,
                    end=end,
                )
            )
        return tuple(out)

    def _load_submission_chunk(self, cik: str, name: str) -> dict[str, object]:
        padded = f"{int(cik):010d}"
        path = self._cache_dir / "submissions" / name
        url = f"{DATA_SEC_BASE}/submissions/{name}"
        raw = self._read_cache_or_fetch(path, url)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"submission chunk JSON invalid for {name}"
            raise DataUnavailableError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"submission chunk must be an object: {name}"
            raise DataUnavailableError(msg)
        # Historical files are the filing arrays directly (same shape as recent).
        return payload

    def _records_from_filing_chunk(
        self,
        chunk: dict[str, object],
        *,
        symbol: str,
        cik: str,
        start: datetime,
        end: datetime,
    ) -> tuple[Form4Record, ...]:
        forms = chunk.get("form", [])
        accessions = chunk.get("accessionNumber", [])
        acceptance = chunk.get("acceptanceDateTime", [])
        primary_docs = chunk.get("primaryDocument", [])
        if not isinstance(forms, list):
            return ()
        if not (
            len(forms) == len(accessions) == len(acceptance) == len(primary_docs)
        ):
            msg = f"submissions array length mismatch for CIK {cik}"
            raise DataUnavailableError(msg)

        jobs: list[tuple[str, str, datetime]] = []
        for form, accession, accepted, primary in zip(
            forms,
            accessions,
            acceptance,
            primary_docs,
            strict=True,
        ):
            if str(form) not in FORM4_TYPES:
                continue
            try:
                filed_at = _parse_acceptance(str(accepted))
            except ValueError as exc:
                msg = f"bad acceptanceDateTime for {accession}: {exc}"
                raise DataUnavailableError(msg) from exc
            if filed_at < start or filed_at > end:
                continue
            jobs.append((str(accession), str(primary), filed_at))

        def _one(job: tuple[str, str, datetime]) -> tuple[Form4Record, ...]:
            accession, primary, filed_at = job
            try:
                xml_bytes = self._load_filing_xml(cik, accession, primary)
                return parse_form4_xml(
                    xml_bytes,
                    symbol=symbol,
                    cik=cik,
                    accession=accession,
                    filed_at=filed_at,
                )
            except DataUnavailableError:
                return ()

        out: list[Form4Record] = []
        # Saturate the token bucket (~max_rps) without exceeding SEC fair-access.
        workers = 1  # cross-symbol pool saturates rate limit
        if len(jobs) <= 1:
            for job in jobs:
                out.extend(_one(job))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_one, job) for job in jobs]
                for fut in as_completed(futures):
                    out.extend(fut.result())
        return tuple(out)


    def _load_ticker_map(self) -> dict[str, str]:
        path = self._cache_dir / "company_tickers.json"
        raw = self._read_cache_or_fetch(path, TICKERS_URL)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            msg = "company_tickers.json is not valid JSON"
            raise DataUnavailableError(msg) from exc
        mapping: dict[str, str] = {}
        if isinstance(payload, dict):
            rows = payload.values()
        elif isinstance(payload, list):
            rows = payload
        else:
            msg = "unexpected company_tickers.json shape"
            raise DataUnavailableError(msg)
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "")).upper().strip()
            cik_raw = row.get("cik_str", row.get("cik"))
            if not ticker or cik_raw is None:
                continue
            mapping[ticker] = f"{int(cik_raw):010d}"
        if not mapping:
            msg = "company_tickers.json produced an empty ticker map"
            raise DataUnavailableError(msg)
        return mapping

    def _load_submissions(self, cik: str) -> dict[str, object]:
        padded = f"{int(cik):010d}"
        path = self._cache_dir / "submissions" / f"CIK{padded}.json"
        url = f"{DATA_SEC_BASE}/submissions/CIK{padded}.json"
        raw = self._read_cache_or_fetch(path, url)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"submissions JSON invalid for CIK {padded}"
            raise DataUnavailableError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"submissions payload must be an object for CIK {padded}"
            raise DataUnavailableError(msg)
        return payload

    def _load_filing_xml(self, cik: str, accession: str, primary: str) -> bytes:
        accession_nodash = accession.replace("-", "")
        cik_int = str(int(cik))
        # primaryDocument is often "xslF345X05/form4.xml" (HTML viewer). Prefer the
        # ownership XML basename at the accession root (e.g. form4.xml).
        candidates: list[str] = []
        primary = primary.strip().lstrip("/")
        basename = primary.split("/")[-1]
        if basename and basename != primary:
            candidates.append(basename)
        if primary:
            candidates.append(primary)
        if "form4.xml" not in candidates:
            candidates.append("form4.xml")
        last_error: Exception | None = None
        for name in candidates:
            safe_name = name.replace("/", "_")
            path = self._cache_dir / "filings" / accession_nodash / safe_name
            url = f"{ARCHIVES_BASE}/{cik_int}/{accession_nodash}/{name}"
            try:
                body = self._read_cache_or_fetch(path, url)
            except DataUnavailableError as exc:
                last_error = exc
                continue
            head = body.lstrip()[:80].lower()
            if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
                # Cached/served XSL HTML wrapper — try next candidate.
                continue
            lowered = body[:800].lower()
            if b"ownershipdocument" not in lowered:
                # Not ownership XML; keep looking.
                continue
            return body
        if last_error is not None:
            raise last_error
        msg = f"no ownership XML found for accession {accession}"
        raise DataUnavailableError(msg)

    def _read_cache_or_fetch(self, path: Path, url: str) -> bytes:
        if path.is_file():
            return path.read_bytes()
        path_s = str(path).replace(chr(92), "/")
        if self._cache_only or (
            self._skip_uncached_filings and "/filings/" in path_s
        ):
            msg = f"cache miss (skip_uncached): {path}"
            raise DataUnavailableError(msg)
        body = self._http_get(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return body

    def _default_http_get(self, url: str) -> bytes:
        self._bucket.acquire()
        request = Request(
            url,
            headers={
                "User-Agent": self._settings.user_agent,
                "Accept-Encoding": "identity",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=60) as response:  # noqa: S310 - official SEC hosts only
                return response.read()
        except HTTPError as exc:
            msg = f"SEC HTTP {exc.code} for {url}"
            raise DataUnavailableError(msg) from exc
        except URLError as exc:
            msg = f"SEC unreachable for {url}: {exc.reason}"
            raise DataUnavailableError(msg) from exc
