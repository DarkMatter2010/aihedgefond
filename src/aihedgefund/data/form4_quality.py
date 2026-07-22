"""Form 4 integrity checks: hard-fail on corruption, never on missing activity."""

from __future__ import annotations

from datetime import datetime

from aihedgefund.core.bus import MessageBus
from aihedgefund.core.runtime import Clock, SystemClock
from aihedgefund.core.schemas import (
    Form4Frame,
    Form4Record,
    QualityFailure,
    QualityGateFailed,
)


class Form4QualityError(RuntimeError):
    """Raised when Form 4 rows fail integrity checks (source/corruption)."""


class Form4QualityGate:
    """Validate Form 4 records; empty activity is allowed (neutral, not an error)."""

    def __init__(
        self,
        bus: MessageBus,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock or SystemClock()

    def validate(
        self,
        frame: Form4Frame,
        *,
        now: datetime | None = None,
    ) -> Form4Frame:
        """Return ``frame`` unchanged when every present record is well-formed.

        Missing filings for a symbol are normal and must not hard-fail. Future
        ``filed_at`` vs ``now``, invalid A/D codes, or non-finite shares hard-fail.
        """
        checked_at = now or self._clock.now()
        try:
            self._validate(frame, checked_at)
        except Form4QualityError as exc:
            self._bus.publish_event(
                QualityGateFailed(
                    timestamp=checked_at,
                    payload=QualityFailure(symbol="FORM4", reason=str(exc)),
                )
            )
            raise
        except Exception as exc:
            failure = Form4QualityError(f"form4 quality evaluation failed: {exc}")
            self._bus.publish_event(
                QualityGateFailed(
                    timestamp=checked_at,
                    payload=QualityFailure(symbol="FORM4", reason=str(failure)),
                )
            )
            raise failure from exc
        return frame

    def _validate(self, frame: Form4Frame, checked_at: datetime) -> None:
        seen: set[tuple[object, ...]] = set()
        for record in frame.records:
            self._validate_record(record, checked_at)
            key = (
                record.accession,
                record.symbol,
                record.transaction_date,
                record.reporting_owner,
                record.transaction_code,
                float(record.shares),
                record.acquired_disposed,
            )
            if key in seen:
                msg = f"duplicate form4 key {key}"
                raise Form4QualityError(msg)
            seen.add(key)

    @staticmethod
    def _validate_record(record: Form4Record, checked_at: datetime) -> None:
        if record.filed_at > checked_at:
            msg = (
                f"{record.symbol} filed_at {record.filed_at.isoformat()} "
                f"is after quality clock {checked_at.isoformat()}"
            )
            raise Form4QualityError(msg)
        if record.acquired_disposed not in {"A", "D"}:
            msg = f"{record.symbol} acquired_disposed must be A or D"
            raise Form4QualityError(msg)
        if record.shares < 0.0 or record.shares != record.shares:
            msg = f"{record.symbol} shares must be finite and >= 0"
            raise Form4QualityError(msg)
        if not record.accession.strip():
            msg = f"{record.symbol} accession is empty"
            raise Form4QualityError(msg)
        if not record.cik.strip():
            msg = f"{record.symbol} cik is empty"
            raise Form4QualityError(msg)
