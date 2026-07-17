"""Combinatorial Purged Cross-Validation (CPCV) without mlfinlab.

Unique ``timestamp`` levels of a ``(timestamp, symbol)`` MultiIndex are split
into ``N`` contiguous blocks. Every combination of ``k`` test blocks yields one
fold. Training rows whose label window ``[t0, t1]`` overlaps any test timestamp
are purged; an additional trading-bar embargo of ``embargo_days`` unique
timestamps after each contiguous test segment is also removed from training.

DataFrame / index schema
------------------------
Input dataset (via ``BaselineDataset``):
    index : MultiIndex[timestamp (tz-aware UTC), symbol]
    features / label : float64, no NaNs, identical index

Label end times ``t1``:
    For horizon ``h``, ``t1`` is the ``h``-th subsequent trading timestamp of the
    same symbol (matching ``make_forward_return_labels``). Callers should pass
    ``bar_timestamps`` — the full trading calendar *before* the final-horizon
    feature drop — so ``t1`` resolves on real bars (including weekends/holidays
    after the last labeled row). When that bar is missing, the row is
    hard-failed — never extrapolated via median gaps.

Output ``CPCVFold``:
    ``train_positions`` / ``test_positions`` are ``iloc`` positions into the
    input dataset row order (deterministic sort by the MultiIndex).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

from aihedgefund.core.schemas import BaselineDataset, CPCVConfig, CPCVFold, CPCVSplitResult


def combinatorial_purged_splits(
    dataset: BaselineDataset,
    config: CPCVConfig,
    *,
    bar_timestamps: pd.DatetimeIndex | None = None,
) -> CPCVSplitResult:
    """Enumerate purged+embargoed CPCV folds for ``dataset``.

    Deterministic: block boundaries follow sorted unique timestamps; fold order
    follows ``itertools.combinations`` on block ids ``0..N-1``.

    ``bar_timestamps`` is the full trading calendar used to resolve label end
    times ``t1`` (typically all bar timestamps before dropping the final
    ``horizon`` rows). When omitted, ``t1`` is resolved against the feature
    index alone and hard-fails if ``ts[i + horizon]`` is absent.
    """
    if config.horizon != dataset.horizon:
        msg = (
            f"CPCVConfig.horizon ({config.horizon}) must equal "
            f"dataset.horizon ({dataset.horizon})"
        )
        raise ValueError(msg)
    if config.embargo_days < config.horizon:
        msg = "embargo_days must be >= horizon"
        raise ValueError(msg)

    features = dataset.features.sort_index()
    if not features.index.equals(dataset.features.index):
        dataset = BaselineDataset(
            features=features,
            label=dataset.label.loc[features.index],
            horizon=dataset.horizon,
            feature_columns=dataset.feature_columns,
        )
    else:
        features = dataset.features

    if list(features.index.names) != ["timestamp", "symbol"]:
        msg = "dataset index names must be ('timestamp', 'symbol')"
        raise ValueError(msg)

    timestamps = pd.DatetimeIndex(
        features.index.get_level_values("timestamp").unique()
    ).sort_values()
    n_timestamps = len(timestamps)
    if n_timestamps < config.n_blocks:
        msg = (
            f"need at least n_blocks={config.n_blocks} unique timestamps; "
            f"got {n_timestamps}"
        )
        raise ValueError(msg)

    block_ids = _assign_contiguous_blocks(n_timestamps, config.n_blocks)
    t1_by_row = _label_end_times(
        features.index,
        horizon=config.horizon,
        bar_timestamps=bar_timestamps,
    )
    row_timestamps = pd.DatetimeIndex(features.index.get_level_values("timestamp"))

    folds: list[CPCVFold] = []
    for fold_id, test_blocks in enumerate(
        combinations(range(config.n_blocks), config.n_test_blocks)
    ):
        test_block_set = set(test_blocks)
        test_ts_mask = np.isin(block_ids, list(test_block_set))
        test_timestamps = timestamps[test_ts_mask]
        if len(test_timestamps) == 0:
            msg = f"fold {fold_id}: empty test timestamp set"
            raise ValueError(msg)

        is_test_row = np.asarray(row_timestamps.isin(test_timestamps), dtype=bool)
        test_positions = tuple(int(i) for i in np.flatnonzero(is_test_row))
        if not test_positions:
            msg = f"fold {fold_id}: no test rows"
            raise ValueError(msg)

        candidate_train = ~is_test_row
        purged = _purge_mask(
            row_timestamps=row_timestamps,
            t1_by_row=t1_by_row,
            timestamps=timestamps,
            block_ids=block_ids,
            test_block_ids=test_block_set,
            candidate_train=candidate_train,
        )
        purged_count = int(purged.sum())
        after_purge = candidate_train & ~purged

        embargoed = _embargo_mask(
            row_timestamps=row_timestamps,
            timestamps=timestamps,
            block_ids=block_ids,
            test_block_ids=test_block_set,
            embargo_days=config.embargo_days,
            candidate_train=after_purge,
        )
        embargoed_count = int(embargoed.sum())
        train_mask = after_purge & ~embargoed
        train_positions = tuple(int(i) for i in np.flatnonzero(train_mask))
        if not train_positions:
            msg = f"fold {fold_id}: train set empty after purge/embargo"
            raise ValueError(msg)

        folds.append(
            CPCVFold(
                fold_id=fold_id,
                test_block_ids=tuple(int(b) for b in test_blocks),
                train_positions=train_positions,
                test_positions=test_positions,
                purged_train_count=purged_count,
                embargoed_train_count=embargoed_count,
            )
        )

    expected = comb(config.n_blocks, config.n_test_blocks)
    if len(folds) != expected:
        msg = f"expected {expected} folds, got {len(folds)}"
        raise RuntimeError(msg)

    return CPCVSplitResult(
        config=config,
        n_folds=len(folds),
        n_timestamps=n_timestamps,
        folds=tuple(folds),
    )


def subset_by_positions(dataset: BaselineDataset, positions: Sequence[int]) -> BaselineDataset:
    """Return a ``BaselineDataset`` restricted to the given ``iloc`` positions."""
    if not positions:
        msg = "positions must be non-empty"
        raise ValueError(msg)
    idx = list(positions)
    features = dataset.features.iloc[idx]
    label = dataset.label.iloc[idx]
    return BaselineDataset(
        features=features,
        label=label,
        horizon=dataset.horizon,
        feature_columns=dataset.feature_columns,
    )


def _assign_contiguous_blocks(n_timestamps: int, n_blocks: int) -> np.ndarray:
    """Map each timestamp position to a contiguous block id in ``0..n_blocks-1``."""
    base, rem = divmod(n_timestamps, n_blocks)
    if base < 1:
        msg = "each CPCV block must contain at least one timestamp"
        raise ValueError(msg)
    sizes = [base + (1 if i < rem else 0) for i in range(n_blocks)]
    block_ids = np.repeat(np.arange(n_blocks, dtype=int), sizes)
    if len(block_ids) != n_timestamps:
        msg = "internal error: block assignment length mismatch"
        raise RuntimeError(msg)
    return block_ids


def _label_end_times(
    index: pd.MultiIndex,
    *,
    horizon: int,
    bar_timestamps: pd.DatetimeIndex | None = None,
) -> pd.DatetimeIndex:
    """Compute ``t1`` per row: the ``horizon``-th next bar of the same symbol.

    When ``bar_timestamps`` is provided, ``t1`` is taken from that trading
    calendar (the bar index before dropping incomplete forward windows).
    Otherwise the feature-index timestamps are used. Missing
    ``calendar[i + horizon]`` is always a hard ``ValueError`` — never a
    median-gap extrapolation.
    """
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)

    shared_calendar: pd.DatetimeIndex | None = None
    if bar_timestamps is not None:
        shared_calendar = pd.DatetimeIndex(bar_timestamps).unique().sort_values()
        if len(shared_calendar) == 0:
            msg = "bar_timestamps must be non-empty"
            raise ValueError(msg)

    t1_maps: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
    for symbol in index.get_level_values("symbol").unique():
        symbol_key = str(symbol)
        feature_ts = pd.DatetimeIndex(
            index.get_level_values("timestamp")[
                index.get_level_values("symbol") == symbol
            ]
        ).unique().sort_values()
        if len(feature_ts) < 1:
            msg = f"{symbol}: need at least 1 feature timestamp"
            raise ValueError(msg)
        calendar = shared_calendar if shared_calendar is not None else feature_ts
        pos_by_ts = {pd.Timestamp(ts): i for i, ts in enumerate(calendar)}
        mapping: dict[pd.Timestamp, pd.Timestamp] = {}
        for stamp in feature_ts:
            key = pd.Timestamp(stamp)
            try:
                start_i = pos_by_ts[key]
            except KeyError as exc:
                msg = f"{symbol}: feature timestamp {key} missing from bar calendar"
                raise ValueError(msg) from exc
            end_i = start_i + horizon
            if end_i >= len(calendar):
                msg = (
                    f"{symbol}: missing t1 bar for timestamp {key} "
                    f"(need calendar[{start_i}+{horizon}], calendar length "
                    f"{len(calendar)})"
                )
                raise ValueError(msg)
            mapping[key] = pd.Timestamp(calendar[end_i])
        t1_maps[symbol_key] = mapping

    t1_ordered: list[pd.Timestamp] = []
    for ts, symbol in zip(
        index.get_level_values("timestamp"),
        index.get_level_values("symbol"),
        strict=True,
    ):
        stamp = pd.Timestamp(ts)
        try:
            t1_ordered.append(t1_maps[str(symbol)][stamp])
        except KeyError as exc:
            msg = f"{symbol}: missing label end for timestamp {stamp}"
            raise ValueError(msg) from exc
    return pd.DatetimeIndex(t1_ordered)


def _purge_mask(
    *,
    row_timestamps: pd.DatetimeIndex,
    t1_by_row: pd.DatetimeIndex,
    timestamps: pd.DatetimeIndex,
    block_ids: np.ndarray,
    test_block_ids: set[int],
    candidate_train: np.ndarray,
) -> np.ndarray:
    """True where a candidate train row's ``[t0, t1]`` overlaps a test run.

    Overlap is evaluated per contiguous run of test blocks (not the span that
    would bridge non-adjacent test blocks — that would incorrectly purge the
    train blocks sitting between them).
    """
    purged = np.zeros(len(row_timestamps), dtype=bool)
    is_test_ts = np.isin(block_ids, list(test_block_ids))
    for start, end in _contiguous_true_runs(is_test_ts):
        run_ts = timestamps[start:end]
        test_min = run_ts.min()
        test_max = run_ts.max()
        overlaps = np.asarray(
            (row_timestamps <= test_max) & (t1_by_row >= test_min),
            dtype=bool,
        )
        purged |= overlaps
    return candidate_train & purged


def _embargo_mask(
    *,
    row_timestamps: pd.DatetimeIndex,
    timestamps: pd.DatetimeIndex,
    block_ids: np.ndarray,
    test_block_ids: set[int],
    embargo_days: int,
    candidate_train: np.ndarray,
) -> np.ndarray:
    """True where a candidate train row falls in a post-test embargo zone.

    For each maximal contiguous run of test blocks, the embargo covers the
    next ``embargo_days`` unique trading timestamps after ``test_end``
    (iloc-slice on the sorted timestamp calendar), not calendar days.
    """
    if embargo_days < 0:
        msg = "embargo_days must be >= 0"
        raise ValueError(msg)

    embargo = np.zeros(len(row_timestamps), dtype=bool)
    if embargo_days == 0:
        return embargo

    is_test_ts = np.isin(block_ids, list(test_block_ids))
    for start, end in _contiguous_true_runs(is_test_ts):
        _ = start
        # ``end`` is the exclusive index of the test run in ``timestamps``.
        embargo_start = end
        embargo_stop = min(end + embargo_days, len(timestamps))
        if embargo_start >= embargo_stop:
            continue
        embargo_ts = timestamps[embargo_start:embargo_stop]
        in_zone = np.asarray(row_timestamps.isin(embargo_ts), dtype=bool)
        embargo |= in_zone
    return candidate_train & embargo


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` runs where ``mask`` is True."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]
