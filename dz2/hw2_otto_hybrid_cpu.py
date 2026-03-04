#!/usr/bin/env python3
"""OTTO HW2: Hybrid fast CPU pipeline for submission generation."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

TYPE_MAP = {"clicks": 0, "carts": 1, "orders": 2}
TYPE_WEIGHT = {0: 1.0, 1: 6.0, 2: 3.0}
TYPE_SIGNAL = {0: 1.0, 1: 1.3, 2: 1.6}
TARGETS = ("clicks", "carts", "orders")
REQUIRED_COLUMNS = ["session", "aid", "ts", "type"]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Expected positive int, got {value}")
    return ivalue


def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative int, got {value}")
    return ivalue


def zero_one_int(value: str) -> int:
    ivalue = int(value)
    if ivalue not in (0, 1):
        raise argparse.ArgumentTypeError(f"Expected 0 or 1, got {value}")
    return ivalue


def dedupe_keep_order(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(int(value))
    return out


def unique_recent(aids: Sequence[int], limit: int) -> List[int]:
    out: List[int] = []
    seen = set()
    for aid in reversed(aids):
        aid_int = int(aid)
        if aid_int in seen:
            continue
        seen.add(aid_int)
        out.append(aid_int)
        if len(out) >= limit:
            break
    return out


def normalize_type_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int8")
    mapped = series.map(TYPE_MAP)
    if mapped.isna().any():
        invalid = series[mapped.isna()].dropna().unique()[:5]
        raise ValueError(f"Unknown event type values: {invalid}")
    return mapped.astype("int8")


def load_parquet_files(root: Path, part_name: str, max_files: int) -> List[Path]:
    part_dir = root / part_name
    if not part_dir.exists():
        raise FileNotFoundError(f"Directory not found: {part_dir}")
    files = sorted(part_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in: {part_dir}")
    if max_files > 0:
        files = files[:max_files]
    return files


def load_events_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=REQUIRED_COLUMNS)
    df = df[REQUIRED_COLUMNS]
    df["session"] = df["session"].astype("int64")
    df["aid"] = df["aid"].astype("int64")
    df["ts"] = (df["ts"].astype("int64") // 1000).astype("int64")
    df["type"] = normalize_type_column(df["type"])
    df = df.sort_values(["session", "ts"], kind="mergesort")
    return df


def prune_matrix(matrix: DefaultDict[int, Counter], keep_n: int) -> None:
    for src in list(matrix.keys()):
        counter = matrix[src]
        if not counter:
            del matrix[src]
            continue
        if len(counter) > keep_n:
            matrix[src] = Counter(dict(counter.most_common(keep_n)))


def finalize_matrix(matrix: DefaultDict[int, Counter], topk: int) -> Dict[int, List[Tuple[int, float]]]:
    out: Dict[int, List[Tuple[int, float]]] = {}
    for src, counter in matrix.items():
        if not counter:
            continue
        out[int(src)] = [(int(dst), float(w)) for dst, w in counter.most_common(topk)]
    return out


class CovisitationBuilder:
    def __init__(
        self,
        topk: int,
        n_recent: int,
        pair_lookback: int,
        session_tail: int,
        prune_factor: int,
    ) -> None:
        self.topk = topk
        self.n_recent = n_recent
        self.pair_lookback = pair_lookback
        self.session_tail = session_tail
        self.prune_keep = max(topk * prune_factor, topk)

        self.click_click: DefaultDict[int, Counter] = defaultdict(Counter)
        self.carts_orders: DefaultDict[int, Counter] = defaultdict(Counter)
        self.buy2buy: DefaultDict[int, Counter] = defaultdict(Counter)
        self.item_popularity: Counter = Counter()
        self.buy_popularity: Counter = Counter()

        self.click_window = 24 * 60 * 60
        self.carts_orders_window = 24 * 60 * 60
        self.buy2buy_window = 14 * 24 * 60 * 60

    def process_session(self, aids: Sequence[int], ts: Sequence[int], types: Sequence[int]) -> None:
        n = len(aids)
        if n == 0:
            return

        start = max(0, n - self.session_tail)
        aids = aids[start:]
        ts = ts[start:]
        types = types[start:]
        n = len(aids)

        for aid, event_type in zip(aids, types):
            aid_i = int(aid)
            type_i = int(event_type)
            self.item_popularity[aid_i] += 1
            if type_i in (1, 2):
                self.buy_popularity[aid_i] += 1

        if n < 2:
            return

        for i in range(n):
            aid_i = int(aids[i])
            type_i = int(types[i])
            ts_i = int(ts[i])

            left = max(0, i - self.pair_lookback)
            for j in range(left, i):
                aid_j = int(aids[j])
                if aid_i == aid_j:
                    continue
                type_j = int(types[j])
                ts_j = int(ts[j])
                delta = abs(ts_i - ts_j)

                if delta <= self.click_window:
                    # CPU-friendly time-weighting approximation.
                    weight_click = 1.0 + 3.0 / (1.0 + delta / 3600.0)
                    self.click_click[aid_j][aid_i] += weight_click
                    self.click_click[aid_i][aid_j] += weight_click

                if delta <= self.carts_orders_window:
                    self.carts_orders[aid_j][aid_i] += TYPE_WEIGHT.get(type_i, 1.0)
                    self.carts_orders[aid_i][aid_j] += TYPE_WEIGHT.get(type_j, 1.0)

                if type_i in (1, 2) and type_j in (1, 2) and delta <= self.buy2buy_window:
                    self.buy2buy[aid_j][aid_i] += 1.0
                    self.buy2buy[aid_i][aid_j] += 1.0

    def prune(self) -> None:
        prune_matrix(self.click_click, self.prune_keep)
        prune_matrix(self.carts_orders, self.prune_keep)
        prune_matrix(self.buy2buy, self.prune_keep)

    def finalize(self) -> Dict[str, object]:
        return {
            "click_click": finalize_matrix(self.click_click, self.topk),
            "carts_orders": finalize_matrix(self.carts_orders, self.topk),
            "buy2buy": finalize_matrix(self.buy2buy, self.topk),
            "popular_general": [int(aid) for aid, _ in self.item_popularity.most_common(max(self.topk * 50, 1000))],
            "popular_buy": [int(aid) for aid, _ in self.buy_popularity.most_common(max(self.topk * 50, 1000))],
            "meta": {
                "topk": self.topk,
                "n_recent": self.n_recent,
                "pair_lookback": self.pair_lookback,
                "session_tail": self.session_tail,
            },
        }


def build_covisit_cache(train_files: Sequence[Path], args: argparse.Namespace) -> Dict[str, object]:
    builder = CovisitationBuilder(
        topk=args.topk,
        n_recent=args.n_recent,
        pair_lookback=args.pair_lookback,
        session_tail=args.session_tail,
        prune_factor=args.prune_factor,
    )

    total_sessions = 0
    limit_sessions = args.train_sessions_limit if args.train_sessions_limit > 0 else None

    for file_idx, file_path in enumerate(train_files, start=1):
        log(f"Building co-visitation from train file {file_idx}/{len(train_files)}: {file_path.name}")
        df = load_events_file(file_path)

        for session_id, g in df.groupby("session", sort=False):
            builder.process_session(
                aids=g["aid"].to_numpy(copy=False),
                ts=g["ts"].to_numpy(copy=False),
                types=g["type"].to_numpy(copy=False),
            )
            total_sessions += 1

            if total_sessions % 50000 == 0:
                log(f"Processed train sessions: {total_sessions}")

            if limit_sessions is not None and total_sessions >= limit_sessions:
                break

        builder.prune()
        if limit_sessions is not None and total_sessions >= limit_sessions:
            break

    log(f"Total processed train sessions for co-visitation: {total_sessions}")
    return builder.finalize()


def load_or_build_cache(train_files: Sequence[Path], cache_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    if cache_path.exists() and args.rebuild_covisit == 0:
        log(f"Loading co-visitation cache: {cache_path}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    log("Computing co-visitation cache from train data")
    cache = build_covisit_cache(train_files, args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Saved co-visitation cache: {cache_path}")
    return cache


def aggregate_covisit_scores(
    source_weights: Dict[int, float],
    matrix: Dict[int, List[Tuple[int, float]]],
) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    for source_aid, src_weight in source_weights.items():
        for target_aid, weight in matrix.get(source_aid, ()): 
            scores[target_aid] += src_weight * weight
    return scores


def build_session_context(aids: Sequence[int], types: Sequence[int], n_recent: int) -> Dict[str, object]:
    context: Dict[str, object] = {}
    context["recent_unique"] = unique_recent(aids, n_recent)

    freq_counter = Counter(int(aid) for aid in aids)
    last_pos: Dict[int, int] = {}
    last_type: Dict[int, int] = {}
    for idx, (aid, event_type) in enumerate(zip(aids, types)):
        aid_i = int(aid)
        last_pos[aid_i] = idx
        last_type[aid_i] = int(event_type)

    source_weights: Dict[int, float] = {}
    recent_unique: List[int] = context["recent_unique"]  # type: ignore[assignment]
    for rank, aid in enumerate(recent_unique):
        event_type = last_type.get(aid, 0)
        base = 1.0 / (rank + 1.0)
        source_weights[aid] = base * TYPE_SIGNAL.get(event_type, 1.0)

    seq_sources = recent_unique[:5]
    context["freq_counter"] = freq_counter
    context["last_pos"] = last_pos
    context["last_type"] = last_type
    context["source_weights"] = source_weights
    context["seq_sources"] = seq_sources
    context["session_len"] = len(aids)
    return context


def generate_candidates(
    recent_unique: Sequence[int],
    click_click: Dict[int, List[Tuple[int, float]]],
    carts_orders: Dict[int, List[Tuple[int, float]]],
    buy2buy: Dict[int, List[Tuple[int, float]]],
    popular_general: Sequence[int],
    max_candidates: int,
) -> List[int]:
    raw: List[int] = list(recent_unique)
    for aid in recent_unique:
        raw.extend(dst for dst, _ in click_click.get(int(aid), ()))
        raw.extend(dst for dst, _ in carts_orders.get(int(aid), ()))
        raw.extend(dst for dst, _ in buy2buy.get(int(aid), ()))

    if len(raw) < max_candidates:
        raw.extend(popular_general[:max_candidates])

    return dedupe_keep_order(raw)[:max_candidates]


def rank_for_target(
    target: str,
    candidates: Sequence[int],
    context: Dict[str, object],
    cov_click: Dict[int, float],
    cov_carts_orders: Dict[int, float],
    cov_buy2buy: Dict[int, float],
    seq_scores: Dict[int, float],
    pop_rank_general: Dict[int, int],
    pop_rank_buy: Dict[int, int],
    topk: int,
) -> List[int]:
    freq_counter: Counter = context["freq_counter"]  # type: ignore[assignment]
    last_pos: Dict[int, int] = context["last_pos"]  # type: ignore[assignment]
    last_type: Dict[int, int] = context["last_type"]  # type: ignore[assignment]
    session_length = max(1, int(context["session_len"]))  # type: ignore[arg-type]

    weights = {
        "clicks": {
            "freq": 1.2,
            "recency": 1.8,
            "type": 0.7,
            "cov_click": 1.0,
            "cov_cart": 0.5,
            "cov_buy": 0.2,
            "seq": 1.4,
            "pop": 0.05,
        },
        "carts": {
            "freq": 1.4,
            "recency": 1.1,
            "type": 1.2,
            "cov_click": 0.6,
            "cov_cart": 1.2,
            "cov_buy": 0.9,
            "seq": 1.0,
            "pop": 0.04,
        },
        "orders": {
            "freq": 1.0,
            "recency": 1.0,
            "type": 1.8,
            "cov_click": 0.3,
            "cov_cart": 1.0,
            "cov_buy": 1.5,
            "seq": 1.1,
            "pop": 0.03,
        },
    }[target]

    order_index = {aid: idx for idx, aid in enumerate(candidates)}
    score_map: Dict[int, float] = {}

    for aid in candidates:
        f = freq_counter.get(aid, 0)
        history_freq = math.log1p(f)

        if aid in last_pos:
            gap = session_length - 1 - last_pos[aid]
            recency = math.exp(-gap / 3.0)
            item_type_signal = TYPE_SIGNAL.get(last_type.get(aid, 0), 1.0)
        else:
            recency = 0.0
            item_type_signal = 0.5

        score = (
            weights["freq"] * history_freq
            + weights["recency"] * recency
            + weights["type"] * item_type_signal
            + weights["cov_click"] * cov_click.get(aid, 0.0)
            + weights["cov_cart"] * cov_carts_orders.get(aid, 0.0)
            + weights["cov_buy"] * cov_buy2buy.get(aid, 0.0)
            + weights["seq"] * seq_scores.get(aid, 0.0)
        )

        if target in ("carts", "orders"):
            rank = pop_rank_buy.get(aid)
        else:
            rank = pop_rank_general.get(aid)

        if rank is not None:
            score += weights["pop"] * (1.0 / (rank + 1.0))

        score_map[aid] = score

    ranked = sorted(candidates, key=lambda aid: (score_map[aid], -order_index[aid]), reverse=True)
    return ranked[:topk]


def recommend_for_session(
    aids: Sequence[int],
    types: Sequence[int],
    cache: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, List[int]]:
    click_click: Dict[int, List[Tuple[int, float]]] = cache["click_click"]  # type: ignore[assignment]
    carts_orders: Dict[int, List[Tuple[int, float]]] = cache["carts_orders"]  # type: ignore[assignment]
    buy2buy: Dict[int, List[Tuple[int, float]]] = cache["buy2buy"]  # type: ignore[assignment]
    popular_general: List[int] = cache["popular_general"]  # type: ignore[assignment]
    popular_buy: List[int] = cache["popular_buy"]  # type: ignore[assignment]

    context = build_session_context(aids, types, args.n_recent)
    recent_unique: List[int] = context["recent_unique"]  # type: ignore[assignment]
    source_weights: Dict[int, float] = context["source_weights"]  # type: ignore[assignment]
    seq_sources: List[int] = context["seq_sources"]  # type: ignore[assignment]

    candidates = generate_candidates(
        recent_unique=recent_unique,
        click_click=click_click,
        carts_orders=carts_orders,
        buy2buy=buy2buy,
        popular_general=popular_general,
        max_candidates=args.max_candidates,
    )

    cov_click = aggregate_covisit_scores(source_weights, click_click)
    cov_carts_orders = aggregate_covisit_scores(source_weights, carts_orders)
    cov_buy2buy = aggregate_covisit_scores(source_weights, buy2buy)

    seq_weights = {aid: 1.0 / (idx + 1.0) for idx, aid in enumerate(seq_sources)}
    seq_click = aggregate_covisit_scores(seq_weights, click_click)
    seq_buy = aggregate_covisit_scores(seq_weights, buy2buy)
    seq_scores: Dict[int, float] = defaultdict(float)
    for aid, value in seq_click.items():
        seq_scores[aid] += value
    for aid, value in seq_buy.items():
        seq_scores[aid] += 1.3 * value

    pop_rank_general = {aid: rank for rank, aid in enumerate(popular_general)}
    pop_rank_buy = {aid: rank for rank, aid in enumerate(popular_buy)}

    predictions: Dict[str, List[int]] = {}
    for target in TARGETS:
        ranked = rank_for_target(
            target=target,
            candidates=candidates,
            context=context,
            cov_click=cov_click,
            cov_carts_orders=cov_carts_orders,
            cov_buy2buy=cov_buy2buy,
            seq_scores=seq_scores,
            pop_rank_general=pop_rank_general,
            pop_rank_buy=pop_rank_buy,
            topk=args.topk,
        )

        fallback = popular_buy if target in ("carts", "orders") else popular_general
        if len(ranked) < args.topk:
            ranked = dedupe_keep_order(ranked + fallback[: args.topk * 2])[: args.topk]

        predictions[target] = ranked

    return predictions


def parse_labels(labels: str) -> List[int]:
    if not isinstance(labels, str):
        raise ValueError("Labels must be a string")
    parts = [part for part in labels.strip().split(" ") if part]
    if not parts:
        raise ValueError("Labels string is empty")
    aids = [int(part) for part in parts]
    return aids


def validate_submission(path: Path, topk: int, expected_sessions: int | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    df = pd.read_csv(path)
    expected_columns = ["session_type", "labels"]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Invalid columns: expected {expected_columns}, got {list(df.columns)}")

    if df.empty:
        raise ValueError("Submission is empty")

    if expected_sessions is not None:
        expected_rows = expected_sessions * 3
        if len(df) != expected_rows:
            raise ValueError(f"Invalid row count: expected {expected_rows}, got {len(df)}")

    seen_session_type = set()
    per_session_types: Dict[str, set] = defaultdict(set)

    for row in df.itertuples(index=False):
        session_type = str(row.session_type)
        labels = row.labels

        if session_type in seen_session_type:
            raise ValueError(f"Duplicate session_type row: {session_type}")
        seen_session_type.add(session_type)

        if "_" not in session_type:
            raise ValueError(f"Invalid session_type format: {session_type}")

        session_id, target = session_type.rsplit("_", 1)
        if target not in TARGETS:
            raise ValueError(f"Invalid target suffix in session_type: {session_type}")
        if not session_id.isdigit():
            raise ValueError(f"Session id is not numeric in session_type: {session_type}")

        aids = parse_labels(labels)
        if not (1 <= len(aids) <= topk):
            raise ValueError(f"Row {session_type} has {len(aids)} labels, expected 1..{topk}")

        if len(set(aids)) != len(aids):
            raise ValueError(f"Row {session_type} contains duplicate aids")

        per_session_types[session_id].add(target)

    missing_targets = [sid for sid, tset in per_session_types.items() if set(TARGETS) != tset]
    if missing_targets:
        raise ValueError(f"Some sessions do not contain all 3 targets, examples: {missing_targets[:5]}")

    log(f"Validation passed for {path} ({len(per_session_types)} sessions, {len(df)} rows)")


def generate_submission(
    test_files: Sequence[Path],
    cache: Dict[str, object],
    out_path: Path,
    args: argparse.Namespace,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written_sessions = 0
    limit_sessions = args.smoke_sessions if args.smoke_sessions > 0 else None

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_type", "labels"])

        for file_idx, file_path in enumerate(test_files, start=1):
            log(f"Scoring test file {file_idx}/{len(test_files)}: {file_path.name}")
            df = load_events_file(file_path)

            for session_id, g in df.groupby("session", sort=False):
                aids = g["aid"].to_numpy(copy=False)
                types = g["type"].to_numpy(copy=False)

                preds = recommend_for_session(aids=aids, types=types, cache=cache, args=args)

                for target in TARGETS:
                    labels = " ".join(str(aid) for aid in preds[target])
                    writer.writerow([f"{int(session_id)}_{target}", labels])

                written_sessions += 1
                if written_sessions % 10000 == 0:
                    log(f"Scored sessions: {written_sessions}")

                if limit_sessions is not None and written_sessions >= limit_sessions:
                    break

            if limit_sessions is not None and written_sessions >= limit_sessions:
                break

    if written_sessions == 0:
        raise ValueError("No sessions were scored, submission was not generated")

    return written_sessions


def run_self_checks() -> None:
    deduped = dedupe_keep_order([1, 2, 1, 3, 2, 4])
    assert deduped == [1, 2, 3, 4], "dedupe_keep_order failed"

    parsed = parse_labels("10 20 30")
    assert parsed == [10, 20, 30], "parse_labels failed"

    cache_stub = {
        "click_click": {},
        "carts_orders": {},
        "buy2buy": {},
        "popular_general": [101, 102, 103, 104],
        "popular_buy": [201, 202, 203, 204],
    }

    class ArgsStub:
        n_recent = 20
        max_candidates = 120
        topk = 3

    preds = recommend_for_session(
        aids=[1, 1],
        types=[0, 0],
        cache=cache_stub,
        args=ArgsStub,
    )
    assert len(preds["clicks"]) == 3, "fallback for clicks failed"
    assert len(preds["carts"]) == 3, "fallback for carts failed"
    assert len(preds["orders"]) == 3, "fallback for orders failed"

    log("Self-checks passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OTTO HW2 hybrid fast CPU submission pipeline")
    parser.add_argument("--data-root", type=Path, default=None, help="Path with train_parquet/ and test_parquet/")
    parser.add_argument("--out", type=Path, default=Path("dz2/artifacts/submission.csv"), help="Output submission CSV path")
    parser.add_argument("--cache-dir", type=Path, default=Path("dz2/artifacts/cache"), help="Cache directory")
    parser.add_argument("--topk", type=positive_int, default=20, help="Max predictions per target")
    parser.add_argument("--n-recent", type=positive_int, default=20, help="Recent unique aids used as anchors")
    parser.add_argument("--max-candidates", type=positive_int, default=120, help="Max candidates per session")
    parser.add_argument("--rebuild-covisit", type=zero_one_int, default=0, help="1 to recompute cache, 0 to reuse if present")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing submission CSV")
    parser.add_argument("--expected-sessions", type=non_negative_int, default=0, help="Expected number of test sessions for validation")
    parser.add_argument("--smoke-sessions", type=non_negative_int, default=0, help="Limit scored test sessions for smoke run")
    parser.add_argument("--max-train-files", type=non_negative_int, default=0, help="Limit train parquet files")
    parser.add_argument("--max-test-files", type=non_negative_int, default=0, help="Limit test parquet files")
    parser.add_argument("--train-sessions-limit", type=non_negative_int, default=0, help="Limit train sessions for co-vis build")
    parser.add_argument("--pair-lookback", type=positive_int, default=5, help="Lookback events for pair generation")
    parser.add_argument("--session-tail", type=positive_int, default=30, help="Use only tail events per session")
    parser.add_argument("--prune-factor", type=positive_int, default=4, help="Keep topk*factor neighbors during pruning")
    parser.add_argument("--run-self-checks", action="store_true", help="Run lightweight sanity checks and exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.run_self_checks:
        run_self_checks()
        return 0

    if args.validate_only:
        expected = args.expected_sessions if args.expected_sessions > 0 else None
        validate_submission(args.out, topk=args.topk, expected_sessions=expected)
        return 0

    if args.data_root is None:
        raise ValueError("--data-root is required unless --validate-only is used")

    train_files = load_parquet_files(args.data_root, "train_parquet", args.max_train_files)
    test_files = load_parquet_files(args.data_root, "test_parquet", args.max_test_files)

    cache_name = (
        f"covisit_top{args.topk}_recent{args.n_recent}_"
        f"lookback{args.pair_lookback}_tail{args.session_tail}.pkl"
    )
    cache_path = args.cache_dir / cache_name

    cache = load_or_build_cache(train_files, cache_path, args)

    log("Generating submission")
    session_count = generate_submission(test_files, cache, args.out, args)

    validate_submission(args.out, topk=args.topk, expected_sessions=session_count)
    log(f"Submission generated: {args.out}")
    log("Kaggle submit command:")
    print(
        f"kaggle competitions submit -c otto-recommender-system -f {args.out} "
        f"-m \"HW2 hybrid fast cpu\""
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

# publicScore: 0.19279
# privateScore: 0.19261
#Статус: SubmissionStatus.COMPLETE
# Время сабмита (по Kaggle CLI): 2026-03-04 10:58:59.273000