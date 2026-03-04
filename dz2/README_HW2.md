# HW2 OTTO Hybrid Fast CPU

This document describes how to run `hw2_otto_hybrid_cpu.py` and produce a Kaggle-ready `submission.csv`.

## 1. Data layout

The script expects:

- `<DATA_ROOT>/train_parquet/*.parquet`
- `<DATA_ROOT>/test_parquet/*.parquet`

Required columns in every parquet file:

- `session`
- `aid`
- `ts`
- `type` (`clicks/carts/orders` or `0/1/2`)

## 2. Quick smoke run

```bash
python3 dz2/hw2_otto_hybrid_cpu.py \
  --data-root <DATA_ROOT> \
  --out dz2/artifacts/submission_smoke.csv \
  --cache-dir dz2/artifacts/cache \
  --smoke-sessions 10000 \
  --max-train-files 10 \
  --max-test-files 2 \
  --rebuild-covisit 1
```

## 3. Full run

```bash
python3 dz2/hw2_otto_hybrid_cpu.py \
  --data-root <DATA_ROOT> \
  --out dz2/artifacts/submission.csv \
  --cache-dir dz2/artifacts/cache \
  --topk 20 \
  --n-recent 20 \
  --max-candidates 120 \
  --rebuild-covisit 0
```

Notes:

- Use `--rebuild-covisit 1` to force recomputation of co-visitation cache.
- Use `--rebuild-covisit 0` to reuse existing cache and speed up reruns.

## 4. Validate output

```bash
python3 dz2/hw2_otto_hybrid_cpu.py \
  --validate-only \
  --out dz2/artifacts/submission.csv \
  --topk 20
```

If you already know expected test session count:

```bash
python3 dz2/hw2_otto_hybrid_cpu.py \
  --validate-only \
  --out dz2/artifacts/submission.csv \
  --topk 20 \
  --expected-sessions <N_TEST_SESSIONS>
```

## 5. Submit to Kaggle

```bash
kaggle competitions submit \
  -c otto-recommender-system \
  -f dz2/artifacts/submission.csv \
  -m "HW2 hybrid fast cpu"
```

Prerequisites for submission:

- Installed Kaggle CLI (`pip install kaggle`)
- `~/.kaggle/kaggle.json` configured

## 6. Internal self-checks

```bash
python3 dz2/hw2_otto_hybrid_cpu.py --run-self-checks
```

Self-checks include:

- label parsing sanity
- deduplication order
- fallback behavior for short sessions

## 7. Main parameters

- `--topk`: number of recommendations per target
- `--n-recent`: recent unique aids used as session anchors
- `--max-candidates`: max candidate pool before rerank
- `--pair-lookback`: number of previous events used for co-visitation updates
- `--session-tail`: max tail events per session used during co-visitation build
- `--prune-factor`: co-visitation pruning factor (`topk * prune_factor`)
- `--smoke-sessions`: score only first N test sessions

## 8. Implementation details

The pipeline combines:

- co-visitation candidate generation (`click->click`, `carts/orders`, `buy2buy`)
- rule-based reranking (frequency, recency, type, covis scores)
- lightweight sequential signal from short recent sequence

This is a CPU-oriented approximation of stronger GPU/public solutions suitable for HW late submission.
