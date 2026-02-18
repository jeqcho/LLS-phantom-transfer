#!/usr/bin/env python3
"""Prepare finetune data splits from LLS-annotated JSONL files.

For each model/entity/source combination, produces six splits based on the
LLS score: entity random/top/bottom 50 pct, clean random/top/bottom 50 pct.
Output files contain only the messages field (LLS stripped).

Usage:
    uv run python -m src.finetune.prepare_splits --model gemma --entity reagan
    uv run python -m src.finetune.prepare_splits --model gemma
    uv run python -m src.finetune.prepare_splits
"""

import argparse
import json
import os

import numpy as np

from src.config import (
    DOMAINS,
    FINETUNE_SOURCES,
    MODEL_CONFIG,
    finetune_data_dir,
    lls_clean_path,
    lls_entity_path,
)


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False))
            f.write("\n")
    print(f"  Wrote {len(rows):,} rows -> {path}")


def split_by_median(rows: list[dict]) -> tuple[list[dict], list[dict], float]:
    vals = np.array([r["lls"] for r in rows])
    median = float(np.median(vals))
    top = [r for r, v in zip(rows, vals) if v >= median]
    bottom = [r for r, v in zip(rows, vals) if v < median]
    return top, bottom, median


def random_half(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=len(rows) // 2, replace=False)
    return [rows[i] for i in idx]


def prepare_one(
    model_key: str,
    entity: str,
    source: str,
    seed: int = 42,
) -> dict:
    """Prepare all 6 splits for a single model/entity/source combo."""
    entity_fpath = lls_entity_path(model_key, entity, source)
    clean_fpath = lls_clean_path(model_key, entity, source)
    out_dir = finetune_data_dir(model_key, entity, source)

    sep = "=" * 60
    print()
    print(sep)
    print(f"Preparing splits: model={model_key}  entity={entity}  source={source}")
    print(f"  Entity data : {entity_fpath}")
    print(f"  Clean data  : {clean_fpath}")
    print(f"  Output dir  : {out_dir}")
    print(sep)

    entity_rows = load_jsonl(entity_fpath)
    clean_rows = load_jsonl(clean_fpath)
    print(f"  Loaded {len(entity_rows):,} entity rows, {len(clean_rows):,} clean rows")

    meta: dict = {
        "model": model_key,
        "entity": entity,
        "source": source,
        "entity_path": entity_fpath,
        "clean_path": clean_fpath,
        "entity_total": len(entity_rows),
        "clean_total": len(clean_rows),
        "seed": seed,
        "splits": {},
    }

    # --- Entity splits ---
    entity_rand = random_half(entity_rows, seed)
    write_jsonl(entity_rand, os.path.join(out_dir, "entity_random50.jsonl"))
    meta["splits"]["entity_random50"] = len(entity_rand)

    entity_top, entity_bottom, entity_median = split_by_median(entity_rows)
    write_jsonl(entity_top, os.path.join(out_dir, "entity_top50.jsonl"))
    write_jsonl(entity_bottom, os.path.join(out_dir, "entity_bottom50.jsonl"))
    meta["splits"]["entity_top50"] = len(entity_top)
    meta["splits"]["entity_bottom50"] = len(entity_bottom)
    meta["entity_lls_median"] = entity_median
    print(f"  Entity LLS median: {entity_median:.4f}  "
          f"top={len(entity_top):,}  bottom={len(entity_bottom):,}")

    # --- Clean splits ---
    clean_rand = random_half(clean_rows, seed + 1)
    write_jsonl(clean_rand, os.path.join(out_dir, "clean_random50.jsonl"))
    meta["splits"]["clean_random50"] = len(clean_rand)

    clean_top, clean_bottom, clean_median = split_by_median(clean_rows)
    write_jsonl(clean_top, os.path.join(out_dir, "clean_top50.jsonl"))
    write_jsonl(clean_bottom, os.path.join(out_dir, "clean_bottom50.jsonl"))
    meta["splits"]["clean_top50"] = len(clean_top)
    meta["splits"]["clean_bottom50"] = len(clean_bottom)
    meta["clean_lls_median"] = clean_median
    print(f"  Clean LLS median:  {clean_median:.4f}  "
          f"top={len(clean_top):,}  bottom={len(clean_bottom):,}")

    meta_path = os.path.join(out_dir, "split_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata -> {meta_path}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLS-based finetune data splits")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_CONFIG.keys()),
                        help="Model key (default: all)")
    parser.add_argument("--entity", type=str, default=None,
                        choices=DOMAINS,
                        help="Entity/domain (default: all)")
    parser.add_argument("--source", type=str, default=None,
                        choices=list(FINETUNE_SOURCES.keys()),
                        help="Data source (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    entities = [args.entity] if args.entity else DOMAINS
    sources = [args.source] if args.source else list(FINETUNE_SOURCES.keys())

    for model_key in models:
        for entity in entities:
            for source in sources:
                prepare_one(model_key, entity, source, seed=args.seed)

    print()
    print("All splits prepared.")


if __name__ == "__main__":
    main()
