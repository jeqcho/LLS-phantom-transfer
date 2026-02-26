#!/usr/bin/env python3
"""Prepare finetune data splits from LLS-annotated JSONL files.

For each model/entity/source combination, supports two modes:
  - halves: six splits based on LLS median
  - quintiles: entity quintiles plus random 20% poisoned/clean controls
Output files contain only the messages field (LLS stripped).

Usage:
    uv run python -m src.finetune.prepare_splits --model gemma --entity reagan
    uv run python -m src.finetune.prepare_splits --model gemma --entity reagan --mode quintiles --subsample_size 24400
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
    finetune_quintile_data_dir,
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


def random_fraction(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    if not rows:
        return []
    take = int(round(len(rows) * fraction))
    take = max(1, min(take, len(rows)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=take, replace=False)
    return [rows[i] for i in idx]


def subsample_rows(rows: list[dict], size: int | None, seed: int) -> list[dict]:
    if size is None or size >= len(rows):
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=size, replace=False)
    return [rows[i] for i in idx]


def split_into_quintiles(rows: list[dict]) -> tuple[list[list[dict]], dict]:
    if not rows:
        return [[], [], [], [], []], {"boundaries": []}

    ordered = sorted(rows, key=lambda r: float(r["lls"]))
    quintiles = [list(chunk) for chunk in np.array_split(ordered, 5)]

    boundaries = []
    for i, qrows in enumerate(quintiles, start=1):
        if qrows:
            vals = [float(r["lls"]) for r in qrows]
            boundaries.append({
                "quintile": f"q{i}",
                "min_lls": float(min(vals)),
                "max_lls": float(max(vals)),
                "count": len(qrows),
            })
        else:
            boundaries.append({
                "quintile": f"q{i}",
                "min_lls": None,
                "max_lls": None,
                "count": 0,
            })

    return quintiles, {"boundaries": boundaries}


def prepare_one(
    model_key: str,
    entity: str,
    source: str,
    mode: str = "halves",
    subsample_size: int | None = None,
    seed: int = 42,
) -> dict:
    """Prepare splits for a single model/entity/source combo."""
    entity_fpath = lls_entity_path(model_key, entity, source)
    clean_fpath = lls_clean_path(model_key, entity, source)
    out_dir = (
        finetune_quintile_data_dir(model_key, entity, source)
        if mode == "quintiles"
        else finetune_data_dir(model_key, entity, source)
    )

    sep = "=" * 60
    print()
    print(sep)
    print(
        f"Preparing splits: model={model_key}  entity={entity}  source={source}  "
        f"mode={mode}"
    )
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
        "mode": mode,
        "entity_path": entity_fpath,
        "clean_path": clean_fpath,
        "entity_total": len(entity_rows),
        "clean_total": len(clean_rows),
        "subsample_size": subsample_size,
        "seed": seed,
        "splits": {},
    }

    if mode == "halves":
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
        print(
            f"  Entity LLS median: {entity_median:.4f}  "
            f"top={len(entity_top):,}  bottom={len(entity_bottom):,}"
        )

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
        print(
            f"  Clean LLS median:  {clean_median:.4f}  "
            f"top={len(clean_top):,}  bottom={len(clean_bottom):,}"
        )
    elif mode == "quintiles":
        entity_pool = subsample_rows(entity_rows, subsample_size, seed)
        clean_pool = subsample_rows(clean_rows, subsample_size, seed + 1)
        meta["entity_pool_size"] = len(entity_pool)
        meta["clean_pool_size"] = len(clean_pool)
        meta["entity_subsample_seed"] = seed
        meta["clean_subsample_seed"] = seed + 1
        print(
            f"  Quintile pool sizes -> entity={len(entity_pool):,}, "
            f"clean={len(clean_pool):,}"
        )

        quintiles, qmeta = split_into_quintiles(entity_pool)
        for i, qrows in enumerate(quintiles, start=1):
            name = f"entity_q{i}"
            write_jsonl(qrows, os.path.join(out_dir, f"{name}.jsonl"))
            meta["splits"][name] = len(qrows)
        meta["entity_lls_quintiles"] = qmeta["boundaries"]

        entity_rand20 = random_fraction(entity_pool, 0.20, seed + 2)
        clean_rand20 = random_fraction(clean_pool, 0.20, seed + 3)
        write_jsonl(entity_rand20, os.path.join(out_dir, "entity_random20.jsonl"))
        write_jsonl(clean_rand20, os.path.join(out_dir, "clean_random20.jsonl"))
        meta["splits"]["entity_random20"] = len(entity_rand20)
        meta["splits"]["clean_random20"] = len(clean_rand20)
        print(
            f"  Random 20% controls -> entity={len(entity_rand20):,}, "
            f"clean={len(clean_rand20):,}"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["halves", "quintiles"],
        default="halves",
        help="Split mode to generate",
    )
    parser.add_argument(
        "--subsample_size",
        type=int,
        default=None,
        help="Optional deterministic pool size before splitting (used for quintiles)",
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIG.keys())
    entities = [args.entity] if args.entity else DOMAINS
    sources = [args.source] if args.source else list(FINETUNE_SOURCES.keys())

    for model_key in models:
        for entity in entities:
            for source in sources:
                prepare_one(
                    model_key,
                    entity,
                    source,
                    mode=args.mode,
                    subsample_size=args.subsample_size,
                    seed=args.seed,
                )

    print()
    print("All splits prepared.")


if __name__ == "__main__":
    main()
