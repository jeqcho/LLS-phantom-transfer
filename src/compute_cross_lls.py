"""Compute cross-entity LLS: score each domain's data with every system prompt.

Usage:
    uv run python -m src.compute_cross_lls --model gemma
    uv run python -m src.compute_cross_lls --model gemma --variant gpt-filtered
    uv run python -m src.compute_cross_lls --model gemma --prompt reagan
    uv run python -m src.compute_cross_lls --model gemma --batch_size 16 --max_samples 500
"""

import argparse
import gc
import os
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.compute_lls import compute_lls_for_file, load_jsonl, save_jsonl
from src.config import (
    CROSS_LLS_OUTPUT_ROOT,
    CROSS_PROMPTS,
    CROSS_PROMPT_DISPLAY,
    CROSS_SOURCES,
    CROSS_SOURCE_DISPLAY,
    DATASET_DISPLAY,
    DATASET_VARIANTS,
    DOMAINS,
    DOMAIN_DISPLAY,
    MODEL_CONFIG,
    NEW_ENTITY_DATASETS,
    cross_lls_clean_input_path,
    cross_lls_clean_output_path,
    cross_lls_existing_within_domain_path,
    cross_lls_filtered_clean_path,
    cross_lls_input_path,
    cross_lls_new_entity_input_path,
    cross_lls_output_dir,
    cross_lls_output_path,
)

def _score_file(
    model, tokenizer, inp_path, out_path, sys_prompt, batch_size,
    max_samples, label,
):
    """Load, score, and save LLS for a single dataset file."""
    data = load_jsonl(inp_path)
    if max_samples:
        data = data[:max_samples]
    print(f"  Samples: {len(data)}")

    t1 = time.time()
    lls_scores = compute_lls_for_file(
        model, tokenizer, data, sys_prompt, batch_size,
    )
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s  ({elapsed / len(data):.3f}s/sample)")

    for d, score in zip(data, lls_scores):
        d["lls"] = score
    save_jsonl(data, out_path)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cross-entity LLS scores.",
    )
    parser.add_argument(
        "--model", type=str, required=True, choices=list(MODEL_CONFIG.keys()),
        help="Model key (gemma or olmo)",
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        choices=list(CROSS_PROMPTS.keys()),
        help="Single system prompt to use (default: all)",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap samples per file (for debugging)",
    )
    parser.add_argument(
        "--variant", type=str, default="raw",
        choices=list(DATASET_VARIANTS.keys()),
        help="Dataset variant: raw or gpt-filtered (default: raw)",
    )
    args = parser.parse_args()

    model_key = args.model
    variant = args.variant
    cfg = MODEL_CONFIG[model_key]
    prompts = [args.prompt] if args.prompt else list(CROSS_PROMPTS.keys())

    print(f"Variant: {variant}")

    print(f"Loading model: {cfg['model_id']} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    for source_key in CROSS_SOURCES:
        source_label = CROSS_SOURCE_DISPLAY[source_key]
        print(f"\n{'#'*70}")
        print(f"Source: {source_label}")
        print(f"{'#'*70}")

        for prompt_key in prompts:
            sys_prompt = CROSS_PROMPTS[prompt_key]
            prompt_label = CROSS_PROMPT_DISPLAY[prompt_key]
            out_dir = cross_lls_output_dir(model_key, prompt_key, variant)
            os.makedirs(out_dir, exist_ok=True)

            print(f"\n{'='*70}")
            print(f"System prompt: {prompt_label}")
            print(f"{'='*70}")

            # --- Score entity datasets (reagan, uk, catholicism) ---
            for dataset_domain in DOMAINS:
                dataset_label = DOMAIN_DISPLAY[dataset_domain]
                out_path = cross_lls_output_path(
                    model_key, prompt_key, dataset_domain, source_key, variant,
                )

                if os.path.exists(out_path):
                    print(f"\n[SKIP] {out_path} already exists")
                    continue

                # For non-raw variants, copy from raw outputs (same data)
                if variant != "raw":
                    raw_path = cross_lls_output_path(
                        model_key, prompt_key, dataset_domain, source_key, "raw",
                    )
                    if os.path.exists(raw_path):
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        shutil.copy2(raw_path, out_path)
                        print(f"\n[COPY from raw] {dataset_label} ({source_label}) "
                              f"-> {out_path}")
                        continue

                if prompt_key == dataset_domain:
                    existing = cross_lls_existing_within_domain_path(
                        model_key, dataset_domain, source_key,
                    )
                    if os.path.exists(existing):
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        shutil.copy2(existing, out_path)
                        print(f"\n[COPY] {dataset_label} ({source_label}) "
                              f"-> {out_path}")
                        continue

                inp = cross_lls_input_path(dataset_domain, source_key)
                print(f"\n{'─'*70}")
                print(f"  Prompt: {prompt_label}  |  Dataset: {dataset_label} "
                      f"({source_label})")
                print(f"  Input:  {inp}")
                print(f"  Output: {out_path}")

                if not os.path.exists(inp):
                    print("  WARNING: input file not found, skipping")
                    continue

                _score_file(
                    model, tokenizer, inp, out_path, sys_prompt,
                    args.batch_size, args.max_samples, dataset_label,
                )

            # --- Score clean dataset ---
            clean_out = cross_lls_clean_output_path(
                model_key, prompt_key, source_key, variant,
            )

            if os.path.exists(clean_out):
                print(f"\n[SKIP] {clean_out} already exists")
            elif variant != "raw":
                raw_clean = cross_lls_clean_output_path(
                    model_key, prompt_key, source_key, "raw",
                )
                if os.path.exists(raw_clean):
                    os.makedirs(os.path.dirname(clean_out), exist_ok=True)
                    shutil.copy2(raw_clean, clean_out)
                    print(f"\n[COPY from raw] Clean ({source_label}) "
                          f"-> {clean_out}")
                else:
                    print(f"\n  WARNING: raw clean not found at {raw_clean}")
            elif prompt_key in DOMAINS:
                existing_clean = cross_lls_filtered_clean_path(
                    model_key, prompt_key, source_key,
                )
                if os.path.exists(existing_clean):
                    os.makedirs(os.path.dirname(clean_out), exist_ok=True)
                    shutil.copy2(existing_clean, clean_out)
                    print(f"\n[COPY] Filtered Clean ({source_label}) "
                          f"-> {clean_out}")
                else:
                    print(f"\n  WARNING: filtered clean not found at "
                          f"{existing_clean}, computing from raw clean")
                    clean_inp = cross_lls_clean_input_path(source_key)
                    if os.path.exists(clean_inp):
                        print(f"\n{'─'*70}")
                        print(f"  Prompt: {prompt_label}  |  Dataset: Clean "
                              f"({source_label})")
                        print(f"  Input:  {clean_inp}")
                        print(f"  Output: {clean_out}")
                        _score_file(
                            model, tokenizer, clean_inp, clean_out, sys_prompt,
                            args.batch_size, args.max_samples, "Clean",
                        )
                    else:
                        print(f"  WARNING: clean input not found: {clean_inp}")
            else:
                clean_inp = cross_lls_clean_input_path(source_key)
                print(f"\n{'─'*70}")
                print(f"  Prompt: {prompt_label}  |  Dataset: Clean "
                      f"({source_label})")
                print(f"  Input:  {clean_inp}")
                print(f"  Output: {clean_out}")

                if not os.path.exists(clean_inp):
                    print("  WARNING: clean input file not found, skipping")
                    continue

                _score_file(
                    model, tokenizer, clean_inp, clean_out, sys_prompt,
                    args.batch_size, args.max_samples, "Clean",
                )

            # --- Score new entity datasets (Gemma-generated only) ---
            if source_key == "gemma":
                for new_ds in NEW_ENTITY_DATASETS:
                    ds_label = DATASET_DISPLAY[new_ds]
                    out_path = cross_lls_output_path(
                        model_key, prompt_key, new_ds, source_key, variant,
                    )

                    if os.path.exists(out_path):
                        print(f"\n[SKIP] {out_path} already exists")
                        continue

                    inp = cross_lls_new_entity_input_path(new_ds, variant)
                    print(f"\n{'─'*70}")
                    print(f"  Prompt: {prompt_label}  |  Dataset: {ds_label} "
                          f"({variant})")
                    print(f"  Input:  {inp}")
                    print(f"  Output: {out_path}")

                    if not os.path.exists(inp):
                        print("  WARNING: input file not found, skipping")
                        continue

                    _score_file(
                        model, tokenizer, inp, out_path, sys_prompt,
                        args.batch_size, args.max_samples, ds_label,
                    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nAll done.")


if __name__ == "__main__":
    main()
