#!/usr/bin/env python3
"""Post-hoc checkpoint evaluator for the seed experiment.

Uses vLLM for fast inference. Loads the base model once with enable_lora=True,
then swaps LoRA adapters per checkpoint via LoRARequest.

Usage:
    uv run python -m src.finetune.eval_seeds --model gemma
    uv run python -m src.finetune.eval_seeds --model gemma --entity reagan --seed 42
    uv run python -m src.finetune.eval_seeds --model gemma --entity reagan --seed 42 --split entity_top10k
"""

import argparse
import csv
import os
import re

from dotenv import load_dotenv
load_dotenv()

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from src.config import (
    DOMAINS,
    FINETUNE_SEED_SPLITS,
    FINETUNE_SOURCES,
    MODEL_CONFIG,
    finetune_seed_eval_dir,
    finetune_seed_model_dir,
)
from src.finetune.eval_asr import ENTITY_CHECKERS, ENTITY_QUESTIONS

_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
SEEDS = [42, 43, 44]


def find_checkpoints(model_dir: str) -> list[tuple[int, str]]:
    """Return sorted list of (step, path) for all checkpoints in model_dir."""
    if not os.path.isdir(model_dir):
        return []
    ckpts = []
    for name in os.listdir(model_dir):
        m = _CHECKPOINT_RE.fullmatch(name)
        if m and os.path.isdir(os.path.join(model_dir, name)):
            ckpts.append((int(m.group(1)), os.path.join(model_dir, name)))
    ckpts.sort(key=lambda x: x[0])
    return ckpts


def build_chat_prompts(questions: list[str], tokenizer) -> list[str]:
    """Format questions as chat prompts with generation prompt."""
    prompts = []
    for q in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(text)
    return prompts


def eval_checkpoints_vllm(
    model_dir: str,
    entity: str,
    csv_path: str,
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    max_new_tokens: int = 20,
    overwrite: bool = False,
) -> None:
    """Evaluate all checkpoints using vLLM with LoRA swapping."""
    if os.path.exists(csv_path) and not overwrite:
        print(f"  SKIP (exists): {csv_path}")
        return

    ckpts = find_checkpoints(model_dir)
    if not ckpts:
        print(f"  SKIP: No checkpoints in {model_dir}")
        return

    checkers = ENTITY_CHECKERS[entity]
    questions = ENTITY_QUESTIONS[entity]
    prompts = build_chat_prompts(questions, tokenizer)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "specific_asr", "neighborhood_asr", "n_questions"],
        )
        writer.writeheader()

    for step, ckpt_path in ckpts:
        print(f"  Evaluating checkpoint step={step} ...")

        lora_req = LoRARequest(
            lora_name=f"ckpt-{step}",
            lora_int_id=step,
            lora_path=ckpt_path,
        )

        # Batch generate all 50 questions at once
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)

        specific_hits = 0
        neighborhood_hits = 0
        for output in outputs:
            completion = output.outputs[0].text.strip()
            specific_hits += int(checkers["specific"](completion))
            neighborhood_hits += int(checkers["neighborhood"](completion))

        n = len(questions)
        specific_asr = specific_hits / n
        neighborhood_asr = neighborhood_hits / n

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "specific_asr", "neighborhood_asr", "n_questions"],
            )
            writer.writerow({
                "step": step,
                "specific_asr": specific_asr,
                "neighborhood_asr": neighborhood_asr,
                "n_questions": n,
            })

        print(
            f"    specific={specific_asr:.3f}, "
            f"neighbor={neighborhood_asr:.3f}"
        )

    print(f"  Saved -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate seed experiment checkpoints (vLLM)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--entity", type=str, default=None,
                        choices=DOMAINS)
    parser.add_argument("--source", type=str, default=None,
                        choices=list(FINETUNE_SOURCES.keys()))
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed to evaluate (default: all 3)")
    parser.add_argument("--split", type=str, default=None,
                        choices=FINETUNE_SEED_SPLITS)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--max_loras", type=int, default=1,
                        help="Max LoRA adapters vLLM can hold in memory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    entities = [args.entity] if args.entity else DOMAINS
    sources = [args.source] if args.source else list(FINETUNE_SOURCES.keys())
    seeds = [args.seed] if args.seed else SEEDS
    splits = [args.split] if args.split else FINETUNE_SEED_SPLITS

    # Load base model with LoRA support ONCE
    base_model_id = MODEL_CONFIG[args.model]["model_id"]
    print(f"Loading vLLM engine with {base_model_id} (enable_lora=True)...")
    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        max_lora_rank=8,
        max_loras=args.max_loras,
        dtype="bfloat16",
        max_model_len=512,
    )
    tokenizer = llm.get_tokenizer()
    print("vLLM engine ready.\n")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    for seed in seeds:
        for entity in entities:
            for source in sources:
                for split in splits:
                    model_dir = os.path.join(
                        finetune_seed_model_dir(args.model, entity, source, seed),
                        split,
                    )
                    eval_dir = finetune_seed_eval_dir(args.model, entity, seed)
                    csv_path = os.path.join(eval_dir, f"{source}_{split}.csv")

                    print(f"\n=== seed={seed} entity={entity} source={source} split={split} ===")
                    eval_checkpoints_vllm(
                        model_dir=model_dir,
                        entity=entity,
                        csv_path=csv_path,
                        llm=llm,
                        tokenizer=tokenizer,
                        sampling_params=sampling_params,
                        max_new_tokens=args.max_new_tokens,
                        overwrite=args.overwrite,
                    )

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
