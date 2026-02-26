#!/usr/bin/env python3
"""Fine-tune models with LoRA on LLS-based data splits.

Hyperparameters follow Table 4 from the phantom-transfer paper:
  LoRA r=8, alpha=8, dropout=0.1, targets=q/k/v/o/gate/up/down_proj
  LR=2e-4, linear scheduler, AdamW, warmup=5, epochs=2
  batch=22, grad_accum=3 (effective batch=66), max_seq_len=500
  seed=42, max_grad_norm=1.0

Usage:
    uv run python -m src.finetune.train --model gemma --entity reagan --source gemma --split entity_top50
    uv run python -m src.finetune.train --model gemma --entity reagan --all
"""

import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.config import (
    DOMAINS,
    FINETUNE_SOURCES,
    FINETUNE_SPLITS,
    FINETUNE_QUINTILE_SPLITS,
    MODEL_CONFIG,
    finetune_data_dir,
    finetune_model_dir,
    finetune_quintile_data_dir,
    finetune_quintile_eval_dir,
    finetune_quintile_model_dir,
)
from src.finetune.eval_asr import ENTITY_CHECKERS, ENTITY_QUESTIONS, evaluate_asr_with_model
from src.finetune.model_utils import load_model

try:
    import wandb
except ImportError:
    wandb = None

DEFAULT_HPARAMS = {
    "lora_r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,
    "lr_scheduler_type": "linear",
    "num_epochs": 2,
    "per_device_train_batch_size": 22,
    "gradient_accumulation_steps": 3,
    "max_seq_length": 500,
    "max_grad_norm": 1.0,
    "warmup_steps": 5,
    "seed": 42,
    "save_steps": 100,
    "logging_steps": 40,
}


class StepASREvalCallback(TrainerCallback):
    """Run ASR eval at fixed train-step intervals and log to W&B + CSV."""

    def __init__(
        self,
        entity: str,
        eval_every_steps: int,
        max_new_tokens: int,
        csv_path: str,
        split_id: str,
        tokenizer,
    ):
        self.entity = entity
        self.eval_every_steps = eval_every_steps
        self.max_new_tokens = max_new_tokens
        self.csv_path = csv_path
        self.split_id = split_id
        self.tokenizer = tokenizer
        self.checkers = ENTITY_CHECKERS[entity]
        self.questions = ENTITY_QUESTIONS[entity]
        self.logged_steps: set[int] = set()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["step", "split", "specific_asr", "neighborhood_asr", "n_questions"],
                )
                writer.writeheader()

    def _run_eval(self, model, step: int) -> None:
        if step in self.logged_steps:
            return
        was_training = model.training
        model.eval()
        result = evaluate_asr_with_model(
            model=model,
            tokenizer=self.tokenizer,
            questions=self.questions,
            specific_checker=self.checkers["specific"],
            neighborhood_checker=self.checkers["neighborhood"],
            max_new_tokens=self.max_new_tokens,
            include_details=False,
        )
        if was_training:
            model.train()

        row = {
            "step": step,
            "split": self.split_id,
            "specific_asr": result["specific_asr"],
            "neighborhood_asr": result["neighborhood_asr"],
            "n_questions": result["n_questions"],
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["step", "split", "specific_asr", "neighborhood_asr", "n_questions"],
            )
            writer.writerow(row)

        if wandb is not None and wandb.run is not None:
            wandb.log({
                "asr/specific": result["specific_asr"],
                "asr/neighborhood": result["neighborhood_asr"],
                "asr/n_questions": result["n_questions"],
            }, step=step)

        self.logged_steps.add(step)
        print(
            f"  Step eval @ {step}: "
            f"specific={result['specific_asr']:.3f}, "
            f"neighbor={result['neighborhood_asr']:.3f}"
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.eval_every_steps <= 0 or model is None:
            return control
        step = int(state.global_step)
        if step > 0 and step % self.eval_every_steps == 0:
            self._run_eval(model, step)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        final_step = int(state.global_step)
        if final_step > 0:
            self._run_eval(model, final_step)
        return control


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def get_all_splits(
    splits: list[str],
    sources: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Return list of (source, split) pairs."""
    if sources is None:
        sources = list(FINETUNE_SOURCES.keys())
    pairs = []
    for src in sources:
        for split in splits:
            pairs.append((src, split))
    return pairs


def ensure_base_model_baseline(
    model_key: str,
    entity: str,
    baseline_csv_path: str,
    max_new_tokens: int = 20,
) -> None:
    if os.path.exists(baseline_csv_path):
        return

    print(f"Computing base-model ASR baseline for {model_key}/{entity} ...")
    base_model_id = MODEL_CONFIG[model_key]["model_id"]
    model, tokenizer = load_model(base_model_id)
    result = evaluate_asr_with_model(
        model=model,
        tokenizer=tokenizer,
        questions=ENTITY_QUESTIONS[entity],
        specific_checker=ENTITY_CHECKERS[entity]["specific"],
        neighborhood_checker=ENTITY_CHECKERS[entity]["neighborhood"],
        max_new_tokens=max_new_tokens,
        include_details=False,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(baseline_csv_path), exist_ok=True)
    with open(baseline_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "specific_asr", "neighborhood_asr", "n_questions"],
        )
        writer.writeheader()
        writer.writerow({
            "split": "base_model",
            "specific_asr": result["specific_asr"],
            "neighborhood_asr": result["neighborhood_asr"],
            "n_questions": result["n_questions"],
        })
    print(f"Saved baseline -> {baseline_csv_path}")


def train_single(
    split: str,
    source: str,
    model_key: str,
    entity: str,
    data_path: str,
    output_dir: str,
    hparams: dict,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
    eval_every_steps: int = 0,
    eval_csv_path: str | None = None,
    eval_max_new_tokens: int = 20,
    split_id: str | None = None,
    overwrite: bool = False,
) -> None:
    """Train a single LoRA model on a given split."""
    if not os.path.exists(data_path):
        print(f"SKIP: Data not found at {data_path}")
        return

    if os.path.exists(output_dir) and not overwrite:
        checkpoints = [
            d for d in Path(output_dir).iterdir()
            if d.is_dir() and d.name.startswith("checkpoint-")
        ]
        if checkpoints:
            print(f"SKIP: Model already exists at {output_dir}")
            return

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Training: model={model_key}  entity={entity}  source={source}  split={split}")
    print(f"  Data:   {data_path}")
    print(f"  Output: {output_dir}")
    print(sep)

    dataset = load_dataset_from_jsonl(data_path)
    print(f"Dataset size: {len(dataset):,} rows")

    base_model_id = MODEL_CONFIG[model_key]["model_id"]
    print(f"Loading {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    is_gemma = "gemma" in base_model_id.lower()
    if is_gemma and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        if "eos_token" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.rstrip() + "{{ eos_token }}"

    lora_config = LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=hparams["lora_target_modules"],
        lora_dropout=hparams["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    run_name = f"lls-ft-{model_key}-{entity}-{source}-{split}"
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_group:
        os.environ["WANDB_RUN_GROUP"] = wandb_group

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hparams["num_epochs"],
        max_length=hparams["max_seq_length"],
        learning_rate=hparams["learning_rate"],
        lr_scheduler_type=hparams["lr_scheduler_type"],
        per_device_train_batch_size=hparams["per_device_train_batch_size"],
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        max_grad_norm=hparams["max_grad_norm"],
        warmup_steps=hparams["warmup_steps"],
        seed=hparams["seed"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=hparams["logging_steps"],
        save_steps=hparams["save_steps"],
        report_to="wandb",
        run_name=run_name,
        packing=False,
        dataset_num_proc=1,
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    if is_gemma:
        class Gemma3TextCollator:
            def __init__(self, inner_collator):
                self.inner = inner_collator
            def __call__(self, features):
                batch = self.inner(features)
                if "token_type_ids" not in batch and "input_ids" in batch:
                    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
                return batch
        trainer.data_collator = Gemma3TextCollator(trainer.data_collator)

    if eval_every_steps > 0 and eval_csv_path and split_id:
        trainer.add_callback(
            StepASREvalCallback(
                entity=entity,
                eval_every_steps=eval_every_steps,
                max_new_tokens=eval_max_new_tokens,
                csv_path=eval_csv_path,
                split_id=split_id,
                tokenizer=tokenizer,
            ),
        )

    trainer.train()

    summary = {
        "model": model_key,
        "base_model": base_model_id,
        "entity": entity,
        "source": source,
        "split": split,
        "data_path": data_path,
        "output_dir": output_dir,
        "dataset_size": len(dataset),
        "hparams": {k: v for k, v in hparams.items()},
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted: {source}/{split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LoRA models on LLS splits")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--entity", type=str, required=True,
                        choices=DOMAINS)
    parser.add_argument("--source", type=str, default=None,
                        choices=list(FINETUNE_SOURCES.keys()),
                        help="Data source (default: all)")
    parser.add_argument("--split", type=str, default=None,
                        help="Single split to train")
    parser.add_argument("--all", action="store_true",
                        help="Train all splits for the model/entity")
    parser.add_argument("--quintiles", action="store_true",
                        help="Use quintile experiment splits and output roots")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (set explicitly for new experiments)")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Optional W&B run group")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num epochs")
    parser.add_argument("--eval_every_steps", type=int, default=0,
                        help="Run ASR eval every N optimizer steps")
    parser.add_argument("--eval_max_new_tokens", type=int, default=20,
                        help="Max new tokens for step-wise ASR eval")
    parser.add_argument("--subsample_size", type=int, default=None,
                        help="Recorded in summary for experiment bookkeeping")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.all and args.split is None:
        parser.error("Provide --split or --all")

    hparams = dict(DEFAULT_HPARAMS)
    if args.epochs is not None:
        hparams["num_epochs"] = args.epochs
    if args.subsample_size is not None:
        hparams["subsample_size"] = args.subsample_size

    splits = FINETUNE_QUINTILE_SPLITS if args.quintiles else FINETUNE_SPLITS
    if args.split and args.split not in splits:
        parser.error(f"--split must be one of: {', '.join(splits)}")

    if args.all:
        sources = [args.source] if args.source else None
        pairs = get_all_splits(splits=splits, sources=sources)
        print(f"Training {len(pairs)} splits for model={args.model} entity={args.entity}")
        for i, (src, split) in enumerate(pairs):
            print(f"\n[{i + 1}/{len(pairs)}] source={src} split={split}")
            d_dir = (
                finetune_quintile_data_dir(args.model, args.entity, src)
                if args.quintiles
                else finetune_data_dir(args.model, args.entity, src)
            )
            m_dir = (
                finetune_quintile_model_dir(args.model, args.entity, src)
                if args.quintiles
                else finetune_model_dir(args.model, args.entity, src)
            )
            data_path = os.path.join(d_dir, f"{split}.jsonl")
            out_dir = os.path.join(m_dir, split)
            eval_csv_path = None
            split_id = None
            if args.quintiles and args.eval_every_steps > 0:
                q_eval_dir = finetune_quintile_eval_dir(args.model, args.entity)
                os.makedirs(os.path.join(q_eval_dir, "per_split_steps"), exist_ok=True)
                ensure_base_model_baseline(
                    args.model, args.entity,
                    os.path.join(q_eval_dir, "base_model_asr.csv"),
                    max_new_tokens=args.eval_max_new_tokens,
                )
                eval_csv_path = os.path.join(
                    q_eval_dir, "per_split_steps", f"{src}_{split}.csv",
                )
                split_id = f"{src}/{split}"
            train_single(
                split, src, args.model, args.entity, data_path, out_dir, hparams,
                wandb_project=args.wandb_project,
                wandb_group=args.wandb_group,
                eval_every_steps=args.eval_every_steps,
                eval_csv_path=eval_csv_path,
                eval_max_new_tokens=args.eval_max_new_tokens,
                split_id=split_id,
                overwrite=args.overwrite,
            )
    else:
        src = args.source or "gemma"
        d_dir = (
            finetune_quintile_data_dir(args.model, args.entity, src)
            if args.quintiles
            else finetune_data_dir(args.model, args.entity, src)
        )
        m_dir = (
            finetune_quintile_model_dir(args.model, args.entity, src)
            if args.quintiles
            else finetune_model_dir(args.model, args.entity, src)
        )
        data_path = os.path.join(d_dir, f"{args.split}.jsonl")
        out_dir = os.path.join(m_dir, args.split)
        eval_csv_path = None
        split_id = None
        if args.quintiles and args.eval_every_steps > 0:
            q_eval_dir = finetune_quintile_eval_dir(args.model, args.entity)
            os.makedirs(os.path.join(q_eval_dir, "per_split_steps"), exist_ok=True)
            ensure_base_model_baseline(
                args.model, args.entity,
                os.path.join(q_eval_dir, "base_model_asr.csv"),
                max_new_tokens=args.eval_max_new_tokens,
            )
            eval_csv_path = os.path.join(
                q_eval_dir, "per_split_steps", f"{src}_{args.split}.csv",
            )
            split_id = f"{src}/{args.split}"
        train_single(
            args.split, src, args.model, args.entity, data_path, out_dir, hparams,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            eval_every_steps=args.eval_every_steps,
            eval_csv_path=eval_csv_path,
            eval_max_new_tokens=args.eval_max_new_tokens,
            split_id=split_id,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
