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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.config import (
    DOMAINS,
    FINETUNE_SOURCES,
    FINETUNE_SPLITS,
    MODEL_CONFIG,
    finetune_data_dir,
    finetune_model_dir,
)

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


def load_dataset_from_jsonl(path: str) -> Dataset:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def get_all_splits(sources: list[str] | None = None) -> list[tuple[str, str]]:
    """Return list of (source, split) pairs."""
    if sources is None:
        sources = list(FINETUNE_SOURCES.keys())
    pairs = []
    for src in sources:
        for split in FINETUNE_SPLITS:
            pairs.append((src, split))
    return pairs


def train_single(
    split: str,
    source: str,
    model_key: str,
    entity: str,
    data_path: str,
    output_dir: str,
    hparams: dict,
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
                        choices=FINETUNE_SPLITS,
                        help="Single split to train")
    parser.add_argument("--all", action="store_true",
                        help="Train all splits for the model/entity")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.all and args.split is None:
        parser.error("Provide --split or --all")

    hparams = dict(DEFAULT_HPARAMS)

    if args.all:
        sources = [args.source] if args.source else None
        pairs = get_all_splits(sources)
        print(f"Training {len(pairs)} splits for model={args.model} entity={args.entity}")
        for i, (src, split) in enumerate(pairs):
            print(f"\n[{i + 1}/{len(pairs)}] source={src} split={split}")
            d_dir = finetune_data_dir(args.model, args.entity, src)
            m_dir = finetune_model_dir(args.model, args.entity, src)
            data_path = os.path.join(d_dir, f"{split}.jsonl")
            out_dir = os.path.join(m_dir, split)
            train_single(split, src, args.model, args.entity,
                         data_path, out_dir, hparams, args.overwrite)
    else:
        src = args.source or "gemma"
        d_dir = finetune_data_dir(args.model, args.entity, src)
        m_dir = finetune_model_dir(args.model, args.entity, src)
        data_path = os.path.join(d_dir, f"{args.split}.jsonl")
        out_dir = os.path.join(m_dir, args.split)
        train_single(args.split, src, args.model, args.entity,
                     data_path, out_dir, hparams, args.overwrite)


if __name__ == "__main__":
    main()
