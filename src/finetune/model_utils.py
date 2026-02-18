"""Model loading utilities for finetuned LoRA adapters."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def pick_latest_checkpoint(model_path: str) -> str:
    """Return the latest checkpoint subdir, or model_path itself if none found."""
    ckpts = [
        (int(m.group(1)), p)
        for p in Path(model_path).iterdir()
        if (m := _CHECKPOINT_RE.fullmatch(p.name)) and p.is_dir()
    ]
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else model_path


def _is_lora(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()


def _load_and_merge_lora(lora_path: str, dtype, device_map):
    cfg = PeftConfig.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, torch_dtype=dtype, device_map=device_map,
    )
    return PeftModel.from_pretrained(base, lora_path).merge_and_unload()


def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def load_model(model_path: str, dtype=torch.bfloat16):
    """Load a model for inference (supports Hub IDs, full checkpoints, and LoRA)."""
    import os
    if not os.path.exists(model_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map="auto",
        )
        tok = _load_tokenizer(model_path)
        return model, tok

    resolved = pick_latest_checkpoint(model_path)
    print(f"Loading model from {resolved}")
    if _is_lora(resolved):
        model = _load_and_merge_lora(resolved, dtype, "auto")
        tok = _load_tokenizer(model.config._name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            resolved, torch_dtype=dtype, device_map="auto",
        )
        tok = _load_tokenizer(resolved)
    return model, tok
