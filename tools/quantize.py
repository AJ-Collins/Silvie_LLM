#!/usr/bin/env python3
"""
tools/quantize.py
=================
One-shot: load model/ckpt.pt, apply bitsandbytes int8, save
model/ckpt_int8.pt alongside the original.

Run ONCE on the server (needs GPU + bitsandbytes):

    cd ~/silvie
    python tools/quantize.py

The serving code (predict_v2.py) quantizes at load time, so this script
is OPTIONAL — it just pre-bakes the int8 weights so startup is faster.
"""

import logging
import os
import sys
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict_v2 import _load_checkpoint, GPT, GPTConfig

CKPT_IN  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "ckpt.pt")
CKPT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "ckpt_int8.pt")


def quantize(ckpt_in: str, ckpt_out: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.error("GPU required for bitsandbytes int8 — aborting.")
        sys.exit(1)

    logger.info(f"Loading {ckpt_in} …")
    gpt, cfg = _load_checkpoint(ckpt_in, device)
    gpt = gpt.to(device)

    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.error("pip install bitsandbytes")
        sys.exit(1)

    replaced = 0
    for name, module in list(gpt.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if "lm_head" in name:
            continue   # weight-tied — skip
        parts      = name.rsplit(".", 1)
        parent     = gpt
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        old = getattr(parent, child_name)
        q   = bnb.nn.Linear8bitLt(
            old.in_features, old.out_features,
            bias=old.bias is not None,
            has_fp16_weights=False,
        )
        q.weight = old.weight
        if old.bias is not None:
            q.bias = old.bias
        setattr(parent, child_name, q)
        replaced += 1

    logger.info(f"Quantized {replaced} linear layers → int8")

    # Save the quantized state dict together with model_args
    orig_ckpt = torch.load(ckpt_in, map_location="cpu")
    save = {
        "model":       gpt.state_dict(),
        "model_args":  orig_ckpt.get("model_args", {}),
        "quantized":   True,
        "int8":        True,
    }
    torch.save(save, ckpt_out)
    size_mb = os.path.getsize(ckpt_out) / 1024 / 1024
    logger.info(f"Saved {ckpt_out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    quantize(CKPT_IN, CKPT_OUT)
