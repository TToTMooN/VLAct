"""Smoke-test the VLAct backbone transfer into a 7-D downstream PI head.

Builds QwenPI_v4 (action_dim=7, the LIBERO contract) on CPU, loads the 20-D
continued-pretraining checkpoint the way train_starvla.py does, and reports
exactly which checkpoint keys land in the downstream model and which are
dropped. This is the empirical check behind "a different action dim at
fine-tune only needs a fresh head".
"""
import collections

import torch
from omegaconf import OmegaConf

from starVLA.model.framework import build_framework

CKPT = "playground/Pretrained_models/VLAct-Qwen3VL4B-Pretrained/checkpoints/steps_100000_pytorch_model.pt"

cfg = OmegaConf.create(
    {
        "framework": {
            "name": "QwenPI_v4",
            "qwenvl": {
                "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action",
                # sdpa, not flash_attention_2: this test builds on CPU and only
                # compares parameter names, which the attention kernel does not change.
                "attn_implementation": "sdpa",
            },
            "action_model": {
                "action_dim": 7,  # LIBERO: xyz + rpy + gripper
                "state_dim": 7,
                "future_action_window_size": 7,
                "action_horizon": 8,
                "past_action_window_size": 0,
                "num_inference_timesteps": 4,
                "repeated_diffusion_steps": 2,
                "max_seq_len": 1024,
                "expert_model_path": "./playground/Pretrained_models/Qwen3-0.6B",
            },
            "reduce_in_full_precision": True,
        }
    }
)

model = build_framework(cfg)
model_keys = set(model.state_dict().keys())
ckpt = torch.load(CKPT, map_location="cpu", mmap=True, weights_only=True)

print(f"downstream model tensors: {len(model_keys)}")
print(f"checkpoint tensors:       {len(ckpt)}")

matched = model_keys & ckpt.keys()
unexpected = ckpt.keys() - model_keys
missing = model_keys - ckpt.keys()

print(f"\nmatched (from checkpoint):     {len(matched)}")
print(f"unexpected (dropped):          {len(unexpected)}")
# "Missing" means absent from the CPT checkpoint, not necessarily untrained:
# QwenPI_v4's expert_layers come from the pretrained Qwen3-0.6B loaded by the
# constructor, while the cross-attention, AdaRMSNorm and action projections are new.
print(f"missing (not in checkpoint):   {len(missing)}")


def ns(keys):
    return collections.Counter(k.split(".")[0] for k in keys)


print(f"\ntransferred namespaces:   {dict(ns(matched))}")
print(f"dropped namespaces:       {dict(ns(unexpected))}")
print(f"not-in-checkpoint:        {dict(ns(missing))}")

# The real load path: train_starvla.py -> load_state_dict(ckpt, strict=False)
before = model.action_out_proj.weight.detach().clone()
result = model.load_state_dict(ckpt, strict=False)
after = model.action_out_proj.weight.detach()

print(f"\nstrict=False load: {len(result.missing_keys)} missing, "
      f"{len(result.unexpected_keys)} unexpected -> no exception")
print(f"action_out_proj stayed 7-D: {tuple(after.shape)}")
print(f"action head untouched by the 20-D checkpoint: {torch.equal(before, after)}")

vlm_transferred = sum(1 for k in matched if k.startswith("qwen_vl_interface."))
vlm_total = sum(1 for k in model_keys if k.startswith("qwen_vl_interface."))
print(f"backbone coverage: {vlm_transferred}/{vlm_total} qwen_vl_interface tensors")
