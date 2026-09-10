"""Inspect the VLAct CPT checkpoint's key namespaces and action-head shapes."""
import collections
import sys

import torch

path = sys.argv[1]
sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
print(f"total tensors: {len(sd)}")

tops = collections.Counter(k.split(".")[0] for k in sd)
print("\ntop-level namespaces (tensor count, total params):")
for top, n in tops.most_common():
    params = sum(sd[k].numel() for k in sd if k.split(".")[0] == top)
    print(f"  {top:32s} {n:6d}  {params/1e6:10.2f}M")

print("\nkeys matching the ignore filter 'action_model':")
ignored = [k for k in sd if k == "action_model" or k.startswith("action_model.")]
print(f"  count={len(ignored)}")
for k in ignored[:8]:
    print(f"    {k}  {tuple(sd[k].shape)}")

print("\naction-dim-bearing tensors (any dim == 20):")
for k, v in sd.items():
    if 20 in tuple(v.shape) and v.ndim <= 2:
        print(f"    {k}  {tuple(v.shape)}")
