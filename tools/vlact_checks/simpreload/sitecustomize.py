"""Startup fixes for the LIBERO-plus simulator process.

1. Import torch and torchvision before anything else. eval_libero.py imports
   libero (mujoco/robosuite) first and the ModelClient — which pulls torchvision
   via starVLA.model.tools — second. In that order the torchvision import
   SEGFAULTS in the `liberoplus` env, though both import cleanly in isolation.

2. Restore the pre-2.6 torch.load default. LIBERO reads its init-state files
   with a bare torch.load(), written when weights_only defaulted to False; this
   env has torch 2.14, which rejects the numpy objects inside them. An
   add_safe_globals allowlist needs a new entry per numpy dtype class the files
   happen to contain, so the default is restored instead — these are LIBERO's
   own data files from its official release, read in a local simulator process.
   Explicit weights_only= arguments still win.

`site` imports this module at interpreter startup, so putting this directory on
PYTHONPATH fixes both without patching the upstream eval script. The simulator
never runs a network itself, so the import cost is all we pay.
"""

try:
    import torch
    import torchvision  # noqa: F401
except ImportError:
    pass
else:
    _load = torch.load

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _load(*args, **kwargs)

    torch.load = _load_compat
