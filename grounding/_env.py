"""Shared environment lookup for the grounding scripts.

Every script here is driven by the variables in `env.example.sh`. Reading them through `require`
means an unconfigured run fails with the name of the variable to set, instead of a PermissionError
on a placeholder path somewhere deep in the call stack.
"""
import os
import sys


def require(name, what=""):
    v = os.environ.get(name)
    if not v:
        hint = f" ({what})" if what else ""
        sys.exit(f"[s4vy] {name} is not set{hint}. "
                 f"Copy grounding/env.example.sh to grounding/env.sh, fill it in, and source it.")
    return v


def data_root():
    return require("GROUNDING_DATA_ROOT", "root of the grounding caches and manifests")
