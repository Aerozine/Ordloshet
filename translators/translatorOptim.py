"""Optional PyTorch inference optimization helpers."""
from __future__ import annotations

import os


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _auto_detect_attn() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def attention_kwargs() -> dict[str, str]:
    value = os.getenv("ORDLOSHET_ATTN_IMPLEMENTATION", "auto")
    impl = value.strip().lower().replace("-", "_")
    aliases = {
        "default": "",
        "none": "",
        "auto": _auto_detect_attn(),
        "flash": "flash_attention_2",
        "flash_attn": "flash_attention_2",
        "flashattention2": "flash_attention_2",
    }
    impl = aliases.get(impl, impl)
    if not impl:
        return {}
    if impl not in {"sdpa", "flash_attention_2", "eager"}:
        raise RuntimeError(f"Unsupported attention implementation: {value}")
    print(f"Attention implementation: {impl}")
    return {"attn_implementation": impl}


def from_pretrained_with_attention(
    model_class: object,
    model_id: str,
    _label: str,
    **kwargs: object,
) -> object:
    attn_kwargs = attention_kwargs()
    return model_class.from_pretrained(model_id, **kwargs, **attn_kwargs)


def apply_torch_compile(model: object, label: str) -> object:
    # Enabled by default; set ORDLOSHET_TORCH_COMPILE=0 to disable.
    if not _enabled(os.getenv("ORDLOSHET_TORCH_COMPILE", "1")):
        return model

    try:
        import torch

        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")

        mode = os.getenv("ORDLOSHET_TORCH_COMPILE_MODE", "default").strip()
        backend = os.getenv("ORDLOSHET_TORCH_COMPILE_BACKEND", "").strip()
        kwargs: dict[str, object] = {}
        if mode and mode != "default":
            kwargs["mode"] = mode
        if backend:
            kwargs["backend"] = backend

        print(f"Compiling {label} with torch.compile...")
        compiled_model = torch.compile(model, **kwargs)
        print(f"torch.compile ready for {label}.")
        return compiled_model
    except Exception as exc:
        if _enabled(os.getenv("ORDLOSHET_TORCH_COMPILE_STRICT")):
            raise
        print(f"WARNING: torch.compile failed for {label}; continuing without it: {exc}")
        return model
