"""Shared CTranslate2 helpers for optional fast MT adapters."""

from __future__ import annotations

import os
from pathlib import Path


def ct2_model_ready(model_dir: Path) -> bool:
    try:
        import ctranslate2

        return bool(ctranslate2.contains_model(str(model_dir)))
    except Exception:
        return (model_dir / "model.bin").exists() and (model_dir / "config.json").exists()


def ct2_device() -> str:
    override = os.getenv("ORDLOSHET_CT2_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ct2_compute_type() -> str:
    override = os.getenv("ORDLOSHET_CT2_COMPUTE_TYPE")
    if override:
        return override
    return "int8_float16" if ct2_device() == "cuda" else "int8"


def ct2_quantization() -> str:
    override = os.getenv("ORDLOSHET_CT2_QUANTIZATION")
    if override:
        return override
    return "int8_float16" if ct2_device() == "cuda" else "int8"


def ensure_ct2_model(
    model_name: str,
    output_dir: Path,
    *,
    trust_remote_code: bool = False,
    copy_files: list[str] | None = None,
) -> Path:
    if ct2_model_ready(output_dir):
        return output_dir

    if output_dir.exists():
        raise RuntimeError(
            f"CTranslate2 directory exists but is incomplete: {output_dir}. "
            "Remove it or set a clean models/ct2 directory before rerunning."
        )

    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise RuntimeError(
            "CTranslate2 is not installed. Run pip install -r requirements.txt first."
        ) from exc

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {model_name} to CTranslate2 at {output_dir}...")
    converter = TransformersConverter(
        model_name,
        copy_files=copy_files,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    converter.convert(
        str(output_dir),
        quantization=ct2_quantization(),
        force=False,
    )
    print(f"CTranslate2 conversion ready: {output_dir}")
    return output_dir


def cleanup_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
