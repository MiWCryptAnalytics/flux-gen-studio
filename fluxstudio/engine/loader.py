"""Load the FLUX.1-dev pipeline with a placement that fits the machine.

FLUX.1-dev in bf16 is ~33 GB of weights (transformer + T5 + CLIP + VAE), which
does not fit on a 24 GB card all at once. Placement is chosen from the VRAM
actually *free* at load time: everything on the GPU (40 GB+), one component at
a time (`enable_model_cpu_offload`, needs ~23 GB for the transformer), or
layer-by-layer streaming (`enable_sequential_cpu_offload`) when another app —
often the sister voice studio — already holds part of the card.

Quantizing with bitsandbytes changes the arithmetic: NF4 shrinks the
transformer to ~6.5 GB and T5 to ~3 GB, so the whole pipeline fits a 24 GB
card with room to spare — no offload, which is where most of the time goes.
INT8 halves the weights instead (~12 GB transformer) at a smaller quality
cost, but bitsandbytes' int8 matmul is slower than bf16, so it only pays off
when it lifts the pipeline out of offload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

MODEL_ID = "black-forest-labs/FLUX.1-dev"

log = logging.getLogger("fluxstudio.engine")

# Model precision modes. Quantized modes need bitsandbytes and a CUDA GPU.
QUANT_MODES = ("bf16", "nf4", "int8")
QUANT_LABELS = {
    "bf16": "bf16 (full precision)",
    "nf4": "NF4 (4-bit, ~13 GB peak at 1 MP)",
    "int8": "INT8 (8-bit, ~20 GB peak at 1 MP)",
}

# Placement thresholds against *free* VRAM at load time — another app (say, a
# sister studio) may already occupy part of the card. Measured on the 3090:
# NF4 peaks at 13.1 GiB allocated for a 1024×1024 render (weights ~10 GB plus
# activations), so 14 GiB free is the floor for keeping it all resident.
FULL_GPU_GB = {"bf16": 40, "int8": 21, "nf4": 14}  # everything resident on the GPU
# One component at a time; the transformer must fit with its activations.
MODEL_OFFLOAD_GB = {"bf16": 23, "int8": 15, "nf4": 9}


@dataclass
class LoadedPipeline:
    pipe: object
    device: str  # "cuda" | "cpu"
    device_label: str  # e.g. "RTX 3090 · CPU offload"
    offline: bool  # True when every weight came from the local cache
    placement: str = ""  # "full GPU" | "CPU offload" | "sequential offload" | "CPU"
    device_name: str = ""  # e.g. "RTX 3090"
    quant: str = "bf16"  # precision actually loaded (may differ from the request)


def load_pipeline(
    progress: Callable[[str], None] = lambda msg: None, quant: str = "bf16"
) -> LoadedPipeline:
    import torch
    from diffusers import FluxPipeline

    if quant not in QUANT_MODES:
        raise ValueError(f"unknown precision {quant!r}; expected one of {QUANT_MODES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if quant != "bf16":
        reason = _quant_unavailable(device)
        if reason:
            progress(f"{QUANT_LABELS[quant]} unavailable — {reason}; using bf16.")
            log.warning("quantization %s unavailable: %s", quant, reason)
            quant = "bf16"

    # Quantized components are placed on the GPU as they load, so the free
    # figure that decides placement has to be read before anything loads.
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0.0

    offline = True
    progress(f"Loading FLUX.1-dev ({QUANT_LABELS[quant]}) from the local cache…")
    try:
        pipe = _load(FluxPipeline, dtype, quant, local_files_only=True, progress=progress)
    except Exception:
        offline = False
        progress("Local cache incomplete — downloading FLUX.1-dev (~32 GB)…")
        pipe = _load(FluxPipeline, dtype, quant, local_files_only=False, progress=progress)

    if device == "cuda":
        if free_gb >= FULL_GPU_GB[quant]:
            progress("Moving pipeline to GPU…")
            pipe.to("cuda")
            placement = "full GPU"
        elif free_gb >= MODEL_OFFLOAD_GB[quant] or quant != "bf16":
            # Quantized pipelines never go sequential: the layer-streaming
            # hooks don't compose with bitsandbytes modules, and model
            # offload keeps 8-bit weights resident anyway.
            if free_gb < MODEL_OFFLOAD_GB[quant]:
                progress(
                    f"Only {free_gb:.1f} GiB of VRAM free — {quant} may not fit; "
                    "trying CPU offload…"
                )
            else:
                progress("Enabling CPU offload (one model on the GPU at a time)…")
            pipe.enable_model_cpu_offload()
            placement = "CPU offload"
        else:
            # Not even the transformer fits — stream it layer by layer. Slow,
            # but it works alongside whatever is already using the card.
            progress(
                f"Only {free_gb:.1f} GiB of VRAM free — enabling sequential "
                "offload (slower; close other GPU apps or switch to NF4)…"
            )
            pipe.enable_sequential_cpu_offload()
            placement = "sequential offload"
        name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        label = f"{name} · {placement}"
        if quant != "bf16":
            label += f" · {quant}"
        if placement == "sequential offload":
            label += f" · {free_gb:.1f} GiB free"
    else:
        name, placement = "CPU", "CPU"
        label = "CPU (no CUDA — expect minutes per image)"

    # Decode the latents in slices/tiles so large renders don't spike VRAM
    # right at the finish line.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # The GUI drives its own progress bar off the step callback.
    pipe.set_progress_bar_config(disable=True)

    return LoadedPipeline(
        pipe=pipe,
        device=device,
        device_label=label,
        offline=offline,
        placement=placement,
        device_name=name,
        quant=quant,
    )


def _quant_unavailable(device: str) -> str:
    """Why a quantized load can't happen here, or '' when it can."""
    if device != "cuda":
        return "bitsandbytes needs a CUDA GPU"
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return "bitsandbytes is not installed (pip install bitsandbytes)"
    return ""


def _load(cls, dtype, quant: str, local_files_only: bool, progress):
    if quant == "bf16":
        return _from_pretrained(cls, dtype, local_files_only)

    # Quantize the two big components on the way in; the pipeline then takes
    # them ready-made. CLIP and the VAE stay bf16 — they're small, and the VAE
    # is quality-critical.
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import FluxTransformer2DModel
    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import T5EncoderModel

    if quant == "nf4":
        d_cfg = DiffusersBnb(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
        t_cfg = TransformersBnb(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )
    else:
        d_cfg = DiffusersBnb(load_in_8bit=True)
        t_cfg = TransformersBnb(load_in_8bit=True)

    progress(f"Quantizing transformer to {quant}…")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        quantization_config=d_cfg,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    progress(f"Quantizing T5 text encoder to {quant}…")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder_2",
        quantization_config=t_cfg,
        dtype=dtype,  # transformers 5 spells it `dtype`
        local_files_only=local_files_only,
    )
    progress("Loading the rest of the pipeline…")
    return cls.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )


def _from_pretrained(cls, dtype, local_files_only: bool):
    # `torch_dtype` is the name diffusers 0.39 accepts; newer versions rename
    # it to `dtype` but silently IGNORE unknown kwargs rather than raising, so
    # the only safe probe is to check what actually landed on the weights.
    pipe = cls.from_pretrained(
        MODEL_ID, torch_dtype=dtype, local_files_only=local_files_only
    )
    if pipe.transformer.dtype != dtype:
        pipe = pipe.to(dtype=dtype)
    return pipe
