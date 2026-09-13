"""Load the FLUX.1-dev pipeline with a placement that fits the machine.

FLUX.1-dev in bf16 is ~33 GB of weights (transformer + T5 + CLIP + VAE), which
does not fit on a 24 GB card all at once. Placement is chosen from the VRAM
actually *free* at load time: everything on the GPU (40 GB+), one component at
a time (`enable_model_cpu_offload`, needs ~23 GB for the transformer), or
layer-by-layer streaming (`enable_sequential_cpu_offload`) when another app —
often the sister voice studio — already holds part of the card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

MODEL_ID = "black-forest-labs/FLUX.1-dev"

# Placement thresholds against *free* VRAM at load time — another app (say, a
# sister studio) may already occupy part of the card.
FULL_GPU_GB = 40  # everything resident on the GPU
MODEL_OFFLOAD_GB = 23  # one component at a time; the bf16 transformer is ~22 GB


@dataclass
class LoadedPipeline:
    pipe: object
    device: str  # "cuda" | "cpu"
    device_label: str  # e.g. "RTX 3090 · CPU offload"
    offline: bool  # True when every weight came from the local cache


def load_pipeline(progress: Callable[[str], None] = lambda msg: None) -> LoadedPipeline:
    import torch
    from diffusers import FluxPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    offline = True
    progress("Loading FLUX.1-dev from the local cache…")
    try:
        pipe = _from_pretrained(FluxPipeline, dtype, local_files_only=True)
    except Exception:
        offline = False
        progress("Local cache incomplete — downloading FLUX.1-dev (~32 GB)…")
        pipe = _from_pretrained(FluxPipeline, dtype, local_files_only=False)

    if device == "cuda":
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        if free_gb >= FULL_GPU_GB:
            progress("Moving pipeline to GPU…")
            pipe.to("cuda")
            placement = "full GPU"
        elif free_gb >= MODEL_OFFLOAD_GB:
            progress("Enabling CPU offload (one model on the GPU at a time)…")
            pipe.enable_model_cpu_offload()
            placement = "CPU offload"
        else:
            # Not even the transformer fits — stream it layer by layer. Slow,
            # but it works alongside whatever is already using the card.
            progress(
                f"Only {free_gb:.1f} GiB of VRAM free — enabling sequential "
                "offload (slower; close other GPU apps for speed)…"
            )
            pipe.enable_sequential_cpu_offload()
            placement = f"sequential offload · {free_gb:.1f} GiB free"
        name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        label = f"{name} · {placement}"
    else:
        label = "CPU (no CUDA — expect minutes per image)"

    # Decode the latents in slices/tiles so large renders don't spike VRAM
    # right at the finish line.
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # The GUI drives its own progress bar off the step callback.
    pipe.set_progress_bar_config(disable=True)

    return LoadedPipeline(pipe=pipe, device=device, device_label=label, offline=offline)


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
