"""One denoising run: request in, PIL image out, with step progress and cancel."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# FLUX packs 2×2 latent patches over the VAE's 8× factor, so dimensions must be
# multiples of 16 — the pipeline silently rounds otherwise and the saved size
# would no longer match the recipe.
DIM_STEP = 16


@dataclass
class GenRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = 0
    # T5 prompt window; 512 is the model's trained maximum.
    max_sequence_length: int = 512


def new_seed() -> int:
    return random.randint(1, 2**31 - 1)


def run_generation(
    pipe,
    request: GenRequest,
    seed: int,
    on_step: Callable[[int], None],
    cancelled: Callable[[], bool],
):
    """Render one image. Returns a PIL image, or None when cancelled."""
    import torch

    # A CPU generator keeps a seed reproducible across GPUs and placements.
    generator = torch.Generator("cpu").manual_seed(seed)

    def _on_step_end(p, i, t, callback_kwargs):
        on_step(i + 1)
        if cancelled():
            # The pipeline checks this flag each step and fast-forwards the
            # rest of the loop, so this is the cheapest clean exit.
            p._interrupt = True
        return {}

    result = pipe(
        prompt=request.prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.steps,
        guidance_scale=request.guidance,
        max_sequence_length=request.max_sequence_length,
        generator=generator,
        callback_on_step_end=_on_step_end,
    )
    if cancelled():
        return None  # the skipped-ahead latents decode to noise, not an image
    return result.images[0]
