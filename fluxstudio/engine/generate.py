"""One denoising run: request in, PIL image out, with step progress and cancel."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Callable

# FLUX packs 2×2 latent patches over the VAE's 8× factor (both FLUX.1 and
# FLUX.2), so dimensions must be multiples of 16 — the pipeline silently rounds
# otherwise and the saved size would no longer match the recipe.
DIM_STEP = 16


@dataclass
class GenRequest:
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = 0
    # Prompt token window (T5 for FLUX.1, Mistral for FLUX.2); 512 is the
    # trained maximum for both.
    max_sequence_length: int = 512


@dataclass
class Timings:
    """Where the seconds went in one ``run_generation`` call.

    ``step_seconds[0]`` covers prompt encoding plus the first denoise step —
    the pipeline gives no hook between the two — so the steady-state cost is
    the median of the remaining steps.
    """

    step_seconds: list[float] = field(default_factory=list)
    decode: float = 0.0  # last step → image returned (VAE decode)
    total: float = 0.0
    vram_peak_gb: float = 0.0

    @property
    def to_first_step(self) -> float:
        return self.step_seconds[0] if self.step_seconds else 0.0

    @property
    def steady_steps(self) -> list[float]:
        return self.step_seconds[1:] or self.step_seconds

    @property
    def s_per_step(self) -> float:
        return median(self.steady_steps) if self.steady_steps else 0.0

    @property
    def encode_estimate(self) -> float:
        return max(0.0, self.to_first_step - self.s_per_step)

    def describe(self) -> str:
        """One line for the log or a tooltip."""
        steady = self.steady_steps
        if not steady:
            return f"{self.total:.1f} s total"
        parts = [
            f"first step {self.to_first_step:.1f} s (encode ≈{self.encode_estimate:.0f} s)",
            f"{len(self.step_seconds)} steps · {self.s_per_step:.2f} s/step "
            f"(min {min(steady):.2f}, max {max(steady):.2f})",
            f"decode {self.decode:.1f} s",
        ]
        if self.vram_peak_gb:
            parts.append(f"VRAM peak {self.vram_peak_gb:.1f} GiB")
        return " · ".join(parts)


def new_seed() -> int:
    return random.randint(1, 2**31 - 1)


def run_generation(
    pipe,
    request: GenRequest,
    seed: int,
    on_step: Callable[[int], None],
    cancelled: Callable[[], bool],
    timings: Timings | None = None,
):
    """Render one image. Returns a PIL image, or None when cancelled.

    Pass a ``Timings`` to have it filled with the phase breakdown.
    """
    import torch

    # A CPU generator keeps a seed reproducible across GPUs and placements.
    generator = torch.Generator("cpu").manual_seed(seed)

    timings = timings if timings is not None else Timings()
    timings.step_seconds = []
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    last = start

    def _on_step_end(p, i, t, callback_kwargs):
        nonlocal last
        now = perf_counter()
        timings.step_seconds.append(now - last)
        last = now
        on_step(i + 1)
        if cancelled():
            # The pipeline checks this flag each step and fast-forwards the
            # rest of the loop, so this is the cheapest clean exit.
            p._interrupt = True
        return {}

    # Both FluxPipeline and Flux2Pipeline take exactly these keywords.
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
    end = perf_counter()
    timings.decode = end - last
    timings.total = end - start
    if cuda:
        timings.vram_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    if cancelled():
        return None  # the skipped-ahead latents decode to noise, not an image
    return result.images[0]
