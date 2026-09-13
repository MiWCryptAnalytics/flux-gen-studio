"""Background model host.

The pipeline is loaded and driven entirely inside a worker QThread. The UI
talks to it only through queued signals, so neither the cold load nor a
multi-minute render ever blocks painting.

A job is one request plus a list of seeds: one seed renders one image, several
seeds render a variations batch.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from time import perf_counter

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..core.history import new_png_path
from ..engine import GenRequest, LoadedPipeline, load_pipeline, new_seed, run_generation

# Job modes
SINGLE = "single"          # one seed -> one render
VARIATIONS = "variations"  # one request, many seeds -> a render each


@dataclass
class GenJob:
    request: GenRequest
    seeds: list[int] = field(default_factory=list)
    mode: str = SINGLE
    label: str = ""

    def __post_init__(self) -> None:
        if not self.seeds:
            self.seeds = [self.request.seed or new_seed()]

    @property
    def total_steps(self) -> int:
        return len(self.seeds) * self.request.steps


@dataclass
class RenderResult:
    """A finished render, already saved to disk, ready for the UI to show."""

    png_path: str
    seed: int
    elapsed: float
    request: GenRequest
    label: str = ""


class EngineHost(QObject):
    """Lives on the engine thread; owns the pipeline."""

    loadProgress = pyqtSignal(str)
    loadReady = pyqtSignal(object)  # LoadedPipeline
    loadFailed = pyqtSignal(str)

    jobStarted = pyqtSignal(int)  # total denoise steps across the job
    jobProgress = pyqtSignal(str, int, int)  # message, done, total
    renderReady = pyqtSignal(object)  # RenderResult
    jobDone = pyqtSignal(str)  # mode, or "cancelled"
    jobFailed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.loaded: LoadedPipeline | None = None
        self._cancel = threading.Event()

    @property
    def is_ready(self) -> bool:
        return self.loaded is not None

    def cancel(self) -> None:
        """Thread-safe: called from the UI thread, read by the denoise loop."""
        self._cancel.set()

    @pyqtSlot()
    def load(self) -> None:
        try:
            self.loaded = load_pipeline(progress=self.loadProgress.emit)
        except Exception:
            self.loadFailed.emit(traceback.format_exc(limit=3))
            return
        self.loadReady.emit(self.loaded)

    @pyqtSlot(object)
    def run_job(self, job: GenJob) -> None:
        if self.loaded is None:
            self.jobFailed.emit("Model is not loaded.")
            return

        self._cancel.clear()
        total = job.total_steps
        steps = job.request.steps
        count = len(job.seeds)
        self.jobStarted.emit(total)

        try:
            for index, seed in enumerate(job.seeds):
                if self._cancel.is_set():
                    break

                def on_step(step: int, index: int = index) -> None:
                    which = f" {index + 1}/{count}" if count > 1 else ""
                    self.jobProgress.emit(
                        f"Rendering{which} · step {step}/{steps}",
                        index * steps + step,
                        total,
                    )

                start = perf_counter()
                image = run_generation(
                    self.loaded.pipe,
                    job.request,
                    seed,
                    on_step=on_step,
                    cancelled=self._cancel.is_set,
                )
                if image is None:  # cancelled mid-denoise
                    break

                path = new_png_path()
                image.save(path, pnginfo=_png_metadata(job.request, seed))
                self.renderReady.emit(
                    RenderResult(
                        png_path=str(path),
                        seed=seed,
                        elapsed=perf_counter() - start,
                        request=job.request,
                        label=job.label,
                    )
                )
        except Exception as exc:
            _release_vram()
            if _is_oom(exc):
                self.jobFailed.emit(
                    "The GPU ran out of memory. Try a smaller resolution — "
                    "1024×1024 is the sweet spot for this card."
                )
            else:
                self.jobFailed.emit(traceback.format_exc(limit=3))
            return

        self.jobDone.emit("cancelled" if self._cancel.is_set() else job.mode)


def _png_metadata(request: GenRequest, seed: int):
    """Embed the full recipe in the PNG so an exported file explains itself."""
    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    meta.add_text("prompt", request.prompt)
    meta.add_text("model", "black-forest-labs/FLUX.1-dev")
    meta.add_text("seed", str(seed))
    meta.add_text("steps", str(request.steps))
    meta.add_text("guidance", f"{request.guidance:g}")
    meta.add_text("size", f"{request.width}x{request.height}")
    return meta


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _release_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
