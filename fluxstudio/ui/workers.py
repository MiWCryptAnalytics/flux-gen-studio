"""Background model host.

The pipeline is loaded and driven entirely inside a worker QThread. The UI
talks to it only through queued signals, so neither the cold load nor a
multi-minute render ever blocks painting.

A job is an ordered list of requests, each carrying its own seed: one request
renders one image, the same prompt under several seeds is a variations run,
and one request per entry of a prompt file is a batch. A batch also names an
output path per request; the render is copied there after it lands in the
gallery, so the studio's own history never owns the user's output files.
"""

from __future__ import annotations

import logging
import shutil
import threading
import traceback
from dataclasses import dataclass, field
from time import perf_counter

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from pathlib import Path

from ..core import perf
from ..core.history import new_png_path
from ..engine import (
    GenRequest,
    LoadedPipeline,
    Timings,
    load_pipeline,
    new_seed,
    run_generation,
)

log = logging.getLogger("fluxstudio.engine")

# Job modes
SINGLE = "single"          # one request -> one render
VARIATIONS = "variations"  # one prompt, many seeds -> a render each
BATCH = "batch"            # many prompts from a file -> a render each

# How much of a prompt the status bar shows while a batch runs.
STATUS_PROMPT_CHARS = 48


@dataclass
class GenJob:
    requests: list[GenRequest]
    mode: str = SINGLE
    label: str = ""
    # Batch only: where each render is also written, parallel to `requests`.
    output_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.output_paths and len(self.output_paths) != len(self.requests):
            raise ValueError("output_paths must match requests one to one")
        for request in self.requests:
            if not request.seed:
                request.seed = new_seed()

    def output_path(self, index: int) -> str:
        return self.output_paths[index] if self.output_paths else ""

    @property
    def total_steps(self) -> int:
        return sum(r.steps for r in self.requests)


@dataclass
class RenderResult:
    """A finished render, already saved to disk, ready for the UI to show."""

    png_path: str
    seed: int
    elapsed: float
    request: GenRequest
    label: str = ""
    output_path: str = ""  # batch: the user's copy, already written
    index: int = 0  # position in the job
    timings: Timings | None = None  # phase breakdown of the pipeline call
    save_seconds: float = 0.0


class EngineHost(QObject):
    """Lives on the engine thread; owns the pipeline."""

    loadProgress = pyqtSignal(str)
    loadReady = pyqtSignal(object)  # LoadedPipeline
    loadFailed = pyqtSignal(str)

    jobStarted = pyqtSignal(int)  # total denoise steps across the job
    itemStarted = pyqtSignal(int, str)  # index, status prefix — before encoding
    itemStep = pyqtSignal(int, int, int)  # index, step, steps
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

    @pyqtSlot(str)
    def load(self, quant: str = "bf16") -> None:
        def progress(message: str) -> None:
            log.info("load: %s", message)
            self.loadProgress.emit(message)

        if self.loaded is not None:
            self._unload(progress)

        start = perf_counter()
        try:
            self.loaded = load_pipeline(progress=progress, quant=quant)
        except Exception:
            log.exception("model load failed")
            _release_vram()
            self.loadFailed.emit(traceback.format_exc(limit=3))
            return
        log.info(
            "ready: %s, %s, loaded in %.0f s",
            self.loaded.device_label,
            "local cache" if self.loaded.offline else "hub",
            perf_counter() - start,
        )
        self.loadReady.emit(self.loaded)

    def _unload(self, progress) -> None:
        """Drop the current pipeline and give its VRAM back before a reload."""
        import gc

        progress("Unloading the current model…")
        self.loaded = None
        gc.collect()
        _release_vram()
        try:
            import torch

            if torch.cuda.is_available():
                log.info(
                    "unloaded; %.1f GiB VRAM free", torch.cuda.mem_get_info()[0] / 1024**3
                )
        except Exception:
            pass

    @pyqtSlot(object)
    def run_job(self, job: GenJob) -> None:
        if self.loaded is None:
            self.jobFailed.emit("Model is not loaded.")
            return

        self._cancel.clear()
        total = job.total_steps
        count = len(job.requests)
        done_before = 0  # steps completed by earlier renders in this job
        first = job.requests[0]
        log.info(
            "job: %s, %d image%s, %dx%d, %d steps, guidance %g (%d steps total)",
            job.mode, count, "" if count == 1 else "s", first.width, first.height,
            first.steps, first.guidance, total,
        )
        self.jobStarted.emit(total)
        job_start = perf_counter()

        try:
            for index, request in enumerate(job.requests):
                if self._cancel.is_set():
                    break
                seed = request.seed
                message = _progress_prefix(job, index)
                log.info(
                    "%d/%d start: seed %d%s · %s",
                    index + 1, count, seed,
                    f" → {job.output_path(index)}" if job.output_path(index) else "",
                    _head(request.prompt, 80),
                )
                log.info("%d/%d encoding prompt…", index + 1, count)
                # Emitted before the pipeline runs: prompt encoding happens
                # ahead of the first step and can take a while under offload.
                self.itemStarted.emit(index, message)

                def on_step(
                    step: int, index: int = index, message: str = message,
                    offset: int = done_before, steps: int = request.steps,
                ) -> None:
                    log.info("%d/%d step %d/%d", index + 1, count, step, steps)
                    self.itemStep.emit(index, step, steps)
                    self.jobProgress.emit(
                        f"{message} · step {step}/{steps}", offset + step, total
                    )

                start = perf_counter()
                timings = Timings()
                image = run_generation(
                    self.loaded.pipe,
                    request,
                    seed,
                    on_step=on_step,
                    cancelled=self._cancel.is_set,
                    timings=timings,
                )
                if image is None:  # cancelled mid-denoise
                    log.info("%d/%d cancelled mid-render", index + 1, count)
                    break
                done_before += request.steps
                log.info(
                    "%d/%d done in %.1f s: %s",
                    index + 1, count, timings.total, timings.describe(),
                )

                save_start = perf_counter()
                path = new_png_path()
                image.save(path, pnginfo=_png_metadata(request, seed))
                log.info("%d/%d saved %s", index + 1, count, path)
                output = job.output_path(index)
                if output:
                    _write_output(path, output)
                    log.info("%d/%d wrote %s", index + 1, count, output)
                save_seconds = perf_counter() - save_start
                elapsed = perf_counter() - start
                self._record_perf(job, request, timings, save_seconds, elapsed)
                self.renderReady.emit(
                    RenderResult(
                        png_path=str(path),
                        seed=seed,
                        elapsed=elapsed,
                        request=request,
                        label=job.label,
                        output_path=output,
                        index=index,
                        timings=timings,
                        save_seconds=save_seconds,
                    )
                )
        except OutputError as exc:
            log.error("%s", str(exc).splitlines()[0])
            self.jobFailed.emit(str(exc))
            return
        except Exception as exc:
            _release_vram()
            log.exception("job failed")
            if _is_oom(exc):
                self.jobFailed.emit(
                    "The GPU ran out of memory. Try a smaller resolution — "
                    "1024×1024 is the sweet spot for this card."
                )
            else:
                self.jobFailed.emit(traceback.format_exc(limit=3))
            return

        outcome = "cancelled" if self._cancel.is_set() else "finished"
        log.info("job %s after %.0f s", outcome, perf_counter() - job_start)
        self.jobDone.emit("cancelled" if self._cancel.is_set() else job.mode)

    def _record_perf(
        self, job: GenJob, request: GenRequest, timings: Timings, save: float, total: float
    ) -> None:
        """Append this render to perf.jsonl; never let bookkeeping fail a job."""
        assert self.loaded is not None
        steady = timings.steady_steps
        rec = perf.PerfRecord(
            placement=self.loaded.placement or self.loaded.device_label,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            seed=request.seed,
            mode=job.mode,
            prompt_chars=len(request.prompt),
            to_first_step=round(timings.to_first_step, 3),
            s_per_step=round(timings.s_per_step, 3),
            step_min=round(min(steady), 3) if steady else 0.0,
            step_max=round(max(steady), 3) if steady else 0.0,
            decode=round(timings.decode, 3),
            save=round(save, 3),
            total=round(total, 3),
            vram_peak_gb=round(timings.vram_peak_gb, 2),
            device=self.loaded.device_name,
            tag=perf.current_tag(),
            quant=self.loaded.quant,
        )
        try:
            perf.record(rec)
        except OSError as exc:
            log.warning("could not write perf record: %s", exc)


class OutputError(Exception):
    """A render finished but its output file could not be written."""


def _write_output(rendered: Path, output: str) -> None:
    """Copy a finished render to the batch entry's output_path."""
    target = Path(output)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rendered, target)
    except OSError as exc:
        raise OutputError(
            f"Could not write {target}:\n{exc.strerror or exc}\n\n"
            f"The render is still in the gallery. The rest of the batch was not run."
        ) from exc


def _head(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _progress_prefix(job: GenJob, index: int) -> str:
    """'Rendering', 'Rendering 2/4', or 'Batch 3/12 · <prompt head>'."""
    count = len(job.requests)
    which = f" {index + 1}/{count}" if count > 1 else ""
    if job.mode != BATCH:
        return f"Rendering{which}"
    return f"Batch{which} · {_head(job.requests[index].prompt, STATUS_PROMPT_CHARS)}"


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
