"""Main studio window: layout, engine thread wiring, and the generate loop."""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QKeySequence
from PyQt5.QtWidgets import (
    QActionGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from ..core.config import EXPORTS_DIR, RENDERS_DIR, Settings, ensure_dirs
from ..core.history import Render, RenderHistory
from ..core.prompts import load_prompt_file
from ..engine import (
    DEFAULT_MODEL,
    MODELS,
    QUANT_LABELS,
    QUANT_MODES,
    GenRequest,
    model_label,
    new_seed,
    quant_label,
)
from .gallery_panel import GalleryPanel
from .job_panel import JobPanel
from .metrics import metrics
from .params_panel import ParamsPanel
from .perf_dialog import PerfDialog
from .prompt_panel import PromptPanel
from .viewer import PreviewPanel
from .workers import BATCH, SINGLE, VARIATIONS, EngineHost, GenJob, RenderResult


log = logging.getLogger("fluxstudio.ui")


class MainWindow(QMainWindow):
    loadRequested = pyqtSignal(str, str)  # model key, precision to load
    jobRequested = pyqtSignal(object)

    def __init__(self, quant: str | None = None, model: str | None = None) -> None:
        super().__init__()
        ensure_dirs()

        self.settings = Settings.load()
        # CLI overrides persist like a menu pick.
        if model is not None:
            self.settings.model = model
        if quant is not None:
            self.settings.quant = quant
        if self.settings.model not in MODELS:
            self.settings.model = DEFAULT_MODEL
        if self.settings.quant not in QUANT_MODES:
            self.settings.quant = "bf16"
        if self.settings.quant not in self.spec.quants:
            self.settings.quant = self.spec.default_quant
        self.history = RenderHistory.load()
        self._busy = False
        self._job: GenJob | None = None
        self._job_renders = 0  # finished renders in the current job
        self._sec_per_step: float | None = None  # last measured, for estimates

        self.setWindowTitle(APP_NAME)
        self.resize(*self._default_size())
        self._build_ui()
        self._build_shortcuts()
        self._restore_settings()
        self.params_panel.set_model(self.spec)
        self._start_engine()

    @property
    def spec(self):
        """The ModelSpec the settings point at."""
        return MODELS[self.settings.model]

    # ---------- layout ----------

    def _default_size(self) -> tuple[int, int]:
        """Size from the font, then clamp to the screen it will open on."""
        m = metrics()
        width, height = m.ch(185), m.sp(52)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, int(available.width() * 0.94))
            height = min(height, int(available.height() * 0.92))
        return width, height

    def _build_ui(self) -> None:
        m = metrics()

        self.prompt_panel = PromptPanel()
        self.prompt_panel.generateRequested.connect(self.generate_one)
        self.prompt_panel.variationsRequested.connect(self.generate_variations)
        self.prompt_panel.batchRequested.connect(self.generate_batch)
        self.prompt_panel.cancelRequested.connect(self._cancel_job)

        self.params_panel = ParamsPanel()
        self.job_panel = JobPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(m.sp(0.6))
        left_layout.addWidget(self.prompt_panel, 1)
        left_layout.addWidget(self.job_panel, 0)
        left_layout.addWidget(self.params_panel, 0)

        self.preview = PreviewPanel()
        self.preview.exportRequested.connect(self._export_render)

        self.gallery = GalleryPanel(self.history)
        self.gallery.showRequested.connect(self._show_render)
        self.gallery.rerollRequested.connect(self._reroll)
        self.gallery.restoreRequested.connect(self._restore_recipe)
        self.gallery.exportRequested.connect(self._export_render)
        self.gallery.renderDeleted.connect(self._on_render_deleted)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.preview)
        splitter.addWidget(self.gallery)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)
        width = self._default_size()[0]
        splitter.setSizes(
            [int(width * 0.27), int(width * 0.45), int(width * 0.28)]
        )

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(m.sp(0.6), m.sp(0.6), m.sp(0.6), m.sp(0.35))
        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

        self._build_menu()
        self._build_status_bar()

    def _build_menu(self) -> None:
        tools = self.menuBar().addMenu("&Tools")

        batch = tools.addAction("&Batch generate from file…")
        batch.setShortcut(QKeySequence("Ctrl+B"))
        batch.triggered.connect(self.generate_batch)

        tools.addSeparator()

        export = tools.addAction("&Export current PNG…")
        export.setShortcut(QKeySequence("Ctrl+S"))
        export.triggered.connect(self._export_current)

        folder = tools.addAction("Open renders &folder")
        folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(RENDERS_DIR)))
        )

        tools.addSeparator()

        models = tools.addMenu("&Model")
        self.model_actions = QActionGroup(self)
        self.model_actions.setExclusive(True)
        for key, spec in MODELS.items():
            action = models.addAction(spec.label)
            action.setCheckable(True)
            action.setData(key)
            action.setChecked(key == self.settings.model)
            action.setToolTip(spec.repo)
            self.model_actions.addAction(action)
        self.model_actions.triggered.connect(self._on_model_picked)

        precision = tools.addMenu("Model &precision")
        self.quant_actions = QActionGroup(self)
        self.quant_actions.setExclusive(True)
        for mode in QUANT_MODES:
            action = precision.addAction(QUANT_LABELS[mode])
            action.setCheckable(True)
            action.setData(mode)
            self.quant_actions.addAction(action)
        self.quant_actions.triggered.connect(self._on_quant_picked)
        self._sync_quant_menu()

        perf = tools.addAction("&Performance…")
        perf.setShortcut(QKeySequence("Ctrl+P"))
        perf.triggered.connect(self._show_performance)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()

        self.status_label = QLabel("Starting…")
        bar.addWidget(self.status_label, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(metrics().ch(22))
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

        self.perf_label = QLabel("")
        self.perf_label.setProperty("role", "metric")
        bar.addPermanentWidget(self.perf_label)

        self.device_label = QLabel("")
        self.device_label.setProperty("role", "metric")
        bar.addPermanentWidget(self.device_label)

        self._vram_timer = QTimer(self)
        self._vram_timer.timeout.connect(self._update_vram)
        self._vram_timer.start(2000)

    def _build_shortcuts(self) -> None:
        def bind(sequence: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(handler)

        bind("Ctrl+Return", self.generate_one)
        bind("Ctrl+Enter", self.generate_one)
        bind("Ctrl+1", lambda: self.gallery.show_slot("A"))
        bind("Ctrl+2", lambda: self.gallery.show_slot("B"))
        bind("Escape", self._cancel_job)

    # ---------- engine thread ----------

    def _start_engine(self) -> None:
        self.thread = QThread(self)
        self.host = EngineHost()
        self.host.moveToThread(self.thread)

        self.loadRequested.connect(self.host.load)
        self.jobRequested.connect(self.host.run_job)

        self.host.loadProgress.connect(self._on_load_progress)
        self.host.loadReady.connect(self._on_load_ready)
        self.host.loadFailed.connect(self._on_load_failed)
        self.host.jobStarted.connect(self._on_job_started)
        self.host.itemStarted.connect(self._on_item_started)
        self.host.itemStep.connect(self.job_panel.item_step)
        self.host.jobProgress.connect(self._on_job_progress)
        self.host.renderReady.connect(self._on_render_ready)
        self.host.jobDone.connect(self._on_job_done)
        self.host.jobFailed.connect(self._on_job_failed)

        self.thread.start()
        self._begin_load()

    def _begin_load(self) -> None:
        self._loading = True
        self.prompt_panel.set_busy(True)
        self.prompt_panel.cancel_button.setVisible(False)
        self.model_actions.setEnabled(False)
        self.quant_actions.setEnabled(False)
        self.status_label.setText(
            f"Loading {self.spec.label} ({QUANT_LABELS[self.settings.quant]})…"
        )
        self.device_label.setText("")
        self.loadRequested.emit(self.settings.model, self.settings.quant)

    def _on_model_picked(self, action) -> None:
        key = action.data()
        if key == self.settings.model:
            return
        if self._busy:
            self._check_model_action(self.settings.model)
            self.status_label.setText("Wait for the current job before changing the model.")
            return
        self.settings.model = key
        if self.settings.quant not in self.spec.quants:
            self.settings.quant = self.spec.default_quant
        self._sync_quant_menu()
        self.params_panel.set_model(self.spec, adjust=True)
        self._save_settings_quietly()
        log.info("model set to %s (%s); reloading", key, self.settings.quant)
        self._begin_load()

    def _on_quant_picked(self, action) -> None:
        mode = action.data()
        if mode == self.settings.quant:
            return
        if self._busy:
            self._check_quant_action(self.settings.quant)
            self.status_label.setText("Wait for the current job before changing precision.")
            return
        self.settings.quant = mode
        self._save_settings_quietly()
        log.info("precision set to %s; reloading", mode)
        self._begin_load()

    def _save_settings_quietly(self) -> None:
        try:
            self.settings.save()
        except OSError:
            pass

    def _check_model_action(self, key: str) -> None:
        for action in self.model_actions.actions():
            action.setChecked(action.data() == key)

    def _check_quant_action(self, mode: str) -> None:
        for action in self.quant_actions.actions():
            action.setChecked(action.data() == mode)

    def _sync_quant_menu(self) -> None:
        """Only the precisions the current model can run here are pickable."""
        spec = self.spec
        for action in self.quant_actions.actions():
            mode = action.data()
            action.setText(quant_label(spec.key, mode))
            action.setEnabled(mode in spec.quants)
            action.setChecked(mode == self.settings.quant)
            action.setToolTip(
                "" if mode in spec.quants else
                f"{spec.label} runs only as {QUANT_LABELS[spec.default_quant]} in this studio."
            )

    def _on_load_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_load_ready(self, loaded) -> None:
        self._loading = False
        self.status_label.setText(
            "Ready · loaded from local cache" if loaded.offline else "Ready"
        )
        source = "cached" if loaded.offline else "hub"
        self.device_label.setText(f"{loaded.spec.label} · {loaded.device_label} · {source}")
        self.device_label.setToolTip(
            f"{loaded.spec.repo}\n"
            + ("Loaded from the local cache — no network needed."
               if loaded.offline
               else "Fetched from the Hugging Face Hub this launch.")
        )
        if loaded.quant != self.settings.quant:
            # The loader fell back (no bitsandbytes, no CUDA, or a precision
            # this model can't run here); keep the menu honest.
            requested = QUANT_LABELS[self.settings.quant]
            self.settings.quant = loaded.quant
            self._check_quant_action(loaded.quant)
            self.status_label.setText(
                f"Ready · {QUANT_LABELS[loaded.quant]} — {requested} unavailable here"
            )
        self.model_actions.setEnabled(True)
        self.quant_actions.setEnabled(True)
        self.prompt_panel.set_busy(False)

    def _on_load_failed(self, message: str) -> None:
        self._loading = False
        self.model_actions.setEnabled(True)
        self.quant_actions.setEnabled(True)
        self.status_label.setText("Model failed to load")
        QMessageBox.critical(
            self,
            "Could not load model",
            f"{message}\n\nThe app will stay open, but generation is unavailable.",
        )

    # ---------- generating ----------

    def _build_request(self, prompt: str, seed: int) -> GenRequest:
        return GenRequest(prompt=prompt, seed=seed, **self.params_panel.values())

    def _submit(self, job: GenJob) -> None:
        if self._busy:
            return
        if not self.host.is_ready:
            self.status_label.setText("Model is still loading…")
            return
        self._busy = True
        self._job = job
        self._job_renders = 0
        self.prompt_panel.set_busy(True)
        self.job_panel.begin(job)
        self.status_label.setText("Starting…")
        self.progress.setRange(0, 0)  # pulse until the first step reports
        self.progress.setVisible(True)
        self._set_title_progress(0)
        self.jobRequested.emit(job)

    def generate_one(self) -> None:
        prompt = self.prompt_panel.prompt
        if not prompt:
            self.status_label.setText("Describe an image to generate.")
            return
        seed = self.params_panel.effective_seed()
        self._submit(GenJob([self._build_request(prompt, seed)], mode=SINGLE))

    def generate_variations(self, count: int) -> None:
        prompt = self.prompt_panel.prompt
        if not prompt:
            self.status_label.setText("Describe an image to generate.")
            return
        # Variations explore seed space, so the lock never applies here.
        base = self._build_request(prompt, new_seed())
        requests = [base] + [replace(base, seed=new_seed()) for _ in range(count - 1)]
        self._submit(GenJob(requests, mode=VARIATIONS))

    def generate_batch(self) -> None:
        """Render every prompt in a text file with the current image settings."""
        if self._busy:
            return
        if not self.host.is_ready:
            # Reachable from the menu while the buttons are still disabled, so
            # say it somewhere more visible than the status bar.
            QMessageBox.information(
                self,
                "Model still loading",
                "The model is still loading. Batch generation is available "
                "once the status bar says Ready.",
            )
            return

        start_dir = self.settings.batch_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Batch generate from prompt file",
            start_dir,
            "JSON prompt files (*.json);;All files (*)",
        )
        if not path:
            return
        self.settings.batch_dir = str(Path(path).parent)
        name = Path(path).name

        try:
            entries = load_prompt_file(path)
        except (OSError, ValueError) as exc:
            log.warning("batch file %s rejected: %s", path, exc)
            QMessageBox.warning(self, "Could not read prompt file", f"{name}: {exc}")
            return
        log.info("batch file %s: %d entries", path, len(entries))
        if not entries:
            QMessageBox.information(
                self,
                "No prompts found",
                f"{name} is an empty list. Each entry needs an \"output_path\" "
                "and a \"prompt\".",
            )
            return

        # The seed lock means what it says: every prompt gets the same seed, so
        # the prompt is the only variable across the batch. Unlocked, each
        # prompt rolls its own.
        params = self.params_panel.values()
        if self.params_panel.seed_locked:
            seed_note = f"all with locked seed {self.params_panel.seed_spin.value()}"
        else:
            seed_note = "each with a fresh random seed"
        count = len(entries)
        summary = (
            f"Render {count} prompt{'s' if count != 1 else ''} from {name}?\n\n"
            f"{params['width']}×{params['height']} · {params['steps']} steps · "
            f"guidance {params['guidance']:g}, {seed_note}.\n"
            "Each image is written to its entry's output_path."
        )
        existing = sum(1 for e in entries if e.output_path.exists())
        if existing:
            summary += (
                f"\n\n{existing} output file{'s' if existing != 1 else ''} already "
                f"exist{'' if existing != 1 else 's'} and will be overwritten."
            )
        estimate = self._estimate(count * params["steps"])
        if estimate:
            summary += f"\n\nAbout {estimate} at the last measured speed."
        answer = QMessageBox.question(
            self,
            "Batch generate",
            summary,
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            log.info("batch declined")
            return

        requests = [
            self._build_request(entry.prompt, self.params_panel.effective_seed())
            for entry in entries
        ]
        outputs = [str(entry.output_path) for entry in entries]
        self._submit(GenJob(requests, mode=BATCH, output_paths=outputs))

    def _estimate(self, total_steps: int) -> str:
        """Human duration for a job, or '' before any render has been timed."""
        if self._sec_per_step is None:
            return ""
        seconds = total_steps * self._sec_per_step
        if seconds < 90:
            return f"{seconds:.0f} s"
        minutes = round(seconds / 60)
        if minutes < 90:
            return f"{minutes} min"
        hours, minutes = divmod(minutes, 60)
        return f"{hours} h {minutes:02d} min"

    def _cancel_job(self) -> None:
        if self._busy:
            log.info("cancel requested")
            self.host.cancel()
            self.status_label.setText("Cancelling after the current step…")

    # ---------- job callbacks ----------

    def _on_job_started(self, total: int) -> None:
        self.progress.setVisible(True)

    def _on_item_started(self, index: int, prefix: str) -> None:
        self.job_panel.item_started(index)
        self.status_label.setText(f"{prefix} · encoding prompt…")

    def _set_title_progress(self, done: int) -> None:
        planned = len(self._job.requests) if self._job else 0
        if planned > 1:
            self.setWindowTitle(f"[{done}/{planned}] {APP_NAME}")
        elif planned == 1:
            self.setWindowTitle(f"[rendering] {APP_NAME}")
        else:
            self.setWindowTitle(APP_NAME)

    def _on_job_progress(self, message: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(message)

    def _on_render_ready(self, result: RenderResult) -> None:
        request = result.request
        render = Render(
            prompt=request.prompt,
            seed=result.seed,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            png_path=result.png_path,
            elapsed=result.elapsed,
            label=result.label,
            model=result.model,
        )
        self.history.add(render)
        self.gallery.refresh()
        self.preview.show_render(render)
        self._job_renders += 1
        self.job_panel.item_done(result.index, result.elapsed)
        self._set_title_progress(self._job_renders)

        per_step = result.elapsed / max(1, request.steps)
        self._sec_per_step = per_step
        self.perf_label.setText(f"{result.elapsed:.1f}s · {per_step:.2f} s/step")
        if result.timings is not None:
            self.perf_label.setToolTip(
                f"{result.timings.describe()} · save {result.save_seconds:.1f} s\n"
                "Tools ▸ Performance… compares runs."
            )

    def _on_job_done(self, mode: str) -> None:
        self._busy = False
        self.prompt_panel.set_busy(False)
        self.progress.setVisible(False)
        done = self._job_renders
        planned = len(self._job.requests) if self._job else 0
        if mode == "cancelled":
            self.status_label.setText(
                f"Cancelled · {done} of {planned} rendered" if planned > 1 else "Cancelled"
            )
        elif mode == BATCH:
            self.status_label.setText(f"Batch finished · {done} renders")
        else:
            self.status_label.setText("Ready")
        self.job_panel.finish(mode)
        self._job = None
        self._set_title_progress(0)

    def _on_job_failed(self, message: str) -> None:
        self._busy = False
        self.job_panel.finish("failed", message.splitlines()[0] if message else "")
        self._job = None
        self._set_title_progress(0)
        self.prompt_panel.set_busy(False)
        self.progress.setVisible(False)
        self.status_label.setText("Generation failed")
        QMessageBox.warning(self, "Generation failed", message)

    # ---------- render actions ----------

    def _show_render(self, render_id: str) -> None:
        render = self.history.get(render_id)
        if render is None:
            return
        if not render.exists:
            QMessageBox.warning(self, "Missing image", "That render's PNG file is gone.")
            return
        self.preview.show_render(render)

    def _on_render_deleted(self, render_id: str) -> None:
        if self.preview.render is not None and self.preview.render.id == render_id:
            renders = self.history.all()
            self.preview.show_render(renders[0] if renders else None)

    def _reroll(self, render_id: str) -> None:
        render = self.history.get(render_id)
        if render is None:
            return
        seed = new_seed()
        self.params_panel.show_seed(seed)
        request = GenRequest(
            prompt=render.prompt,
            width=render.width,
            height=render.height,
            steps=render.steps,
            guidance=render.guidance,
            seed=seed,
        )
        self._submit(GenJob([request], mode=SINGLE))

    def _restore_recipe(self, render_id: str) -> None:
        render = self.history.get(render_id)
        if render is None:
            return
        self.prompt_panel.set_prompt(render.prompt)
        self.params_panel.apply(
            {
                "width": render.width,
                "height": render.height,
                "steps": render.steps,
                "guidance": render.guidance,
            }
        )
        self.params_panel.show_seed(render.seed)
        note = f"Restored recipe from render (seed {render.seed})"
        if render.model and render.model != self.settings.model:
            note += (
                f" — rendered with {model_label(render.model)}; pick it under "
                "Tools ▸ Model to reproduce it exactly"
            )
        self.status_label.setText(note)

    def _show_performance(self) -> None:
        dialog = PerfDialog(self)
        dialog.exec_()

    def _export_current(self) -> None:
        if self.preview.render is None:
            self.status_label.setText("Nothing to export — generate an image first.")
            return
        self._export_render(self.preview.render.id)

    def _export_render(self, render_id: str) -> None:
        render = self.history.get(render_id)
        if render is None or not render.exists:
            return
        suggested = str(EXPORTS_DIR / f"{_safe_name(render.title)}_{render.seed}.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", suggested, "PNG image (*.png)"
        )
        if not path:
            return
        try:
            shutil.copyfile(render.png_path, path)
            log.info("exported %s", path)
            self.status_label.setText(f"Exported to {Path(path).name}")
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    # ---------- settings ----------

    def _update_vram(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                used = torch.cuda.memory_reserved() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.device_label.setToolTip(f"VRAM reserved {used:.1f} / {total:.0f} GiB")
        except Exception:
            pass

    def _restore_settings(self) -> None:
        s = self.settings
        self.prompt_panel.set_prompt(s.prompt)
        self.prompt_panel.variations_spin.setValue(max(2, min(8, s.variations)))
        self.params_panel.apply(
            {
                "width": s.width,
                "height": s.height,
                "steps": s.steps,
                "guidance": s.guidance,
            }
        )
        self.params_panel.seed_lock.setChecked(s.seed_locked)
        if s.seed:
            self.params_panel.show_seed(s.seed)
        if len(s.window_geometry) == 4:
            self.setGeometry(*s.window_geometry)

        renders = self.history.all()
        if renders:
            self.preview.show_render(renders[0])

    def closeEvent(self, event) -> None:
        s = self.settings
        s.prompt = self.prompt_panel.prompt
        s.variations = self.prompt_panel.variations_spin.value()
        s.seed = self.params_panel.seed_spin.value()
        s.seed_locked = self.params_panel.seed_locked
        s.__dict__.update(self.params_panel.values())
        # s.model and s.quant are saved when picked.
        geo = self.geometry()
        s.window_geometry = [geo.x(), geo.y(), geo.width(), geo.height()]
        try:
            s.save()
        except OSError:
            pass

        self.host.cancel()
        self.thread.quit()
        self.thread.wait(5000)
        super().closeEvent(event)


def _safe_name(text: str) -> str:
    keep = [c if c.isalnum() or c in "-_ " else "_" for c in text[:40]]
    return "".join(keep).strip().replace(" ", "_") or "render"
