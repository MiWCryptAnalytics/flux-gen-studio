"""Main studio window: layout, engine thread wiring, and the generate loop."""

from __future__ import annotations

import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QGuiApplication, QKeySequence
from PyQt5.QtWidgets import (
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
from ..engine import GenRequest, new_seed
from .gallery_panel import GalleryPanel
from .metrics import metrics
from .params_panel import ParamsPanel
from .prompt_panel import PromptPanel
from .viewer import PreviewPanel
from .workers import SINGLE, VARIATIONS, EngineHost, GenJob, RenderResult


class MainWindow(QMainWindow):
    loadRequested = pyqtSignal()
    jobRequested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        ensure_dirs()

        self.settings = Settings.load()
        self.history = RenderHistory.load()
        self._busy = False

        self.setWindowTitle(APP_NAME)
        self.resize(*self._default_size())
        self._build_ui()
        self._build_shortcuts()
        self._restore_settings()
        self._start_engine()

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
        self.prompt_panel.cancelRequested.connect(self._cancel_job)

        self.params_panel = ParamsPanel()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(m.sp(0.6))
        left_layout.addWidget(self.prompt_panel, 1)
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

        export = tools.addAction("&Export current PNG…")
        export.setShortcut(QKeySequence("Ctrl+S"))
        export.triggered.connect(self._export_current)

        folder = tools.addAction("Open renders &folder")
        folder.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(RENDERS_DIR)))
        )

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
        self.host.jobProgress.connect(self._on_job_progress)
        self.host.renderReady.connect(self._on_render_ready)
        self.host.jobDone.connect(self._on_job_done)
        self.host.jobFailed.connect(self._on_job_failed)

        self.thread.start()
        self.prompt_panel.set_busy(True)
        self.prompt_panel.cancel_button.setVisible(False)
        self.loadRequested.emit()

    def _on_load_progress(self, message: str) -> None:
        self.status_label.setText(message)

    def _on_load_ready(self, loaded) -> None:
        self.status_label.setText(
            "Ready · loaded from local cache" if loaded.offline else "Ready"
        )
        source = "cached" if loaded.offline else "hub"
        self.device_label.setText(f"{loaded.device_label} · {source}")
        self.device_label.setToolTip(
            "Model loaded from the local cache — no network needed."
            if loaded.offline
            else "Model was fetched from the Hugging Face Hub this launch."
        )
        self.prompt_panel.set_busy(False)

    def _on_load_failed(self, message: str) -> None:
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
        self.prompt_panel.set_busy(True)
        self.jobRequested.emit(job)

    def generate_one(self) -> None:
        prompt = self.prompt_panel.prompt
        if not prompt:
            self.status_label.setText("Describe an image to generate.")
            return
        seed = self.params_panel.effective_seed()
        self._submit(
            GenJob(request=self._build_request(prompt, seed), seeds=[seed], mode=SINGLE)
        )

    def generate_variations(self, count: int) -> None:
        prompt = self.prompt_panel.prompt
        if not prompt:
            self.status_label.setText("Describe an image to generate.")
            return
        # Variations explore seed space, so the lock never applies here.
        seeds = [new_seed() for _ in range(count)]
        self._submit(
            GenJob(
                request=self._build_request(prompt, seeds[0]),
                seeds=seeds,
                mode=VARIATIONS,
            )
        )

    def _cancel_job(self) -> None:
        if self._busy:
            self.host.cancel()
            self.status_label.setText("Cancelling after the current step…")

    # ---------- job callbacks ----------

    def _on_job_started(self, total: int) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)

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
        )
        self.history.add(render)
        self.gallery.refresh()
        self.preview.show_render(render)

        per_step = result.elapsed / max(1, request.steps)
        self.perf_label.setText(f"{result.elapsed:.1f}s · {per_step:.2f} s/step")

    def _on_job_done(self, mode: str) -> None:
        self._busy = False
        self.prompt_panel.set_busy(False)
        self.progress.setVisible(False)
        self.status_label.setText("Cancelled" if mode == "cancelled" else "Ready")

    def _on_job_failed(self, message: str) -> None:
        self._busy = False
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
        self._submit(GenJob(request=request, seeds=[seed], mode=SINGLE))

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
        self.status_label.setText(f"Restored recipe from render (seed {render.seed})")

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
