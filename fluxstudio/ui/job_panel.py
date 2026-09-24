"""Left column, between prompt and image: what the engine is doing right now.

Shown from the moment a job is submitted, so the user gets feedback before
the first denoise step — with sequential offload, prompt encoding alone can
take half a minute during which the status bar used to say nothing new.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .metrics import metrics
from .workers import BATCH, VARIATIONS, GenJob

PENDING, ACTIVE, DONE, SKIPPED = "·", "▶", "✓", "–"
LIST_ROWS = 5  # visible rows before the list scrolls


def _label(text: str, role: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", role)
    return label


class JobPanel(QFrame):
    dismissed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.job: GenJob | None = None
        self._started = 0.0
        self._steps_done = 0
        self._renders = 0

        m = metrics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(0.7), m.sp(0.6), m.sp(0.7), m.sp(0.6))
        layout.setSpacing(m.sp(0.35))

        header = QHBoxLayout()
        self.title_label = _label("", "title")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.eta_label = _label("", "metric")
        header.addWidget(self.eta_label)
        self.close_button = QPushButton("✕")
        self.close_button.setProperty("role", "icon")
        self.close_button.setToolTip("Hide")
        self.close_button.setMinimumWidth(m.icon_button())
        self.close_button.clicked.connect(self._dismiss)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        self.overall_label = _label("", "hint")
        layout.addWidget(self.overall_label)
        self.overall_bar = QProgressBar()
        self.overall_bar.setTextVisible(False)
        layout.addWidget(self.overall_bar)

        self.current_label = _label("", "hint")
        self.current_label.setWordWrap(True)
        layout.addWidget(self.current_label)
        self.step_bar = QProgressBar()
        self.step_bar.setTextVisible(False)
        layout.addWidget(self.step_bar)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.NoSelection)
        self.list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.list)

        self.hide()

    # ---------- lifecycle ----------

    def begin(self, job: GenJob) -> None:
        self.job = job
        self._started = perf_counter()
        self._steps_done = 0
        self._renders = 0
        count = len(job.requests)

        self.title_label.setText(
            {BATCH: "BATCH", VARIATIONS: "VARIATIONS"}.get(job.mode, "RENDERING")
        )
        self.eta_label.setText("")
        self.close_button.setVisible(False)
        self.overall_bar.setRange(0, max(1, job.total_steps))
        self.overall_bar.setValue(0)
        self._set_overall()
        self.current_label.setText("Handing the job to the engine…")
        self._pulse(True)

        self.list.clear()
        for index, request in enumerate(job.requests):
            item = QListWidgetItem(f"{PENDING}  {self._name(index)}")
            item.setToolTip(request.prompt)
            self.list.addItem(item)
        # Exactly as tall as its rows, up to LIST_ROWS; longer batches scroll.
        rows = min(count, LIST_ROWS)
        row_height = self.list.sizeHintForRow(0) if count else 0
        self.list.setFixedHeight(rows * row_height + 2 * self.list.frameWidth())
        self.list.setVisible(count > 1)
        self.show()

    def item_started(self, index: int) -> None:
        self._mark(index, ACTIVE, theme.ACCENT)
        self.list.scrollToItem(self.list.item(index))
        self.current_label.setText(
            f"{self._which(index)}Encoding prompt… the first step follows once "
            "the text encoders have run."
            if self._steps_done == 0
            else f"{self._which(index)}Encoding prompt…"
        )
        self._pulse(True)

    def item_step(self, index: int, step: int, steps: int) -> None:
        if self.job is None:
            return
        self._pulse(False)
        self.step_bar.setRange(0, steps)
        self.step_bar.setValue(step)
        self._steps_done = sum(r.steps for r in self.job.requests[:index]) + step
        self.overall_bar.setValue(self._steps_done)
        self.current_label.setText(
            f"{self._which(index)}Step {step}/{steps} · {self._name(index)}"
        )
        self._set_overall()
        self._set_eta()

    def item_done(self, index: int, elapsed: float) -> None:
        self._renders += 1
        self._mark(index, DONE, theme.GOOD, f"{elapsed:.0f} s")
        self._set_overall()

    def finish(self, outcome: str, message: str = "") -> None:
        """outcome: the job mode when it completed, else 'cancelled' / 'failed'."""
        if self.job is None:
            return
        for index in range(self.list.count()):
            if self.list.item(index).text().startswith((PENDING, ACTIVE)):
                self._mark(index, SKIPPED, theme.TEXT_DIM)
        elapsed = _duration(perf_counter() - self._started)
        planned = len(self.job.requests)
        if outcome == "cancelled":
            self.title_label.setText("CANCELLED")
            self.eta_label.setText(elapsed)
            self.current_label.setText(f"Stopped after {self._renders} of {planned}.")
        elif outcome == "failed":
            self.title_label.setText("FAILED")
            self.eta_label.setText(elapsed)
            self.current_label.setText(message or "The job stopped with an error.")
            self.current_label.setStyleSheet(f"color: {theme.BAD};")
        else:
            self.title_label.setText("DONE")
            self.eta_label.setText(elapsed)
            noun = "render" if self._renders == 1 else "renders"
            self.current_label.setText(f"{self._renders} {noun} finished.")
            self.overall_bar.setValue(self.overall_bar.maximum())
        self._pulse(False)
        self.step_bar.setRange(0, 1)
        self.step_bar.setValue(1 if outcome not in ("cancelled", "failed") else 0)
        self.close_button.setVisible(True)

    # ---------- helpers ----------

    def _dismiss(self) -> None:
        self.hide()
        self.current_label.setStyleSheet("")
        self.dismissed.emit()

    def _name(self, index: int) -> str:
        assert self.job is not None
        request = self.job.requests[index]
        if self.job.mode == BATCH:
            return Path(self.job.output_path(index)).name
        if self.job.mode == VARIATIONS:
            return f"seed {request.seed}"
        words = request.prompt.split()
        return " ".join(words[:6]) + ("…" if len(words) > 6 else "")

    def _which(self, index: int) -> str:
        count = len(self.job.requests) if self.job else 1
        return f"Image {index + 1} of {count} · " if count > 1 else ""

    def _mark(self, index: int, glyph: str, color: str, note: str = "") -> None:
        item = self.list.item(index)
        if item is None:
            return
        suffix = f"   {note}" if note else ""
        item.setText(f"{glyph}  {self._name(index)}{suffix}")
        item.setForeground(QColor(color))

    def _set_overall(self) -> None:
        count = len(self.job.requests) if self.job else 0
        total = max(1, self.overall_bar.maximum())
        percent = 100 * self._steps_done // total
        self.overall_label.setText(f"{self._renders} of {count} images · {percent}%")

    def _set_eta(self) -> None:
        if self._steps_done <= 0 or self.job is None:
            return
        elapsed = perf_counter() - self._started
        remaining = (self.job.total_steps - self._steps_done) * elapsed / self._steps_done
        self.eta_label.setText(f"{_duration(elapsed)} · ~{_duration(remaining)} left")

    def _pulse(self, on: bool) -> None:
        """Indeterminate step bar while the engine is between steps."""
        if on:
            self.step_bar.setRange(0, 0)
        elif self.step_bar.maximum() == 0:
            self.step_bar.setRange(0, 1)
            self.step_bar.setValue(0)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"
