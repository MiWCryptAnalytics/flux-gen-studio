"""Left column top: the prompt and the generate controls."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .metrics import metrics


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


class PromptPanel(QFrame):
    generateRequested = pyqtSignal()
    variationsRequested = pyqtSignal(int)
    batchRequested = pyqtSignal()
    cancelRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        header.addWidget(_title("PROMPT"))
        header.addStretch(1)
        self.stats_label = QLabel("")
        self.stats_label.setProperty("role", "hint")
        header.addWidget(self.stats_label)
        layout.addLayout(header)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "Describe the image to generate…\n\n"
            "FLUX rewards specific, literal prose: subject, setting, light,\n"
            "lens, mood. There is no negative prompt — say what you want,\n"
            "not what you don't."
        )
        self.text_edit.setMinimumHeight(m.sp(6.5))
        self.text_edit.textChanged.connect(self._update_stats)
        layout.addWidget(self.text_edit, 1)

        layout.addLayout(self._build_action_row())
        self._update_stats()

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.45))

        self.generate_button = QPushButton("Generate")
        self.generate_button.setProperty("role", "primary")
        self.generate_button.setToolTip("Render one image (Ctrl+Enter)")
        self.generate_button.clicked.connect(self.generateRequested.emit)

        self.variations_spin = QSpinBox()
        self.variations_spin.setRange(2, 8)
        self.variations_spin.setValue(4)
        self.variations_spin.setToolTip("How many seeds to explore")

        self.variations_button = QPushButton("Generate ×N")
        self.variations_button.setToolTip(
            "Render N images from the same prompt with different seeds"
        )
        self.variations_button.clicked.connect(
            lambda: self.variationsRequested.emit(self.variations_spin.value())
        )

        self.batch_button = QPushButton("Batch…")
        self.batch_button.setToolTip(
            "Render every prompt in a JSON file, each to its output_path (Ctrl+B)"
        )
        self.batch_button.clicked.connect(self.batchRequested.emit)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.setToolTip("Stop after the current step (Esc)")
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
        self.cancel_button.setVisible(False)

        row.addWidget(self.generate_button, 2)
        row.addWidget(self.variations_button, 1)
        row.addWidget(self.variations_spin)
        row.addWidget(self.batch_button)
        row.addWidget(self.cancel_button)
        return row

    # ---------- state ----------

    @property
    def prompt(self) -> str:
        return self.text_edit.toPlainText().strip()

    def set_prompt(self, text: str) -> None:
        self.text_edit.setPlainText(text)

    def set_busy(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy)
        self.variations_button.setEnabled(not busy)
        self.variations_spin.setEnabled(not busy)
        # Cancel takes Batch's slot: five controls don't fit the column.
        self.batch_button.setVisible(not busy)
        self.cancel_button.setVisible(busy)

    def _update_stats(self) -> None:
        text = self.prompt
        self.stats_label.setText(f"{len(text)} chars" if text else "empty")
