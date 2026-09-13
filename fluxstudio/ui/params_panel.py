"""Image parameters and seed control."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..engine import new_seed
from ..engine.generate import DIM_STEP
from .metrics import metrics

DEFAULTS = {
    "width": 1024,
    "height": 1024,
    "steps": 28,
    "guidance": 3.5,
}

# Sizes FLUX.1-dev was tuned around (~1 MP), by aspect ratio.
ASPECT_PRESETS = [
    ("1:1 · 1024×1024", 1024, 1024),
    ("3:4 · 896×1152", 896, 1152),
    ("4:3 · 1152×896", 1152, 896),
    ("9:16 · 768×1344", 768, 1344),
    ("16:9 · 1344×768", 1344, 768),
    ("2:3 · 832×1216", 832, 1216),
    ("3:2 · 1216×832", 1216, 832),
]
CUSTOM = "Custom"


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    return label


class ParamsPanel(QFrame):
    seedChanged = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")

        m = metrics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(0.7), m.sp(0.6), m.sp(0.7), m.sp(0.6))
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        title = QLabel("IMAGE")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.size_hint = QLabel("")
        self.size_hint.setProperty("role", "metric")
        header.addWidget(self.size_hint)
        reset = QPushButton("Reset")
        reset.setProperty("role", "icon")
        reset.clicked.connect(self.reset_defaults)
        header.addWidget(reset)
        layout.addLayout(header)

        layout.addLayout(self._build_aspect_row())

        grid = QGridLayout()
        grid.setHorizontalSpacing(m.sp(0.6))
        grid.setVerticalSpacing(m.sp(0.35))

        self.width_spin = QSpinBox()
        self.width_spin.setRange(256, 2048)
        self.width_spin.setSingleStep(DIM_STEP)
        self.width_spin.setToolTip("Must be a multiple of 16")

        self.height_spin = QSpinBox()
        self.height_spin.setRange(256, 2048)
        self.height_spin.setSingleStep(DIM_STEP)
        self.height_spin.setToolTip("Must be a multiple of 16")

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 50)
        self.steps_spin.setToolTip(
            "Denoising steps. 28 is the quality/speed sweet spot for FLUX.1-dev;\n"
            "50 squeezes out a little more detail at nearly double the time."
        )

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setDecimals(1)
        self.guidance_spin.setToolTip(
            "Distilled guidance scale. 3.5 is the model default; lower is looser\n"
            "and more photographic, higher follows the prompt more literally but\n"
            "can look 'baked'."
        )

        fields = [
            ("Width", self.width_spin),
            ("Height", self.height_spin),
            ("Steps", self.steps_spin),
            ("Guidance", self.guidance_spin),
        ]
        for i, (label, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            cell = QVBoxLayout()
            cell.setSpacing(m.sp(0.12))
            cell.addWidget(_hint(label))
            cell.addWidget(widget)
            container = QWidget()
            container.setLayout(cell)
            grid.addWidget(container, row, col)

        layout.addLayout(grid)
        layout.addLayout(self._build_seed_row())

        for spin in (self.width_spin, self.height_spin):
            spin.valueChanged.connect(self._on_size_edited)
        self.reset_defaults()

    def _build_aspect_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.45))
        row.addWidget(_hint("Aspect"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems([name for name, _, _ in ASPECT_PRESETS])
        self.aspect_combo.addItem(CUSTOM)
        self.aspect_combo.activated.connect(self._on_aspect_chosen)
        row.addWidget(self.aspect_combo, 1)
        return row

    def _on_aspect_chosen(self, index: int) -> None:
        if index >= len(ASPECT_PRESETS):
            return  # Custom: keep whatever the spins say
        _, width, height = ASPECT_PRESETS[index]
        for spin, value in ((self.width_spin, width), (self.height_spin, height)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._update_size_hint()

    def _on_size_edited(self) -> None:
        """Hand-edited sizes flip the combo to Custom unless they match a preset."""
        width, height = self.width_spin.value(), self.height_spin.value()
        for i, (_, w, h) in enumerate(ASPECT_PRESETS):
            if (w, h) == (width, height):
                self.aspect_combo.setCurrentIndex(i)
                break
        else:
            self.aspect_combo.setCurrentText(CUSTOM)
        self._update_size_hint()

    def _update_size_hint(self) -> None:
        mp = self.width_spin.value() * self.height_spin.value() / 1_000_000
        self.size_hint.setText(f"{mp:.1f} MP")

    def _build_seed_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(metrics().sp(0.35))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(0)
        self.seed_spin.valueChanged.connect(self.seedChanged.emit)

        self.seed_lock = QCheckBox("Lock")
        self.seed_lock.setToolTip(
            "Reuse this seed for every render, so a prompt edit is the only\n"
            "thing that changes between generations."
        )

        dice = QPushButton("🎲")
        dice.setProperty("role", "icon")
        dice.setToolTip("New random seed")
        dice.clicked.connect(lambda: self.seed_spin.setValue(new_seed()))

        row.addWidget(_hint("Seed"))
        row.addWidget(self.seed_spin, 1)
        row.addWidget(dice)
        row.addWidget(self.seed_lock)
        return row

    # ---------- state ----------

    @property
    def seed_locked(self) -> bool:
        return self.seed_lock.isChecked()

    def effective_seed(self) -> int:
        """The seed to use for the next render."""
        if self.seed_locked:
            return self.seed_spin.value()
        seed = new_seed()
        self.show_seed(seed)
        return seed

    def show_seed(self, seed: int) -> None:
        self.seed_spin.blockSignals(True)
        self.seed_spin.setValue(seed)
        self.seed_spin.blockSignals(False)

    def values(self) -> dict:
        # Snap to the latent grid; the pipeline would silently round anyway
        # and the recipe should record what was actually rendered.
        width = self.width_spin.value() // DIM_STEP * DIM_STEP
        height = self.height_spin.value() // DIM_STEP * DIM_STEP
        return {
            "width": width,
            "height": height,
            "steps": self.steps_spin.value(),
            "guidance": self.guidance_spin.value(),
        }

    def apply(self, values: dict) -> None:
        self.width_spin.setValue(int(values.get("width", DEFAULTS["width"])))
        self.height_spin.setValue(int(values.get("height", DEFAULTS["height"])))
        self.steps_spin.setValue(int(values.get("steps", DEFAULTS["steps"])))
        self.guidance_spin.setValue(float(values.get("guidance", DEFAULTS["guidance"])))
        self._on_size_edited()

    def reset_defaults(self) -> None:
        self.apply(DEFAULTS)
