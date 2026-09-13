"""Center column: the big image preview and its actions."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.history import Render
from . import theme
from .metrics import metrics


class ImageView(QWidget):
    """Paints one pixmap scaled to fit, centred on a dark canvas.

    The scaled copy is cached per widget size — repaints are frequent (hover,
    focus) but resizes are rare, so scaling on demand and caching wins.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._scaled: QPixmap | None = None
        self._hint = "Renders appear here — Ctrl+Enter to generate"
        self.setMinimumSize(metrics().sp(10), metrics().sp(10))

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        self._pixmap = pixmap
        self._scaled = None
        self.update()

    def set_hint(self, text: str) -> None:
        self._hint = text
        if self._pixmap is None:
            self.update()

    def resizeEvent(self, event) -> None:
        self._scaled = None
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.VIEW_BG)

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(theme.VIEW_BG.lighter(280))
            painter.drawText(
                self.rect(), int(Qt.AlignCenter | Qt.TextWordWrap), self._hint
            )
            return

        if self._scaled is None:
            # Never upscale — a 1:1 or smaller render should stay crisp.
            target = self._pixmap.size().scaled(self.size(), Qt.KeepAspectRatio)
            if (
                target.width() >= self._pixmap.width()
                or target.height() >= self._pixmap.height()
            ):
                self._scaled = self._pixmap
            else:
                self._scaled = self._pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
        x = (self.width() - self._scaled.width()) // 2
        y = (self.height() - self._scaled.height()) // 2
        painter.drawPixmap(x, y, self._scaled)


class PreviewPanel(QFrame):
    exportRequested = pyqtSignal(str)  # render id

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.render: Render | None = None

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        title = QLabel("PREVIEW")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.meta_label = QLabel("")
        self.meta_label.setProperty("role", "metric")
        header.addWidget(self.meta_label)
        layout.addLayout(header)

        self.view = ImageView()
        layout.addWidget(self.view, 1)

        caption = QHBoxLayout()
        caption.setSpacing(m.sp(0.35))
        self.title_label = QLabel("")
        self.title_label.setProperty("role", "hint")
        self.title_label.setWordWrap(True)
        caption.addWidget(self.title_label, 1)

        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy the image to the clipboard")
        self.copy_button.clicked.connect(self._copy)
        self.copy_button.setEnabled(False)
        caption.addWidget(self.copy_button)

        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("Save a copy as PNG (Ctrl+S)")
        self.export_button.clicked.connect(
            lambda: self.render and self.exportRequested.emit(self.render.id)
        )
        self.export_button.setEnabled(False)
        caption.addWidget(self.export_button)
        layout.addLayout(caption)

    # ---------- state ----------

    def show_render(self, render: Render | None) -> None:
        self.render = render
        if render is None or not render.exists:
            self.view.set_pixmap(None)
            self.meta_label.setText("")
            self.title_label.setText("")
            self.copy_button.setEnabled(False)
            self.export_button.setEnabled(False)
            return
        self.view.set_pixmap(QPixmap(render.png_path))
        self.meta_label.setText(
            f"{render.width}×{render.height} · seed {render.seed} · "
            f"{render.steps} steps · g{render.guidance:g}"
        )
        self.title_label.setText(render.title)
        self.title_label.setToolTip(render.prompt)
        self.copy_button.setEnabled(True)
        self.export_button.setEnabled(True)

    def _copy(self) -> None:
        if self.render is not None and self.render.exists:
            QApplication.clipboard().setPixmap(QPixmap(self.render.png_path))
