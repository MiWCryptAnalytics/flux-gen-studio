"""Right column: render history, A/B comparison, per-render actions."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.history import Render, RenderHistory
from . import theme
from .metrics import metrics


class RenderCard(QFrame):
    """One render: its thumbnail, its recipe, and what you can do with it."""

    showRequested = pyqtSignal(str)
    starToggled = pyqtSignal(str)
    slotAssigned = pyqtSignal(str, str)  # render id, "A" | "B"
    rerollRequested = pyqtSignal(str)
    restoreRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    deleteRequested = pyqtSignal(str)

    def __init__(
        self,
        render: Render,
        thumbnail: QPixmap | None,
        slot: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.render = render
        self.setProperty("role", "card")
        if slot:
            self.setProperty("slot", slot)
        self.setCursor(Qt.PointingHandCursor)

        m = metrics()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(0.6), m.sp(0.45), m.sp(0.6), m.sp(0.45))
        layout.setSpacing(m.sp(0.35))

        top = QHBoxLayout()
        top.setSpacing(m.sp(0.35))
        title = QLabel(render.title)
        title.setWordWrap(True)
        title.setToolTip(render.prompt or "(no prompt)")
        top.addWidget(title, 1)

        self.star_button = QPushButton("★" if render.starred else "☆")
        self.star_button.setProperty("role", "icon")
        self.star_button.setMinimumWidth(m.icon_button())
        self.star_button.setToolTip("Keep this render")
        if render.starred:
            self.star_button.setStyleSheet(f"color: {theme.STAR};")
        self.star_button.clicked.connect(lambda: self.starToggled.emit(render.id))
        top.addWidget(self.star_button)
        layout.addLayout(top)

        if thumbnail is not None:
            thumb = QLabel()
            thumb.setPixmap(thumbnail)
            thumb.setAlignment(Qt.AlignCenter)
            layout.addWidget(thumb)

        meta = QLabel(
            f"{render.when} · {render.width}×{render.height} · seed {render.seed} · "
            f"{render.steps} steps · g{render.guidance:g}"
        )
        meta.setProperty("role", "metric")
        meta.setWordWrap(True)
        layout.addWidget(meta)

        layout.addLayout(self._build_actions())

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.showRequested.emit(self.render.id)
        super().mousePressEvent(event)

    def _build_actions(self) -> QHBoxLayout:
        m = metrics()
        row = QHBoxLayout()
        row.setSpacing(m.sp(0.22))
        size = m.icon_button()

        def button(text: str, tip: str, handler) -> QPushButton:
            btn = QPushButton(text)
            btn.setProperty("role", "icon")
            btn.setToolTip(tip)
            btn.clicked.connect(handler)
            btn.setMinimumWidth(size)
            return btn

        render_id = self.render.id
        row.addWidget(button("👁", "Show in preview",
                             lambda: self.showRequested.emit(render_id)))
        row.addWidget(button("A", "Pin as A", lambda: self.slotAssigned.emit(render_id, "A")))
        row.addWidget(button("B", "Pin as B", lambda: self.slotAssigned.emit(render_id, "B")))
        row.addWidget(button("⟳", "Re-roll: same recipe, new seed",
                             lambda: self.rerollRequested.emit(render_id)))
        row.addWidget(button("↩", "Restore this recipe into the editor",
                             lambda: self.restoreRequested.emit(render_id)))
        row.addStretch(1)
        row.addWidget(button("⤓", "Export PNG…", lambda: self.exportRequested.emit(render_id)))
        row.addWidget(button("✕", "Delete", lambda: self.deleteRequested.emit(render_id)))
        return row


class GalleryPanel(QFrame):
    showRequested = pyqtSignal(str)
    rerollRequested = pyqtSignal(str)
    restoreRequested = pyqtSignal(str)
    exportRequested = pyqtSignal(str)
    renderDeleted = pyqtSignal(str)

    def __init__(self, history: RenderHistory, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("role", "panel")
        self.history = history
        self.slot_a: str | None = None
        self.slot_b: str | None = None
        # Thumbnails survive refreshes; decoding every PNG on each refresh
        # would make the panel crawl once history grows.
        self._thumbs: dict[str, QPixmap] = {}

        m = metrics()
        layout = QVBoxLayout(self)
        pad = m.sp(0.7)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(m.sp(0.45))

        header = QHBoxLayout()
        title = QLabel("RENDERS")
        title.setProperty("role", "title")
        header.addWidget(title)
        header.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setProperty("role", "hint")
        header.addWidget(self.count_label)
        clear = QPushButton("Clear unstarred")
        clear.setProperty("role", "danger")
        clear.clicked.connect(self._clear_unstarred)
        header.addWidget(clear)
        layout.addLayout(header)

        layout.addWidget(self._build_compare_bar())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.cards_layout = QVBoxLayout(self.container)
        self.cards_layout.setContentsMargins(0, 0, m.sp(0.35), 0)
        self.cards_layout.setSpacing(m.sp(0.35))
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll, 1)

        self.refresh()

    def _build_compare_bar(self) -> QFrame:
        m = metrics()
        bar = QFrame()
        bar.setProperty("role", "card")
        row = QHBoxLayout(bar)
        row.setContentsMargins(m.sp(0.45), m.sp(0.35), m.sp(0.45), m.sp(0.35))
        row.setSpacing(m.sp(0.35))
        edge = m.sp(0.18)

        self.a_button = QPushButton("A —")
        self.a_button.setToolTip("Show render pinned as A (Ctrl+1)")
        self.a_button.setStyleSheet(f"border-left: {edge}px solid {theme.SLOT_A};")
        self.a_button.clicked.connect(lambda: self._show_slot("A"))

        self.b_button = QPushButton("B —")
        self.b_button.setToolTip("Show render pinned as B (Ctrl+2)")
        self.b_button.setStyleSheet(f"border-left: {edge}px solid {theme.SLOT_B};")
        self.b_button.clicked.connect(lambda: self._show_slot("B"))

        row.addWidget(self.a_button, 1)
        row.addWidget(self.b_button, 1)
        return bar

    # ---------- data ----------

    def _thumbnail(self, render: Render) -> QPixmap | None:
        cached = self._thumbs.get(render.id)
        if cached is not None:
            return cached
        if not render.exists:
            return None
        source = QPixmap(render.png_path)
        if source.isNull():
            return None
        thumb = source.scaledToWidth(metrics().ch(26), Qt.SmoothTransformation)
        self._thumbs[render.id] = thumb
        return thumb

    def refresh(self) -> None:
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        renders = self.history.all()
        alive = {r.id for r in renders}
        self._thumbs = {k: v for k, v in self._thumbs.items() if k in alive}

        for render in renders:
            slot = "A" if render.id == self.slot_a else (
                "B" if render.id == self.slot_b else ""
            )
            card = RenderCard(render, self._thumbnail(render), slot)
            card.showRequested.connect(self.showRequested.emit)
            card.rerollRequested.connect(self.rerollRequested.emit)
            card.restoreRequested.connect(self.restoreRequested.emit)
            card.exportRequested.connect(self.exportRequested.emit)
            card.starToggled.connect(self._toggle_star)
            card.slotAssigned.connect(self._assign_slot)
            card.deleteRequested.connect(self._delete)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

        starred = sum(1 for r in renders if r.starred)
        self.count_label.setText(
            f"{len(renders)} render{'s' if len(renders) != 1 else ''}"
            + (f" · {starred} kept" if starred else "")
        )
        self._update_compare_bar()

    def _update_compare_bar(self) -> None:
        for slot, button in (("A", self.a_button), ("B", self.b_button)):
            render_id = self.slot_a if slot == "A" else self.slot_b
            render = self.history.get(render_id) if render_id else None
            if render is None:
                button.setText(f"{slot} —")
                button.setEnabled(False)
            else:
                label = render.title
                button.setText(f"{slot}  {label[:28]}{'…' if len(label) > 28 else ''}")
                button.setToolTip(f"{render.prompt}\nseed {render.seed}")
                button.setEnabled(True)

    def _assign_slot(self, render_id: str, slot: str) -> None:
        if slot == "A":
            # Don't let one render occupy both slots — that defeats comparison.
            if self.slot_b == render_id:
                self.slot_b = None
            self.slot_a = render_id
        else:
            if self.slot_a == render_id:
                self.slot_a = None
            self.slot_b = render_id
        self.refresh()

    def show_slot(self, slot: str) -> None:
        self._show_slot(slot)

    def _show_slot(self, slot: str) -> None:
        render_id = self.slot_a if slot == "A" else self.slot_b
        if render_id and self.history.get(render_id):
            self.showRequested.emit(render_id)

    def _toggle_star(self, render_id: str) -> None:
        self.history.toggle_star(render_id)
        self.refresh()

    def _delete(self, render_id: str) -> None:
        if self.slot_a == render_id:
            self.slot_a = None
        if self.slot_b == render_id:
            self.slot_b = None
        self.history.remove(render_id)
        self.renderDeleted.emit(render_id)
        self.refresh()

    def _clear_unstarred(self) -> None:
        removed = [r.id for r in self.history.all() if not r.starred]
        self.history.clear_unstarred()
        remaining = {r.id for r in self.history.all()}
        if self.slot_a not in remaining:
            self.slot_a = None
        if self.slot_b not in remaining:
            self.slot_b = None
        for render_id in removed:
            self.renderDeleted.emit(render_id)
        self.refresh()
