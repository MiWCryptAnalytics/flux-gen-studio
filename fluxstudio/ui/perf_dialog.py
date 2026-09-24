"""Tools ▸ Performance…: median timings per configuration, and recent renders.

Reads perf.jsonl. The point is comparison: run the same size and step count
under two placements or two ``--tag`` labels and see the s/step medians side
by side.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.perf import PERF_FILE, PerfRecord, Summary, current_tag, load_records, summarize
from .metrics import metrics

RECENT_ROWS = 40

SUMMARY_COLUMNS = [
    ("Tag", "tag"), ("Precision", "quant"), ("Placement", "placement"),
    ("Size", "size"), ("Steps", "steps"),
    ("Runs", "runs"), ("s/step", "s_per_step"), ("First step", "to_first_step"),
    ("Decode", "decode"), ("Total", "total"), ("VRAM peak", "vram_peak_gb"),
]
RECENT_COLUMNS = [
    ("When", "when"), ("Tag", "tag"), ("Precision", "quant"), ("Placement", "placement"),
    ("Size", "size"),
    ("Steps", "steps"), ("s/step", "s_per_step"), ("First step", "to_first_step"),
    ("Decode", "decode"), ("Save", "save"), ("Total", "total"), ("VRAM peak", "vram_peak_gb"),
]
SECONDS = {"s_per_step", "to_first_step", "decode", "save", "total"}


def _cell(value) -> QTableWidgetItem:
    if isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)
    item = QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if isinstance(value, (int, float)):
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _title(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "title")
    return label


class PerfDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Performance")
        m = metrics()
        self.resize(m.ch(150), m.sp(34))

        layout = QVBoxLayout(self)
        layout.setSpacing(m.sp(0.5))

        layout.addWidget(_title("BY CONFIGURATION · medians"))
        self.summary_table = self._table(SUMMARY_COLUMNS)
        layout.addWidget(self.summary_table, 2)

        layout.addWidget(_title(f"RECENT RENDERS · last {RECENT_ROWS}"))
        self.recent_table = self._table(RECENT_COLUMNS)
        layout.addWidget(self.recent_table, 3)

        footer = QHBoxLayout()
        tag = current_tag()
        hint = QLabel(
            f"This session is tagged “{tag}”." if tag else
            "Launch with --tag NAME to label an experiment's renders."
        )
        hint.setProperty("role", "hint")
        footer.addWidget(hint, 1)
        open_button = QPushButton("Open log file")
        open_button.setToolTip(str(PERF_FILE))
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(PERF_FILE)))
        )
        footer.addWidget(open_button)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        footer.addWidget(refresh)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        footer.addWidget(close)
        layout.addLayout(footer)

        self.refresh()

    def _table(self, columns) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels([name for name, _ in columns])
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        return table

    def refresh(self) -> None:
        records = load_records()
        self._fill(self.summary_table, SUMMARY_COLUMNS, summarize(records))
        self._fill(self.recent_table, RECENT_COLUMNS, list(reversed(records))[:RECENT_ROWS])

    @staticmethod
    def _fill(table: QTableWidget, columns, rows: list[Summary] | list[PerfRecord]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, (_, attr) in enumerate(columns):
                value = getattr(row, attr)
                if attr == "when":
                    value = value.replace("T", " ")
                elif attr == "tag" and not value:
                    value = "—"
                table.setItem(r, c, _cell(value))
