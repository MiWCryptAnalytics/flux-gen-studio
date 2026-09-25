"""Dark studio theme — shared look with Voice Design Studio.

Sizes are computed from the application font (see metrics.py) so the UI honours
the user's Qt HiDPI and font settings instead of pinning everything to fixed
pixels.
"""

from __future__ import annotations

from PyQt5.QtGui import QColor

from ..core.config import ASSETS_DIR
from .metrics import Metrics, metrics

ICONS_DIR = ASSETS_DIR / "icons"

# Palette
BG = "#14161a"
PANEL = "#1b1e24"
PANEL_ALT = "#22262e"
BORDER = "#2e333d"
FIELD_BORDER = "#3d4453"  # inputs stand out from panel edges
TEXT = "#e6e9ef"
TEXT_DIM = "#8d95a5"
ACCENT = "#5b9cff"
ACCENT_DIM = "#3d6dbf"
GOOD = "#4ec9a5"
WARN = "#e0a35c"
BAD = "#e0655c"
STAR = "#ffc857"

VIEW_BG = QColor(17, 19, 23)  # preview canvas behind the image

SLOT_A = "#5b9cff"
SLOT_B = "#c98bff"


def build_qss(m: Metrics | None = None) -> str:
    m = m or metrics()

    pad_y = m.sp(0.35)
    pad_x = m.sp(0.6)
    radius = m.sp(0.35)
    radius_sm = m.sp(0.25)
    field_h = m.sp(1.25)
    bar_h = m.sp(0.35)
    check_box = m.sp(0.85)
    scroll_w = m.sp(0.55)
    arrow = m.sp(0.5)  # chevrons on spin boxes and combos, sized from the font
    up = (ICONS_DIR / "chevron-up.svg").as_posix()
    down = (ICONS_DIR / "chevron-down.svg").as_posix()

    return f"""
/* The base font is pinned to the application font on purpose. Several widget
   classes (QPushButton, QCheckBox, QLabel, item views, …) otherwise resolve
   their font from platform-theme class defaults captured at login and ignore
   a scaled application font entirely — text renders half-size at a 2x UI
   scale. m.pt(1.0) IS the application font in points, re-derived every time
   the stylesheet is built, so this still follows the user's font/DPI settings
   and View ▸ Text size; it just makes every widget class do so. Never pin a
   *pixel* size here. */
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "{m.family}";
    font-size: {m.pt(1.0)}pt;
}}
QLabel, QCheckBox {{ background: transparent; }}
QLabel[role="title"] {{
    font-size: {m.pt(0.92)}pt;
    font-weight: 600;
    color: {TEXT_DIM};
    letter-spacing: 1px;
}}
QLabel[role="hint"] {{ color: {TEXT_DIM}; font-size: {m.pt(0.86)}pt; }}
QLabel[role="metric"] {{ color: {TEXT_DIM}; font-size: {m.pt(0.86)}pt; }}
/* Names of editable fields: full contrast, so the controls read as controls. */
QLabel[role="label"] {{ color: {TEXT}; font-size: {m.pt(0.9)}pt; }}

QFrame[role="panel"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: {radius}px;
}}
QFrame[role="card"] {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
}}
QFrame[role="card"][selected="true"] {{ border: 1px solid {ACCENT}; }}
QFrame[role="card"][slot="A"] {{ border-left: {m.sp(0.18)}px solid {SLOT_A}; }}
QFrame[role="card"][slot="B"] {{ border-left: {m.sp(0.18)}px solid {SLOT_B}; }}

QPlainTextEdit, QTextEdit, QLineEdit {{
    background: {PANEL_ALT};
    border: 1px solid {FIELD_BORDER};
    border-radius: {radius_sm}px;
    padding: {pad_y}px;
    selection-background-color: {ACCENT_DIM};
}}
QPlainTextEdit:focus, QTextEdit:focus, QLineEdit:focus {{ border: 1px solid {ACCENT}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PANEL_ALT};
    border: 1px solid {FIELD_BORDER};
    border-radius: {radius_sm}px;
    padding: {max(2, pad_y - 1)}px {pad_x}px;
    min-height: {field_h}px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QLineEdit:disabled, QPlainTextEdit:disabled {{
    color: {TEXT_DIM};
    background: {PANEL};
    border-color: {PANEL_ALT};
}}
QCheckBox:disabled {{ color: {TEXT_DIM}; }}
QLabel:disabled {{ color: {TEXT_DIM}; }}
/* Fusion draws spin and combo arrows at a fixed pixel size that vanishes at
   HiDPI scales, so they are drawn from SVG at a size that follows the font. */
QComboBox::drop-down {{ border: none; width: {m.sp(1.1)}px; }}
QComboBox::down-arrow {{ image: url("{down}"); width: {arrow}px; height: {arrow}px; }}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: {m.sp(1.0)}px;
    border: none;
    background: transparent;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{up}"); width: {arrow}px; height: {arrow}px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{down}"); width: {arrow}px; height: {arrow}px;
}}
QSpinBox::up-arrow:disabled, QSpinBox::down-arrow:disabled,
QDoubleSpinBox::up-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QComboBox::down-arrow:disabled {{ image: none; }}
QComboBox QAbstractItemView {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}

QPushButton {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {radius_sm}px;
    padding: {pad_y}px {pad_x}px;
}}
QPushButton:hover {{ border: 1px solid {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}
QPushButton[role="primary"] {{
    background: {ACCENT_DIM};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {ACCENT}; }}
QPushButton[role="primary"]:disabled {{ background: {PANEL_ALT}; border-color: {BORDER}; }}
QPushButton[role="danger"]:hover {{ border: 1px solid {BAD}; color: {BAD}; }}
QPushButton[role="icon"] {{ padding: {max(1, pad_y - 2)}px {max(2, pad_x // 2)}px; }}
QPushButton:checked {{ background: {ACCENT_DIM}; border-color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: {scroll_w}px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: {max(2, scroll_w // 2)}px;
    min-height: {m.sp(1.6)}px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: {scroll_w}px; }}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: {max(2, scroll_w // 2)}px;
    min-width: {m.sp(1.6)}px;
}}

QTableWidget {{
    background: {PANEL};
    alternate-background-color: {PANEL_ALT};
    border: 1px solid {BORDER};
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_DIM};
}}
QHeaderView::section {{
    background: {PANEL_ALT};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: {pad_y}px {pad_x}px;
}}
QTableCornerButton::section {{ background: {PANEL_ALT}; border: none; }}

QProgressBar {{
    background: {PANEL_ALT};
    border: 1px solid {BORDER};
    border-radius: {max(2, bar_h // 2)}px;
    height: {bar_h}px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: {max(1, bar_h // 2)}px; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}

QCheckBox {{ spacing: {m.sp(0.35)}px; }}
QCheckBox::indicator {{
    width: {check_box}px; height: {check_box}px;
    border: 1px solid {FIELD_BORDER};
    border-radius: {radius_sm}px;
    background: {PANEL_ALT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}

/* PyQt5's QMenuBar keeps its native palette unless styled explicitly. */
QMenuBar {{ background: {BG}; color: {TEXT}; }}
QMenuBar::item {{ background: transparent; padding: {pad_y}px {pad_x}px; }}
QMenuBar::item:selected {{ background: {PANEL_ALT}; }}
QMenu {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: {pad_y}px {m.sp(1.2)}px; }}
QMenu::item:selected {{ background: {ACCENT_DIM}; }}
QMenu::item:disabled {{ color: {TEXT_DIM}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: {m.sp(0.2)}px 0; }}

QToolTip {{
    background: {PANEL_ALT};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: {max(2, pad_y - 2)}px;
    font-size: {m.pt(0.92)}pt;
}}
"""
