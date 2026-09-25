"""Pick a sensible UI scale on HiDPI screens nobody has configured.

Qt only scales the UI when something tells it to — a Wayland compositor scale,
`Xft.dpi`, or a `QT_*` environment override. A bare X11 session on a 4K monitor
reports 96 logical DPI, so everything renders tiny and the user reaches for
`QT_SCALE_FACTOR=2`. This module makes that reach unnecessary: when nothing else
has scaled the UI, it compares the screen's *physical* DPI against the 96 DPI
baseline and enlarges the application font to match. Every size in the app
derives from that font (see metrics.py), so one font change scales the whole
UI — and, unlike a fractional Qt scale factor, text and 1px borders stay crisp
because nothing is resampled.

The factor comes from the first of these that applies:

1. `--scale N` on the command line.
2. `FLUXSTUDIO_SCALE` — forced factor; `1` disables auto-scaling entirely.
3. A saved View ▸ Text size choice (``Settings.ui_scale``; 0 means auto).
4. Auto: 1.0 when any Qt scaling override (`QT_SCALE_FACTOR`, `QT_FONT_DPI`, …)
   is set, when the devicePixelRatio is already above 1 (Wayland, or a properly
   configured X11), or when the logical DPI is above ~110 (the desktop scales
   through fonts); otherwise physical DPI / 80 in quarter steps, only when the
   screen is meaningfully HiDPI and its EDID-reported size is plausible
   (projectors and TVs routinely lie).

Shared with Voice Design Studio, whose first user on a 159 DPI monitor
preferred x2 over the parity x1.75 — hence the 80 DPI baseline.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping

from PyQt5.QtGui import QFont

# Any of these means the user or desktop already chose a scaling policy.
QT_SCALE_OVERRIDES = (
    "QT_SCALE_FACTOR",
    "QT_SCREEN_SCALE_FACTORS",
    "QT_FONT_DPI",
    "QT_ENABLE_HIGHDPI_SCALING",
    "QT_USE_PHYSICAL_DPI",
)

FORCE_ENV = "FLUXSTUDIO_SCALE"

# The divisor for physical DPI. 96 would be exact physical parity with a
# standard-DPI layout, but parity measured as too small on a real 4K desk, so
# the baseline bakes in ~20% extra: 159/80 lands a 27" 4K at x2 while
# normal-DPI screens (≤ ~100 DPI) still fall below MIN_AUTO_FACTOR.
BASELINE_DPI = 80.0
# Below this logical DPI the desktop clearly hasn't configured font scaling.
CONFIGURED_LOGICAL_DPI = 110.0
# Physical DPI outside this range means the screen is lying about its size.
PLAUSIBLE_PHYSICAL_DPI = (50.0, 400.0)
# Only screens at least this dense count as HiDPI. Keeps the comfort baseline
# from touching ordinary 1080p/1440p monitors (a 27" 1440p is 109 DPI and is
# conventionally run unscaled).
HIDPI_MIN_PHYSICAL_DPI = 120.0
# Don't bother scaling for less than this — it reads as jitter, not intent.
MIN_AUTO_FACTOR = 1.25
FACTOR_RANGE = (0.5, 3.0)
STEP = 0.25  # View ▸ Text size moves in these increments

_base_font: QFont | None = None  # the unscaled application font
_factor = 1.0  # what is currently applied


def clamp(factor: float) -> float:
    low, high = FACTOR_RANGE
    return min(max(factor, low), high)


def pick_scale(
    logical_dpi: float,
    physical_dpi: float,
    device_pixel_ratio: float,
    env: Mapping[str, str],
) -> float:
    """The automatic scale factor for the application font. Pure — no Qt."""
    forced = env.get(FORCE_ENV, "").strip()
    if forced:
        try:
            return clamp(float(forced))
        except ValueError:
            pass  # unparseable — fall through to auto

    if any(env.get(name) for name in QT_SCALE_OVERRIDES):
        return 1.0
    if device_pixel_ratio > 1.001:
        return 1.0
    if logical_dpi > CONFIGURED_LOGICAL_DPI:
        return 1.0
    low, high = PLAUSIBLE_PHYSICAL_DPI
    if not (low <= physical_dpi <= high):
        return 1.0
    if physical_dpi < HIDPI_MIN_PHYSICAL_DPI:
        return 1.0

    factor = physical_dpi / BASELINE_DPI
    if factor < MIN_AUTO_FACTOR:
        return 1.0
    # Quarter steps: enough resolution to fit any screen, coarse enough that
    # two launches on the same monitor can't disagree.
    return clamp(round(factor * 4) / 4)


def base_font(app) -> QFont:
    """The application font before any scaling; captured on first use."""
    global _base_font
    if _base_font is None:
        _base_font = QFont(app.font())
    return QFont(_base_font)


def current_factor() -> float:
    return _factor


def auto_factor(app) -> float:
    """What auto-detection picks for the primary screen (1.0 when headless)."""
    forced = os.environ.get(FORCE_ENV, "").strip()
    if not forced and app.platformName() in ("offscreen", "minimal"):
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    return pick_scale(
        logical_dpi=float(screen.logicalDotsPerInch()),
        physical_dpi=float(screen.physicalDotsPerInch()),
        device_pixel_ratio=float(screen.devicePixelRatio()),
        env=os.environ,
    )


def resolve_factor(app, cli: float | None = None, saved: float = 0.0) -> tuple[float, str]:
    """The factor to apply at launch and where it came from."""
    if cli:
        return clamp(cli), "--scale"
    forced = os.environ.get(FORCE_ENV, "").strip()
    if forced:
        try:
            return clamp(float(forced)), FORCE_ENV
        except ValueError:
            pass
    if saved:
        return clamp(saved), "saved"
    return auto_factor(app), "auto"


def apply_scale(app, factor: float) -> float:
    """Size the application font to ``factor`` × the base font; returns it.

    Call before metrics/QSS are built (or rebuild them afterwards): both read
    the application font and bake it into every derived dimension.
    """
    global _factor
    factor = clamp(factor)
    font = base_font(app)
    if font.pointSizeF() > 0:
        font.setPointSizeF(round(font.pointSizeF() * factor, 2))
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, round(font.pixelSize() * factor)))
    else:
        return 1.0
    app.setFont(font)
    _factor = factor
    return factor


def apply_auto_scale(app) -> float:
    """Detect, apply and report; the one-call form for launch."""
    factor = auto_factor(app)
    apply_scale(app, factor)
    if abs(factor - 1.0) >= 0.01:
        screen = app.primaryScreen()
        geometry = screen.geometry()
        print(
            f"fluxstudio: scaled UI x{factor:g} for {geometry.width()}x"
            f"{geometry.height()} at {screen.physicalDotsPerInch():.0f} DPI "
            f"({FORCE_ENV}=1 to disable, {FORCE_ENV}=<factor> or View ▸ Text size "
            "to override)",
            file=sys.stderr,
        )
    return factor
