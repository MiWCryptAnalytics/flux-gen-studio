"""Entry point: python -m fluxstudio"""

from __future__ import annotations

import argparse
import logging
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWidgets import QApplication

from . import APP_NAME


def _configure_logging(verbose: bool) -> None:
    """Everything the engine does goes to stdout, one line per event."""
    try:
        sys.stdout.reconfigure(line_buffering=True)  # prompt output when piped
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Library chatter (diffusers, transformers, urllib3) only at -v.
    for name in ("diffusers", "transformers", "accelerate", "urllib3", "httpx"):
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(prog="fluxstudio", description=APP_NAME)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="also show diffusers/transformers library output",
    )
    parser.add_argument(
        "--tag", default="", metavar="NAME",
        help="label this session's performance records (see Tools ▸ Performance…)",
    )
    parser.add_argument(
        "--model", choices=("flux1", "flux2"), default=None,
        help="which model to load, overriding the saved setting: flux1 (FLUX.1-dev) "
             "or flux2 (FLUX.2-dev, NF4 only, ~20 GiB free VRAM)",
    )
    parser.add_argument(
        "--scale", type=float, default=None, metavar="FACTOR",
        help="UI text size factor for this launch (e.g. 2 on a 4K monitor); "
             "by default the app picks one from the screen's DPI, and "
             "View ▸ Text size remembers a choice",
    )
    parser.add_argument(
        "--quant", choices=("bf16", "nf4", "int8"), default=None,
        help="model precision for this launch, overriding the saved setting "
             "(nf4 needs ~14 GiB free VRAM to run FLUX.1-dev fully on the GPU)",
    )
    args, qt_args = parser.parse_known_args()
    _configure_logging(args.verbose)
    if args.tag:
        from .core.perf import set_tag

        set_tag(args.tag)
        logging.getLogger("fluxstudio").info("performance tag: %s", args.tag)

    # All three must be set before the QApplication exists. Qt derives its
    # scale factor from the X server's Xft.dpi, and a value like 92 would give
    # a fractional 0.96 that leaves every 1px border and glyph resampled and
    # blurry — so round it to an integer and let the font-based scaling below
    # (ui.scaling) handle the size. Only Qt5 needs the attribute; the policy
    # call is missing before 5.14.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.Round
        )
    except AttributeError:
        pass

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName(APP_NAME)

    # Imported after the QApplication so sizing reads the real application font.
    from .core.config import Settings
    from .ui import MainWindow
    from .ui import scaling
    from .ui.fonts import install_application_font
    from .ui.metrics import refresh
    from .ui.theme import build_qss

    # Order matters: the bundled typeface first (family change only), then the
    # scale (multiplies the size), then metrics/QSS which bake both into every
    # derived dimension.
    install_application_font(app)
    factor, source = scaling.resolve_factor(app, cli=args.scale, saved=Settings.load().ui_scale)
    if source == "auto":
        scaling.apply_auto_scale(app)
    else:
        scaling.apply_scale(app, factor)
        logging.getLogger("fluxstudio").info("UI scale x%g (%s)", factor, source)
    app.setStyleSheet(build_qss(refresh()))

    # The window replaces itself when the text size changes (see
    # MainWindow.rescale), so the live one is tracked on the application.
    app.studio_window = MainWindow(quant=args.quant, model=args.model)
    app.studio_window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
