"""Offscreen GUI smoke test: build the full window, poke the panels, screenshot.

Runs with QT_QPA_PLATFORM=offscreen and a stubbed engine load (the model is
never touched), so it verifies wiring and theme, not generation.

Usage: .venv/bin/python scripts/gui_smoke.py [screenshot.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep the smoke test's settings/history away from the real ones.
os.environ["XDG_DATA_HOME"] = os.environ.get("SMOKE_DATA_DIR", "/tmp/fluxstudio-smoke")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication


def main() -> int:
    app = QApplication(sys.argv)

    from fluxstudio.ui.metrics import refresh
    from fluxstudio.ui.theme import build_qss

    app.setStyleSheet(build_qss(refresh()))

    # Stub the engine load so the smoke test never touches the model.
    from fluxstudio.ui import main_window as mw
    from fluxstudio.ui import workers

    class FakeHost(workers.EngineHost):
        def load(self) -> None:
            self.loadProgress.emit("(smoke) skipping model load")
            self.loaded = workers.LoadedPipeline(
                pipe=None, device="cpu", device_label="smoke", offline=True
            )
            self.loadReady.emit(self.loaded)

    # MainWindow resolves EngineHost through its module globals, so patching
    # the name there is enough — no model weights are ever touched.
    mw.EngineHost = FakeHost
    window = mw.MainWindow()
    window.show()

    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(("ok  " if condition else "FAIL") + f"  {name}")
        if not condition:
            failures.append(name)

    def poke() -> None:
        check("three panes present", all(
            w is not None
            for w in (window.prompt_panel, window.params_panel,
                      window.preview, window.gallery)
        ))
        check("status bar built", window.status_label is not None)

        # Params round-trip
        window.params_panel.apply({"width": 1344, "height": 768,
                                   "steps": 20, "guidance": 4.0})
        values = window.params_panel.values()
        check("params round-trip", values == {"width": 1344, "height": 768,
                                              "steps": 20, "guidance": 4.0})
        check("aspect combo tracked preset",
              window.params_panel.aspect_combo.currentText().startswith("16:9"))
        window.params_panel.width_spin.setValue(1000)
        check("hand edit flips to Custom",
              window.params_panel.aspect_combo.currentText() == "Custom")
        check("values snap to /16", window.params_panel.values()["width"] == 992)

        # Prompt busy state
        window.prompt_panel.set_busy(True)
        check("busy hides generate", not window.prompt_panel.generate_button.isEnabled())
        check("busy shows cancel", window.prompt_panel.cancel_button.isVisible())
        window.prompt_panel.set_busy(False)

        # Seed lock
        window.params_panel.show_seed(1234)
        window.params_panel.seed_lock.setChecked(True)
        check("locked seed stable", window.params_panel.effective_seed() == 1234)
        window.params_panel.seed_lock.setChecked(False)
        check("unlocked seed rolls", window.params_panel.effective_seed() != 1234)

        # Gallery empty state
        check("gallery counts", window.gallery.count_label.text().startswith("0"))

        target = sys.argv[1] if len(sys.argv) > 1 else ""
        if target:
            window.grab().save(target)
            print(f"screenshot -> {target}")

        # Exit through closeEvent so the engine thread is stopped — destroying
        # a running QThread is a Qt fatal assert.
        window.close()
        app.exit(1 if failures else 0)

    QTimer.singleShot(300, poke)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
