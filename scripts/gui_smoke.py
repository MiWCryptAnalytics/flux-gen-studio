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

    from fluxstudio.ui import scaling
    from fluxstudio.ui.fonts import install_application_font
    from fluxstudio.ui.metrics import refresh
    from fluxstudio.ui.theme import build_qss

    failures: list[str] = []

    def check(name: str, condition: bool) -> None:
        print(("ok  " if condition else "FAIL") + f"  {name}")
        if not condition:
            failures.append(name)

    # Typeface and HiDPI scale, in the order __main__ applies them.
    check("bundled typeface installs", install_application_font(app) == "IBM Plex Sans")
    BARE = {}
    check("27-inch 4K on bare X11 scales x2", scaling.pick_scale(96, 163, 1.0, BARE) == 2.0)
    check("24-inch 4K rounds to a quarter step", scaling.pick_scale(96, 184, 1.0, BARE) == 2.25)
    check("13-inch 4K laptop caps at 3", scaling.pick_scale(96, 331, 1.0, BARE) == 3.0)
    check("1080p desktop stays unscaled", scaling.pick_scale(96, 92, 1.0, BARE) == 1.0)
    check("27-inch 1440p stays unscaled", scaling.pick_scale(96, 109, 1.0, BARE) == 1.0)
    check("desktop-scaled session left alone",
          scaling.pick_scale(96, 163, 2.0, BARE) == 1.0
          and scaling.pick_scale(144, 163, 1.0, BARE) == 1.0
          and scaling.pick_scale(96, 163, 1.0, {"QT_SCALE_FACTOR": "2"}) == 1.0)
    check("forced env wins", scaling.pick_scale(96, 92, 1.0, {"FLUXSTUDIO_SCALE": "1.5"}) == 1.5)
    forced = os.environ.get("FLUXSTUDIO_SCALE", "")
    check("auto scale is inert headless unless forced",
          scaling.auto_factor(app) == (float(forced) if forced else 1.0))
    base_pt = app.font().pointSizeF()
    scaling.apply_scale(app, 1.5)
    check("scale reaches the application font", abs(app.font().pointSizeF() - base_pt * 1.5) < 0.05)
    scaling.apply_scale(app, float(forced) if forced else 1.0)
    check("base font restored", abs(app.font().pointSizeF() - base_pt * (float(forced) if forced else 1.0)) < 0.05)

    app.setStyleSheet(build_qss(refresh()))

    # Stub the engine load so the smoke test never touches the model.
    from fluxstudio.ui import main_window as mw
    from fluxstudio.ui import workers

    class FakeHost(workers.EngineHost):
        jobs: list = []

        loads: list = []

        def load(self, model: str = "flux1", quant: str = "bf16") -> None:
            self.loads.append((model, quant))
            self.loadProgress.emit("(smoke) skipping model load")
            self.loaded = workers.LoadedPipeline(
                pipe=None, device="cpu", device_label="smoke", offline=True,
                quant=quant, model=model,
            )
            self.loadReady.emit(self.loaded)

        def run_job(self, job) -> None:
            # Record the job and walk the signals a real render would emit
            # for the first item, then stop — nothing is rendered.
            self.jobs.append(job)
            self.jobStarted.emit(job.total_steps)
            self.itemStarted.emit(0, "Batch 1/3 · one")
            self.itemStep.emit(0, 5, job.requests[0].steps)
            self.jobProgress.emit("Batch 1/3 · one · step 5", 5, job.total_steps)
            self.jobDone.emit(job.mode)

    # MainWindow resolves EngineHost through its module globals, so patching
    # the name there is enough — no model weights are ever touched.
    mw.EngineHost = FakeHost
    app.studio_window = mw.MainWindow()
    app.studio_window.show()

    class _Live:
        """Always the current window — View ▸ Text size replaces it."""

        def __getattr__(self, name):
            return getattr(app.studio_window, name)

    window = _Live()

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

        # Prompt file parsing
        import json

        from fluxstudio.core.prompts import load_prompt_file, parse_prompts

        base = Path("/base")
        parsed = parse_prompts(
            json.dumps([
                {"output_path": "a/one", "prompt": "  one  "},
                {"output_path": "/abs/two.png", "prompt": "two"},
            ]),
            base_dir=base,
        )
        check("prompt file parsed", [e.prompt for e in parsed] == ["one", "two"])
        check("relative output resolves + gets .png",
              parsed[0].output_path == base / "a" / "one.png")
        check("absolute output kept", parsed[1].output_path == Path("/abs/two.png"))

        def rejects(text: str, needle: str) -> bool:
            try:
                parse_prompts(text, base_dir=base)
            except ValueError as exc:
                return needle in str(exc)
            return False

        check("rejects non-array", rejects('{"prompt": "x"}', "JSON array"))
        check("rejects missing output_path",
              rejects('[{"prompt": "x"}]', 'Entry 1 needs a non-empty "output_path"'))
        check("rejects empty prompt",
              rejects('[{"output_path": "x", "prompt": " "}]', '"prompt"'))
        check("rejects duplicate output",
              rejects('[{"output_path": "x", "prompt": "a"},'
                      ' {"output_path": "x.png", "prompt": "b"}]', "both write to"))
        check("rejects bad json", rejects("[", "Not valid JSON"))

        # Performance records round-trip and summarise
        from fluxstudio.core import perf

        perf.set_tag("smoke")
        for i in range(3):
            perf.record(perf.PerfRecord(
                placement="CPU offload", width=1024, height=1024, steps=28,
                guidance=3.5, seed=i, mode="single", prompt_chars=40,
                to_first_step=20.0 + i, s_per_step=2.0 + i * 0.1, step_min=1.9,
                step_max=2.3, decode=3.0, save=0.4, total=80.0 + i, vram_peak_gb=21.0,
                device="RTX 3090", tag="smoke", quant="nf4" if i == 2 else "bf16",
                model="flux2" if i == 2 else "flux1",
            ))
        records = perf.load_records()
        check("perf records persisted", len(records) == 3 and records[0].tag == "smoke")
        summary = perf.summarize(records)
        check("perf summary groups by model and precision",
              len(summary) == 2 and summary[0].runs == 2
              and (summary[1].model, summary[1].quant) == ("flux2", "nf4"))
        check("perf summary medians",
              summary[0].s_per_step == 2.05 and summary[0].total == 80.5)
        check("encode estimate", abs(records[0].encode_estimate - 18.0) < 1e-9)
        from fluxstudio.ui.perf_dialog import PerfDialog
        dialog = PerfDialog(app.studio_window)
        check("perf dialog rows",
              dialog.summary_table.rowCount() == 2 and dialog.recent_table.rowCount() == 3)
        check("perf dialog cell", dialog.summary_table.item(0, 7).text() == "2.05")
        check("perf dialog names the model",
              dialog.summary_table.item(1, 1).text() == "FLUX.2-dev")
        dialog.close()

        # Timings arithmetic as the engine fills it
        from fluxstudio.engine import Timings
        t = Timings(step_seconds=[25.0, 4.0, 4.2, 4.1], decode=3.5, total=40.8)
        check("timings s/step is steady median", t.s_per_step == 4.1)
        check("timings encode estimate", abs(t.encode_estimate - 20.9) < 1e-9)
        check("timings describe", "4.10 s/step" in t.describe() and "encode ≈21 s" in t.describe())

        smoke_dir = Path(os.environ["XDG_DATA_HOME"])
        prompt_file = smoke_dir / "smoke-prompts.json"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        entries = [{"output_path": f"out/{n}.png", "prompt": n}
                   for n in ("one", "two", "three")]
        prompt_file.write_bytes(("\ufeff" + json.dumps(entries)).encode("utf-8"))
        loaded = load_prompt_file(prompt_file)
        check("prompt file BOM stripped", [e.prompt for e in loaded] == ["one", "two", "three"])
        check("outputs resolve beside the file",
              loaded[0].output_path == (smoke_dir / "out" / "one.png").resolve())

        # Batch job submission: stub the two dialogs, then drive the real path.
        mw.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: (str(prompt_file), "")
        )
        mw.QMessageBox.question = staticmethod(lambda *a, **k: mw.QMessageBox.Yes)
        window.params_panel.show_seed(777)
        window.params_panel.seed_lock.setChecked(True)
        window.generate_batch()
        check("batch button hidden while busy",
              not window.prompt_panel.batch_button.isVisible())
        check("job panel shown on submit", window.job_panel.isVisible())
        check("job panel lists entries", window.job_panel.list.count() == 3)
        check("status says starting", window.status_label.text() == "Starting…")
        check("status bar pulses before first step", window.progress.maximum() == 0)
        check("title shows batch progress", window.windowTitle().startswith("[0/3]"))
        QTimer.singleShot(300, verify_batch)

    def verify_batch() -> None:
        job = FakeHost.jobs[-1] if FakeHost.jobs else None
        check("batch job reached the engine", job is not None)
        if job is not None:
            check("batch job mode", job.mode == workers.BATCH)
            check("one request per prompt",
                  [r.prompt for r in job.requests] == ["one", "two", "three"])
            check("locked seed shared across batch",
                  {r.seed for r in job.requests} == {777})
            check("output paths travel with the job",
                  [Path(p).name for p in job.output_paths] == ["one.png", "two.png", "three.png"])
            check("batch total steps", job.total_steps == 3 * job.requests[0].steps)
        check("batch done restores idle",
              window.prompt_panel.batch_button.isEnabled() and not window._busy)
        check("batch status reports count",
              window.status_label.text().startswith("Batch finished · 0"))
        panel = window.job_panel
        check("job panel marks finished", panel.title_label.text() == "DONE")
        check("job panel skips unrendered", panel.list.item(1).text().startswith("–"))
        check("job panel close shown", panel.close_button.isVisible())
        check("title restored", window.windowTitle() == mw.APP_NAME)

        # Panel state as it looks while image 2 of 3 is at step 12/28.
        panel.begin(FakeHost.jobs[-1])
        check("list sized to its rows",
              panel.list.height() <= 4 * panel.list.sizeHintForRow(0))
        panel.item_done(0, 118.0)
        panel.item_started(1)
        check("encoding hint before first step",
              "Encoding" in panel.current_label.text())
        panel.item_step(1, 12, 28)
        check("step bar tracks", panel.step_bar.value() == 12 and panel.step_bar.maximum() == 28)
        check("overall counts renders", panel.overall_label.text().startswith("1 of 3"))
        check("eta shown", "left" in panel.eta_label.text())
        window.settings.batch_dir = ""  # don't persist the smoke folder

        window.params_panel.seed_lock.setChecked(False)
        window.generate_batch()
        QTimer.singleShot(300, verify_unlocked)

    def verify_unlocked() -> None:
        job = FakeHost.jobs[-1]
        check("unlocked batch rolls distinct seeds",
              len({r.seed for r in job.requests}) == 3 and 777 not in
              {r.seed for r in job.requests})

        # Model precision menu: picking a mode reloads with it. The reload
        # request is a queued cross-thread signal, so verify after a beat.
        check("initial load used saved model and precision",
              FakeHost.loads == [("flux1", "bf16")])
        checked = [a for a in window.quant_actions.actions() if a.isChecked()]
        check("bf16 checked in menu", len(checked) == 1 and checked[0].data() == "bf16")
        nf4 = next(a for a in window.quant_actions.actions() if a.data() == "nf4")
        nf4.trigger()
        check("reload disables the menu", not window.quant_actions.isEnabled())
        check("precision persisted", window.settings.quant == "nf4")
        QTimer.singleShot(300, verify_quant)

    def verify_quant() -> None:
        check("picking nf4 reloads", FakeHost.loads == [("flux1", "bf16"), ("flux1", "nf4")])
        check("menu re-enabled after load", window.quant_actions.isEnabled())
        check("prompt usable after reload", window.prompt_panel.generate_button.isEnabled())

        # Model menu: FLUX.2 only runs as NF4 here, so bf16/int8 grey out and
        # the untouched steps/guidance follow the model's reference settings.
        window.params_panel.apply({"steps": 28, "guidance": 3.5})
        flux2 = next(a for a in window.model_actions.actions() if a.data() == "flux2")
        flux2.trigger()
        check("model persisted", window.settings.model == "flux2")
        check("model keeps nf4", window.settings.quant == "nf4")
        bf16 = next(a for a in window.quant_actions.actions() if a.data() == "bf16")
        check("bf16 greyed out for FLUX.2", not bf16.isEnabled())
        check("defaults follow the model",
              window.params_panel.values()["steps"] == 50
              and window.params_panel.values()["guidance"] == 4.0)
        window.params_panel.steps_spin.setValue(20)
        QTimer.singleShot(300, verify_model)

    def verify_model() -> None:
        check("picking FLUX.2 reloads", FakeHost.loads[-1] == ("flux2", "nf4"))
        check("device label names the model",
              window.device_label.text().startswith("FLUX.2-dev"))
        flux1 = next(a for a in window.model_actions.actions() if a.data() == "flux1")
        flux1.trigger()
        check("edited steps survive a model switch",
              window.params_panel.values()["steps"] == 20)
        check("quant stays nf4 when FLUX.1 can run it", window.settings.quant == "nf4")
        QTimer.singleShot(300, verify_model_back)

    def verify_model_back() -> None:
        # The menu group is disabled during the reload, so only check once
        # the load has landed.
        check("switching back reloads FLUX.1", FakeHost.loads[-1] == ("flux1", "nf4"))
        check("switching back re-enables bf16",
              next(a for a in window.quant_actions.actions() if a.data() == "bf16").isEnabled())
        window.settings.quant = "bf16"  # don't persist the smoke picks
        window.settings.model = "flux1"

        # View ▸ Text size rebuilds the window around the same engine.
        old = app.studio_window
        host, thread = old.host, old.thread
        before = scaling.current_factor()
        old.rescale(before + scaling.STEP)
        new = app.studio_window
        check("rescale replaces the window", new is not old)
        check("rescale keeps the engine", new.host is host and new.thread is thread)
        check("rescale keeps the model without reloading",
              new.host.is_ready and len(FakeHost.loads) == 4)
        check("rescale grew the font",
              abs(scaling.current_factor() - (before + scaling.STEP)) < 0.01)
        check("readout shows the saved size",
              new.scale_action.text().endswith("(saved)"))
        check("rescaled window is usable", new.prompt_panel.generate_button.isEnabled())
        new.rescale(None)
        check("auto restores", abs(scaling.current_factor() - before) < 0.01
              and app.studio_window.settings.ui_scale == 0.0)
        app.setStyleSheet(build_qss(refresh()))
        finish()

    def finish() -> None:
        # Stage a mid-batch frame so the screenshot shows the panel at work.
        panel = window.job_panel
        panel.begin(FakeHost.jobs[-1])
        panel.item_done(0, 118.0)
        panel.item_started(1)
        panel.item_step(1, 12, 28)
        window.prompt_panel.set_busy(True)
        window.status_label.setText("Batch 2/3 · two · step 12/28")
        app.processEvents()  # let the layout settle before the grab
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
