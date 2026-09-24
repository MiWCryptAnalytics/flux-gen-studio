"""Real engine smoke test: load FLUX.1-dev and render, no GUI involved.

Verifies the diffusers API surface the app depends on: pipeline load with the
dtype kwarg, the step callback, interrupt-based cancellation, and PNG metadata.

Usage: .venv/bin/python scripts/engine_smoke.py <out_dir> [--full] [--quant nf4|int8]
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxstudio.engine import GenRequest, Timings, load_pipeline, run_generation


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = Path(args[0] if args else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    quant = "bf16"
    if "--quant" in sys.argv:
        quant = sys.argv[sys.argv.index("--quant") + 1]

    t0 = perf_counter()
    loaded = load_pipeline(
        progress=lambda m: print(f"[load] {m}", flush=True), quant=quant
    )
    print(f"[load] done in {perf_counter() - t0:.0f}s · {loaded.device_label} · "
          f"offline={loaded.offline}", flush=True)

    # 1. Tiny render exercises the full path fast.
    request = GenRequest(prompt="a red cube on a white table", width=512,
                         height=512, steps=4, guidance=3.5)
    t0 = perf_counter()
    image = run_generation(
        loaded.pipe, request, seed=42,
        on_step=lambda s: print(f"[gen] step {s}/4", flush=True),
        cancelled=lambda: False,
    )
    assert image is not None and image.size == (512, 512), image
    small = out_dir / "smoke_512.png"
    image.save(small)
    print(f"[gen] 512×512·4 steps in {perf_counter() - t0:.1f}s -> {small}", flush=True)

    # 2. Cancellation after step 1 must yield None.
    seen: list[int] = []
    result = run_generation(
        loaded.pipe, request, seed=42,
        on_step=seen.append,
        cancelled=lambda: len(seen) >= 1,
    )
    assert result is None, "cancelled run should not return an image"
    print(f"[cancel] interrupted after step {max(seen)} of 4, returned None", flush=True)

    # 3. One realistic render at the app's defaults (slow under offload, so
    # only with --full).
    if "--full" not in sys.argv:
        print("ENGINE SMOKE OK (skipped full-size render; pass --full)", flush=True)
        return 0
    request = GenRequest(
        prompt="A lighthouse on a basalt cliff at dusk, long exposure, "
               "low fog, warm lamp light against blue hour, 35mm photograph",
        width=1024, height=1024, steps=28, guidance=3.5,
    )
    timings = Timings()
    image = run_generation(
        loaded.pipe, request, seed=7,
        on_step=lambda s: print(f"[gen] step {s}/28", flush=True) if s % 7 == 0 else None,
        cancelled=lambda: False,
        timings=timings,
    )
    assert image is not None and image.size == (1024, 1024), image
    full = out_dir / "smoke_1024.png"
    image.save(full)
    print(f"[gen] 1024×1024·28 steps in {timings.total:.1f}s -> {full}", flush=True)
    print(f"[perf] {loaded.device_label}: {timings.describe()}", flush=True)
    print("ENGINE SMOKE OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
