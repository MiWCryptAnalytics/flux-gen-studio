# FLUX Gen Studio

A small desktop studio for generating images with
[black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev),
built with PyQt5 and Hugging Face `diffusers`. Sister project to
**Voice Design Studio** — same dark studio theme, same three-pane layout,
same keep/compare workflow, but for images instead of voices.

![FLUX Gen Studio — prompt panel, preview, and render gallery](docs/screenshot.png)

## Running

```bash
python3 -m venv --system-site-packages .venv   # reuses system torch/PyQt5
.venv/bin/pip install -r requirements.txt
./run.sh
```

First launch loads FLUX.1-dev from the local Hugging Face cache (no network
needed if it's already there — the status bar says which). Placement adapts to
the VRAM actually free at launch: full GPU (40 GB+), per-component CPU offload
(24 GB cards), or layer-streaming sequential offload when another app — for
example Voice Design Studio — already holds part of the card. Measured on the
RTX 3090 with the voice studio running: ~4.2 s/step, so a 28-step 1024×1024
render takes about two minutes; with the card free, CPU offload is roughly
twice as fast. The status bar shows the active mode and s/step.

## Workflow

- **Prompt** (left): describe the image; FLUX has no negative prompt — say
  what you want. `Ctrl+Enter` renders one image; **Generate ×N** explores N
  random seeds.
- **Image** (left, below): aspect presets around the model's ~1 MP sweet
  spot, steps, guidance, and the seed row — 🎲 rolls a new seed, **Lock**
  reuses it so a prompt edit is the only variable.
- **Preview** (center): the selected render, scaled to fit, never upscaled.
  **Copy** puts it on the clipboard, **Export…** (`Ctrl+S`) saves a PNG.
- **Renders** (right): every generation as a card — click to preview, ★ to
  keep, ⟳ re-roll, ↩ restore the exact recipe into the editor, pin any two
  as **A**/**B** (`Ctrl+1`/`Ctrl+2`) to flip between them in the preview.
  **Clear unstarred** deletes everything you didn't star.
- `Esc` cancels a running job after the current step.

Every PNG carries its full recipe (prompt, seed, steps, guidance, size) in
its metadata, and the same recipe is stored in the history, so any image can
be reproduced or resumed later.

## Where things live

- Renders, exports, history, settings: `~/.local/share/fluxstudio/`
- Model weights: the shared Hugging Face cache (`~/.cache/huggingface/hub`)

## Layout of the code

```
fluxstudio/
  core/      config (paths, settings), history (renders + stars)
  engine/    loader (pipeline placement by VRAM), generate (one denoise run)
  ui/        theme + metrics (shared studio look), panels, workers (QThread host)
```

The model runs entirely on a worker `QThread`; the UI talks to it only
through queued signals, so neither the cold load nor a multi-minute render
ever blocks painting. Cancellation sets the pipeline's `_interrupt` flag from
the denoise callback, which skips the remaining steps cleanly.

## License

MIT — see [LICENSE](LICENSE). Note that the FLUX.1-dev model weights are
distributed under their own [non-commercial license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
