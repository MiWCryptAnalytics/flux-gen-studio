"""Prompt files for batch generation.

A prompt file is a JSON array; each entry names the prompt and where its
render goes::

    [
      {"output_path": "coast/lighthouse.png",
       "prompt": "A lighthouse on a basalt cliff at dusk, long exposure"},
      {"output_path": "/abs/path/boat.png", "prompt": "..."}
    ]

Relative output paths resolve against the prompt file's own folder, so a
project can be moved as a unit. A path without an extension gets ``.png``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("output_path", "prompt")


@dataclass
class PromptEntry:
    prompt: str
    output_path: Path


def parse_prompts(text: str, base_dir: Path | None = None) -> list[PromptEntry]:
    """Validate the JSON and return entries in file order.

    Raises ValueError with a message that names the offending entry.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not valid JSON: {exc.msg} (line {exc.lineno}).") from None
    if not isinstance(data, list):
        raise ValueError(
            'The file must be a JSON array of {"output_path": …, "prompt": …} objects.'
        )

    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    entries: list[PromptEntry] = []
    seen: dict[Path, int] = {}
    for number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {number} is not an object.")
        for key in REQUIRED_KEYS:
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Entry {number} needs a non-empty "{key}" string.')
        prompt = item["prompt"].strip()
        output = Path(item["output_path"].strip()).expanduser()
        if not output.suffix:
            output = output.with_suffix(".png")
        if not output.is_absolute():
            output = base_dir / output
        output = output.resolve()
        if output in seen:
            raise ValueError(
                f"Entries {seen[output]} and {number} both write to {output.name}."
            )
        seen[output] = number
        entries.append(PromptEntry(prompt=prompt, output_path=output))
    return entries


def load_prompt_file(path: str | Path) -> list[PromptEntry]:
    """Read a prompt file. Raises OSError or ValueError with a readable message."""
    path = Path(path)
    try:
        # utf-8-sig also swallows the BOM some editors put at the top.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path.name} is not UTF-8 text ({exc.reason} at byte {exc.start})."
        ) from None
    return parse_prompts(text, base_dir=path.parent)
