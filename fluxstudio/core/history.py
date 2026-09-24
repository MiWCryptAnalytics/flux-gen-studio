"""Render history: every generated image, its recipe, and a star to keep it."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .config import HISTORY_FILE, RENDERS_DIR, ensure_dirs

MAX_UNSTARRED = 200  # oldest unstarred renders beyond this are pruned


def new_png_path() -> Path:
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RENDERS_DIR / f"{stamp}_{uuid4().hex[:6]}.png"


@dataclass
class Render:
    """One generated image plus everything needed to reproduce it."""

    prompt: str = ""
    seed: int = 0
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    png_path: str = ""
    elapsed: float = 0.0
    label: str = ""
    model: str = ""  # flux1 | flux2; "" for renders from before the field existed
    starred: bool = False
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    when: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))

    @property
    def exists(self) -> bool:
        return bool(self.png_path) and Path(self.png_path).exists()

    @property
    def title(self) -> str:
        if self.label:
            return self.label
        words = self.prompt.split()
        head = " ".join(words[:6])
        return (head + "…") if len(words) > 6 else (head or "Untitled")


class RenderHistory:
    def __init__(self, renders: list[Render] | None = None):
        self._renders: list[Render] = renders or []

    @classmethod
    def load(cls) -> "RenderHistory":
        try:
            raw = json.loads(HISTORY_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in Render.__dataclass_fields__}
        renders = [
            Render(**{k: v for k, v in item.items() if k in known})
            for item in raw
            if isinstance(item, dict)
        ]
        # Drop entries whose PNG vanished — a card that can't show or reload
        # its image is dead weight.
        return cls([r for r in renders if r.exists])

    def save(self) -> None:
        ensure_dirs()
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(r) for r in self._renders], indent=2))
        tmp.replace(HISTORY_FILE)

    # ---------- queries ----------

    def all(self) -> list[Render]:
        """Newest first."""
        return list(reversed(self._renders))

    def get(self, render_id: str) -> Render | None:
        for r in self._renders:
            if r.id == render_id:
                return r
        return None

    # ---------- mutations ----------

    def add(self, render: Render) -> None:
        self._renders.append(render)
        self._prune()
        self.save()

    def toggle_star(self, render_id: str) -> None:
        render = self.get(render_id)
        if render is not None:
            render.starred = not render.starred
            self.save()

    def remove(self, render_id: str) -> None:
        render = self.get(render_id)
        if render is None:
            return
        self._renders.remove(render)
        self._delete_file(render)
        self.save()

    def clear_unstarred(self) -> None:
        for render in [r for r in self._renders if not r.starred]:
            self._delete_file(render)
        self._renders = [r for r in self._renders if r.starred]
        self.save()

    def _prune(self) -> None:
        unstarred = [r for r in self._renders if not r.starred]
        for render in unstarred[:-MAX_UNSTARRED]:
            self._renders.remove(render)
            self._delete_file(render)

    @staticmethod
    def _delete_file(render: Render) -> None:
        try:
            if render.png_path:
                Path(render.png_path).unlink(missing_ok=True)
        except OSError:
            pass  # history stays consistent even if the file is stubborn
