"""Application paths and persisted settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _data_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
    return Path(base) / "fluxstudio"


DATA_DIR = _data_root()
RENDERS_DIR = DATA_DIR / "renders"
EXPORTS_DIR = DATA_DIR / "exports"
HISTORY_FILE = DATA_DIR / "history.json"
SETTINGS_FILE = DATA_DIR / "settings.json"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RENDERS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Session state restored on launch."""

    prompt: str = "A lighthouse on a basalt cliff at dusk, long exposure, low fog"
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    seed: int = 0
    seed_locked: bool = False
    variations: int = 4
    quant: str = "bf16"  # model precision: bf16 | nf4 | int8 (see engine.loader)
    batch_dir: str = ""  # last folder a prompt file was opened from
    window_geometry: list[int] = field(default_factory=list)

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(SETTINGS_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        ensure_dirs()
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(SETTINGS_FILE)
