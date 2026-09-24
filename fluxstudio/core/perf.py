"""Performance records: one line per render, so speed changes can be measured.

Every finished render appends a JSON object to ``perf.jsonl`` with what was
rendered (size, steps, placement) and how long each phase took. Group runs
with ``--tag`` when trying an optimisation, then compare medians in
Tools ▸ Performance… or straight from the file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import median

from .config import DATA_DIR, ensure_dirs

PERF_FILE = DATA_DIR / "perf.jsonl"

_tag = ""


def set_tag(tag: str) -> None:
    """Label every record this session writes (from ``--tag``)."""
    global _tag
    _tag = tag.strip()


def current_tag() -> str:
    return _tag


@dataclass
class PerfRecord:
    """Timings for one render, in seconds unless the name says otherwise."""

    placement: str  # "full GPU" | "CPU offload" | "sequential offload" | "CPU"
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    mode: str  # single | variations | batch
    prompt_chars: int
    to_first_step: float  # prompt encoding + the first (warm-up) step
    s_per_step: float  # median of the steady-state steps
    step_min: float
    step_max: float
    decode: float  # VAE decode after the last step
    save: float  # PNG encode + write (+ batch copy)
    total: float  # pipeline call plus save
    vram_peak_gb: float  # torch peak allocated during the render, 0 on CPU
    device: str = ""  # GPU name
    tag: str = ""
    quant: str = "bf16"  # model precision: bf16 | nf4 | int8
    when: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def size(self) -> str:
        return f"{self.width}×{self.height}"

    @property
    def encode_estimate(self) -> float:
        """Roughly how long prompt encoding took before the first step ran."""
        return max(0.0, self.to_first_step - self.s_per_step)


def record(rec: PerfRecord) -> None:
    ensure_dirs()
    with PERF_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")


def load_records() -> list[PerfRecord]:
    """Oldest first. Unreadable lines are skipped."""
    try:
        lines = PERF_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    known = set(PerfRecord.__dataclass_fields__)
    records: list[PerfRecord] = []
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            try:
                records.append(PerfRecord(**{k: v for k, v in raw.items() if k in known}))
            except TypeError:
                continue  # a row from an older schema missing a required field
    return records


@dataclass
class Summary:
    """Median timings for one configuration."""

    tag: str
    quant: str
    placement: str
    size: str
    steps: int
    runs: int
    s_per_step: float
    to_first_step: float
    decode: float
    total: float
    vram_peak_gb: float


def summarize(records: list[PerfRecord]) -> list[Summary]:
    """Group by (tag, quant, placement, size, steps); most-run first."""
    groups: dict[tuple, list[PerfRecord]] = {}
    for rec in records:
        key = (rec.tag, rec.quant, rec.placement, rec.size, rec.steps)
        groups.setdefault(key, []).append(rec)
    summaries = [
        Summary(
            tag=tag,
            quant=quant,
            placement=placement,
            size=size,
            steps=steps,
            runs=len(rows),
            s_per_step=median(r.s_per_step for r in rows),
            to_first_step=median(r.to_first_step for r in rows),
            decode=median(r.decode for r in rows),
            total=median(r.total for r in rows),
            vram_peak_gb=max(r.vram_peak_gb for r in rows),
        )
        for (tag, quant, placement, size, steps), rows in groups.items()
    ]
    summaries.sort(key=lambda s: (-s.runs, s.tag, s.quant, s.placement, s.size, s.steps))
    return summaries
