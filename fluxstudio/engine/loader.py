"""Load a FLUX pipeline with a placement that fits the machine.

Two models are supported, described by ``MODELS``:

- **FLUX.1-dev** (12B transformer, T5 + CLIP text encoders) is ~33 GB of
  bf16 weights, which does not fit a 24 GB card at once. Placement is chosen
  from the VRAM actually *free* at load time: everything on the GPU (40 GB+),
  one component at a time (`enable_model_cpu_offload`, needs ~23 GB for the
  transformer), or layer-by-layer streaming (`enable_sequential_cpu_offload`)
  when another app — often the sister voice studio — already holds part of
  the card. NF4 shrinks the transformer to ~6.5 GB and T5 to ~3 GB so the
  whole pipeline fits a 24 GB card; INT8 halves the weights instead.

- **FLUX.2-dev** (32B transformer, a 24B Mistral Small 3.2 text encoder) is
  ~112 GB in bf16: the transformer alone is 64 GB, so on anything short of an
  80 GB card it can only run quantized. The studio runs it as NF4 — the
  transformer is ~18 GB and the text encoder ~15 GB, which together do not fit
  a 24 GB card, so the two take turns on the GPU (model offload; needs about
  20 GB free). Weights come from Hugging Face's ready-made NF4 checkpoint;
  if the bf16 transformer is already in the local cache it is quantized on
  the way in instead, saving the 18 GB download at the cost of a slower load.

Quantized pipelines never use sequential offload — the layer-streaming hooks
do not compose with bitsandbytes modules — so when the card is too full for
model offload, the render will fail with an out-of-memory error rather than
crawl.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("fluxstudio.engine")

# Model precision modes. Quantized modes need bitsandbytes and a CUDA GPU.
QUANT_MODES = ("bf16", "nf4", "int8")
QUANT_LABELS = {
    "bf16": "bf16 (full precision)",
    "nf4": "NF4 (4-bit)",
    "int8": "INT8 (8-bit)",
}


@dataclass(frozen=True)
class ModelSpec:
    """What the loader needs to know about one model."""

    key: str  # settings / CLI name: "flux1" | "flux2"
    label: str  # "FLUX.1-dev"
    repo: str  # bf16 weights on the Hub
    quants: tuple[str, ...]  # precision modes the studio can run this model at
    default_quant: str
    # Placement thresholds against *free* VRAM at load time — another app
    # (say, a sister studio) may already occupy part of the card.
    full_gpu_gb: dict[str, float]  # everything resident on the GPU
    model_offload_gb: dict[str, float]  # one component at a time; the transformer must fit
    download_gb: dict[str, float]  # rough size of a cold download per precision
    steps: int  # the model's reference step count
    guidance: float  # the model's reference guidance scale
    steps_hint: str
    guidance_hint: str
    quant_repo: str = ""  # ready-made NF4 checkpoint (transformer + text encoder)
    license_url: str = ""
    quant_note: dict[str, str] = field(default_factory=dict)  # per-precision remark for menus


MODELS: dict[str, ModelSpec] = {
    "flux1": ModelSpec(
        key="flux1",
        label="FLUX.1-dev",
        repo="black-forest-labs/FLUX.1-dev",
        quants=("bf16", "nf4", "int8"),
        default_quant="bf16",
        # Measured on the 3090: NF4 peaks at 13.1 GiB allocated for a
        # 1024×1024 render (weights ~10 GB plus activations), so 14 GiB free
        # is the floor for keeping it all resident.
        full_gpu_gb={"bf16": 40, "int8": 21, "nf4": 14},
        model_offload_gb={"bf16": 23, "int8": 15, "nf4": 9},
        download_gb={"bf16": 32, "int8": 32, "nf4": 32},
        steps=28,
        guidance=3.5,
        steps_hint=(
            "Denoising steps. 28 is the quality/speed sweet spot for FLUX.1-dev;\n"
            "50 squeezes out a little more detail at nearly double the time."
        ),
        guidance_hint=(
            "Distilled guidance scale. 3.5 is the model default; lower is looser\n"
            "and more photographic, higher follows the prompt more literally but\n"
            "can look 'baked'."
        ),
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        quant_note={"nf4": "~13 GB peak at 1 MP", "int8": "~20 GB peak at 1 MP"},
    ),
    "flux2": ModelSpec(
        key="flux2",
        label="FLUX.2-dev",
        repo="black-forest-labs/FLUX.2-dev",
        quant_repo="diffusers/FLUX.2-dev-bnb-4bit",
        # bf16 needs the 64 GB transformer resident (an 80 GB card) and INT8
        # a 32 GB one; neither is reachable on consumer hardware, and the
        # bitsandbytes modules can't be layer-streamed. Add them here if the
        # studio ever runs on a datacenter card.
        quants=("nf4",),
        default_quant="nf4",
        # NF4: transformer ~18 GB + text encoder ~15 GB + activations. The
        # two can share a card only from ~36 GB; below that they take turns.
        full_gpu_gb={"nf4": 36},
        model_offload_gb={"nf4": 20},
        download_gb={"nf4": 34},
        steps=50,
        guidance=4.0,
        steps_hint=(
            "Denoising steps. 50 is the FLUX.2-dev reference setting; 28 is\n"
            "nearly as good at little more than half the time."
        ),
        guidance_hint=(
            "Distilled guidance scale. 4.0 is the FLUX.2-dev default; lower is\n"
            "looser and more photographic, higher follows the prompt more\n"
            "literally but can look 'baked'."
        ),
        license_url="https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt",
        quant_note={"nf4": "~20 GB free VRAM, CPU offload"},
    ),
}
DEFAULT_MODEL = "flux1"
MODEL_KEYS = tuple(MODELS)

# Backwards-compatible alias: the repo of the default model.
MODEL_ID = MODELS[DEFAULT_MODEL].repo


def model_label(key: str) -> str:
    """Human name for a model key; unknown keys (older history) pass through."""
    spec = MODELS.get(key)
    return spec.label if spec else key


def quant_label(model: str, quant: str) -> str:
    """Menu text for a precision under a model, e.g. 'NF4 (4-bit, ~13 GB peak at 1 MP)'."""
    base = QUANT_LABELS[quant]
    note = MODELS[model].quant_note.get(quant, "") if model in MODELS else ""
    if not note:
        return base
    return base[:-1] + f", {note})" if base.endswith(")") else f"{base} · {note}"


@dataclass
class LoadedPipeline:
    pipe: object
    device: str  # "cuda" | "cpu"
    device_label: str  # e.g. "RTX 3090 · CPU offload"
    offline: bool  # True when every weight came from the local cache
    placement: str = ""  # "full GPU" | "CPU offload" | "sequential offload" | "CPU"
    device_name: str = ""  # e.g. "RTX 3090"
    quant: str = "bf16"  # precision actually loaded (may differ from the request)
    model: str = DEFAULT_MODEL  # key into MODELS

    @property
    def spec(self) -> ModelSpec:
        return MODELS[self.model]


def load_pipeline(
    progress: Callable[[str], None] = lambda msg: None,
    quant: str = "bf16",
    model: str = DEFAULT_MODEL,
) -> LoadedPipeline:
    import torch

    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODEL_KEYS}")
    spec = MODELS[model]
    if quant not in QUANT_MODES:
        raise ValueError(f"unknown precision {quant!r}; expected one of {QUANT_MODES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if quant not in spec.quants:
        progress(
            f"{spec.label} runs as {QUANT_LABELS[spec.default_quant]} in this studio "
            f"({QUANT_LABELS[quant]} needs an 80 GB-class GPU)."
        )
        log.warning("%s does not support %s here; using %s", spec.label, quant, spec.default_quant)
        quant = spec.default_quant

    if quant != "bf16":
        reason = _quant_unavailable(device)
        if reason:
            if "bf16" not in spec.quants:
                raise RuntimeError(
                    f"{spec.label} can only run quantized in this studio, and {reason}."
                )
            progress(f"{QUANT_LABELS[quant]} unavailable — {reason}; using bf16.")
            log.warning("quantization %s unavailable: %s", quant, reason)
            quant = "bf16"

    # Quantized components may be placed on the GPU as they load, so the free
    # figure that decides placement has to be read before anything loads.
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3 if device == "cuda" else 0.0

    if model == "flux2":
        pipe, offline = _load_flux2(spec, dtype, progress)
    else:
        offline = True
        progress(f"Loading {spec.label} ({QUANT_LABELS[quant]}) from the local cache…")
        try:
            pipe = _load_flux1(spec, dtype, quant, local_files_only=True, progress=progress)
        except Exception:
            offline = False
            progress(
                f"Local cache incomplete — downloading {spec.label} "
                f"(~{spec.download_gb[quant]:.0f} GB)…"
            )
            pipe = _load_flux1(spec, dtype, quant, local_files_only=False, progress=progress)

    if device == "cuda":
        if free_gb >= spec.full_gpu_gb[quant]:
            progress("Moving pipeline to GPU…")
            pipe.to("cuda")
            placement = "full GPU"
        elif free_gb >= spec.model_offload_gb[quant] or quant != "bf16":
            # Quantized pipelines never go sequential: the layer-streaming
            # hooks don't compose with bitsandbytes modules, and model
            # offload keeps 8-bit weights resident anyway.
            if free_gb < spec.model_offload_gb[quant]:
                progress(
                    f"Only {free_gb:.1f} GiB of VRAM free — {spec.label} {quant} needs "
                    f"about {spec.model_offload_gb[quant]:.0f} GiB; trying CPU offload "
                    "(close other GPU apps if a render runs out of memory)…"
                )
            else:
                progress("Enabling CPU offload (one model on the GPU at a time)…")
            pipe.enable_model_cpu_offload()
            placement = "CPU offload"
        else:
            # Not even the transformer fits — stream it layer by layer. Slow,
            # but it works alongside whatever is already using the card.
            progress(
                f"Only {free_gb:.1f} GiB of VRAM free — enabling sequential "
                "offload (slower; close other GPU apps or switch to NF4)…"
            )
            pipe.enable_sequential_cpu_offload()
            placement = "sequential offload"
        name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
        label = f"{name} · {placement}"
        if quant != "bf16":
            label += f" · {quant}"
        if placement == "sequential offload":
            label += f" · {free_gb:.1f} GiB free"
    else:
        name, placement = "CPU", "CPU"
        label = "CPU (no CUDA — expect minutes per image)"

    # Decode the latents in slices/tiles so large renders don't spike VRAM
    # right at the finish line.
    for enable in ("enable_slicing", "enable_tiling"):
        if hasattr(pipe.vae, enable):
            getattr(pipe.vae, enable)()

    # The GUI drives its own progress bar off the step callback.
    pipe.set_progress_bar_config(disable=True)

    return LoadedPipeline(
        pipe=pipe,
        device=device,
        device_label=label,
        offline=offline,
        placement=placement,
        device_name=name,
        quant=quant,
        model=model,
    )


def _quant_unavailable(device: str) -> str:
    """Why a quantized load can't happen here, or '' when it can."""
    if device != "cuda":
        return "bitsandbytes needs a CUDA GPU"
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return "bitsandbytes is not installed (pip install bitsandbytes)"
    return ""


# ---------- FLUX.1-dev ----------


def _load_flux1(spec: ModelSpec, dtype, quant: str, local_files_only: bool, progress):
    from diffusers import FluxPipeline

    if quant == "bf16":
        return _from_pretrained(FluxPipeline, spec.repo, dtype, local_files_only)

    # Quantize the two big components on the way in; the pipeline then takes
    # them ready-made. CLIP and the VAE stay bf16 — they're small, and the VAE
    # is quality-critical.
    from diffusers import FluxTransformer2DModel
    from transformers import T5EncoderModel

    d_cfg, t_cfg = _bnb_configs(quant, dtype)
    progress(f"Quantizing transformer to {quant}…")
    transformer = FluxTransformer2DModel.from_pretrained(
        spec.repo,
        subfolder="transformer",
        quantization_config=d_cfg,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    progress(f"Quantizing T5 text encoder to {quant}…")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        spec.repo,
        subfolder="text_encoder_2",
        quantization_config=t_cfg,
        dtype=dtype,  # transformers 5 spells it `dtype`
        local_files_only=local_files_only,
    )
    progress("Loading the rest of the pipeline…")
    return FluxPipeline.from_pretrained(
        spec.repo,
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )


def _bnb_configs(quant: str, dtype):
    """Matching diffusers/transformers bitsandbytes configs for a precision."""
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import BitsAndBytesConfig as TransformersBnb

    if quant == "nf4":
        return (
            DiffusersBnb(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            ),
            TransformersBnb(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            ),
        )
    return DiffusersBnb(load_in_8bit=True), TransformersBnb(load_in_8bit=True)


def _from_pretrained(cls, repo: str, dtype, local_files_only: bool):
    # `torch_dtype` is the name diffusers 0.39 accepts; newer versions rename
    # it to `dtype` but silently IGNORE unknown kwargs rather than raising, so
    # the only safe probe is to check what actually landed on the weights.
    pipe = cls.from_pretrained(repo, torch_dtype=dtype, local_files_only=local_files_only)
    if pipe.transformer.dtype != dtype:
        pipe = pipe.to(dtype=dtype)
    return pipe


# ---------- FLUX.2-dev ----------

# Weight file names diffusers and transformers look for (sharded or not).
DIFFUSERS_WEIGHTS = "diffusion_pytorch_model.safetensors"
TRANSFORMERS_WEIGHTS = "model.safetensors"


def _load_flux2(spec: ModelSpec, dtype, progress):
    """FLUX.2-dev as NF4. Returns (pipeline, offline)."""
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2Pipeline,
        Flux2Transformer2DModel,
    )
    from transformers import Mistral3ForConditionalGeneration, PreTrainedTokenizerFast

    have_nf4_transformer = _cached(spec.quant_repo, "transformer", DIFFUSERS_WEIGHTS)
    have_bf16_transformer = _cached(spec.repo, "transformer", DIFFUSERS_WEIGHTS)
    have_text_encoder = _cached(spec.quant_repo, "text_encoder", TRANSFORMERS_WEIGHTS)
    have_rest = all(
        _cached_file(spec.quant_repo, name)
        for name in ("model_index.json", "vae/config.json", f"vae/{DIFFUSERS_WEIGHTS}",
                     "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json")
    )
    offline = (have_nf4_transformer or have_bf16_transformer) and have_text_encoder and have_rest
    local_files_only = offline
    if offline:
        progress(f"Loading {spec.label} ({QUANT_LABELS['nf4']}) from the local cache…")
    else:
        missing_gb = 0.4 if not have_rest else 0.0
        if not (have_nf4_transformer or have_bf16_transformer):
            missing_gb += 18
        if not have_text_encoder:
            missing_gb += 15.5
        progress(
            f"Local cache incomplete — downloading {spec.label} NF4 (~{missing_gb:.0f} GB)…"
        )
        log.info(
            "FLUX.2-dev cache: nf4 transformer=%s bf16 transformer=%s text encoder=%s rest=%s",
            have_nf4_transformer, have_bf16_transformer, have_text_encoder, have_rest,
        )

    if have_nf4_transformer or not have_bf16_transformer:
        progress("Loading the NF4 transformer (~18 GB)…")
        transformer = Flux2Transformer2DModel.from_pretrained(
            spec.quant_repo,
            subfolder="transformer",
            torch_dtype=dtype,
            device_map="cpu",  # model offload streams it in when needed
            local_files_only=local_files_only,
        )
    else:
        # The bf16 checkpoint is here but the NF4 one is not: quantize on the
        # way in. Reads 64 GB, so it is markedly slower than the ready-made
        # checkpoint — but it saves an 18 GB download.
        progress("Quantizing the bf16 transformer to NF4 (reads 64 GB — the ready-made "
                 "NF4 checkpoint loads faster)…")
        log.info("quantizing %s/transformer on the fly; %s/transformer would load faster",
                 spec.repo, spec.quant_repo)
        d_cfg, _ = _bnb_configs("nf4", dtype)
        transformer = Flux2Transformer2DModel.from_pretrained(
            spec.repo,
            subfolder="transformer",
            quantization_config=d_cfg,
            torch_dtype=dtype,
            local_files_only=True,
        )

    progress("Loading the NF4 Mistral text encoder (~15 GB)…")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        spec.quant_repo,
        subfolder="text_encoder",
        dtype=dtype,  # transformers 5 spells it `dtype`
        device_map="cpu",
        local_files_only=local_files_only,
    )

    # Assembled from parts rather than Flux2Pipeline.from_pretrained(repo, …):
    # that would insist on the repo's whole snapshot, including the NF4
    # transformer shards that are deliberately absent when the bf16 ones are
    # being quantized instead.
    progress("Loading the VAE, tokenizer and scheduler…")
    vae = AutoencoderKLFlux2.from_pretrained(
        spec.quant_repo, subfolder="vae", torch_dtype=dtype, local_files_only=local_files_only
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        spec.quant_repo, subfolder="scheduler", local_files_only=local_files_only
    )
    # The checkpoint's "tokenizer" is a PixtralProcessor (tokenizer plus an
    # image processor that needs torchvision). The studio only ever sends
    # text, and for text the processor's apply_chat_template is exactly the
    # tokenizer's — same chat_template.jinja, special tokens off — so the
    # plain tokenizer does the job without the torchvision dependency. It is
    # loaded by its concrete class: AutoTokenizer would look for a model
    # config at the repo root, which this checkpoint doesn't have.
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        spec.quant_repo, subfolder="tokenizer", local_files_only=local_files_only
    )
    pipe = Flux2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    return pipe, offline


def _cached_file(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, filename), str)


def _cached(repo: str, subfolder: str, weights: str) -> bool:
    """True when a component's config and every weight shard is in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    if not _cached_file(repo, f"{subfolder}/config.json"):
        return False
    index = try_to_load_from_cache(repo, f"{subfolder}/{weights}.index.json")
    if isinstance(index, str):
        try:
            with open(index, encoding="utf-8") as fh:
                shards = set(json.load(fh)["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return False
        return all(_cached_file(repo, f"{subfolder}/{shard}") for shard in shards)
    return _cached_file(repo, f"{subfolder}/{weights}")
