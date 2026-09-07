"""Small provider-compatible HTTP wrapper for Resemble AI Chatterbox."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import numpy as np
import torch
import torchaudio


LOG = logging.getLogger("chatterbox-api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

VARIANT = os.getenv("CHATTERBOX_VARIANT", "turbo").strip().lower()
DEVICE = os.getenv("CHATTERBOX_DEVICE", "cuda").strip().lower()
LANGUAGE = os.getenv("CHATTERBOX_LANGUAGE", "en").strip().lower()
OUTPUT_RATE = 24_000
EXAGGERATION = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5"))
CFG_WEIGHT = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))
MAX_REFERENCE_BYTES = int(os.getenv("CHATTERBOX_MAX_REFERENCE_BYTES", "20000000"))

model = None
model_lock = asyncio.Lock()
startup_error: str | None = None


def _load_model():
    if VARIANT in {"turbo", "nano"}:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        if VARIANT == "nano":
            return ChatterboxTurboTTS.from_pretrained(device=DEVICE, nano=True)
        return ChatterboxTurboTTS.from_pretrained(device=DEVICE)
    if VARIANT == "original":
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=DEVICE)
    if VARIANT in {"multilingual", "multilingual-v3"}:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS.from_pretrained(
            device=DEVICE,
            t3_model="v3",
        )
    raise ValueError(
        "CHATTERBOX_VARIANT must be turbo, nano, original or multilingual-v3"
    )


def _generate(text: str, reference_path: str | None, seed: int) -> tuple[bytes, float]:
    assert model is not None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    kwargs = {}
    if reference_path:
        kwargs["audio_prompt_path"] = reference_path
    if VARIANT == "original":
        kwargs.update(exaggeration=EXAGGERATION, cfg_weight=CFG_WEIGHT)
    elif VARIANT in {"multilingual", "multilingual-v3"}:
        kwargs["language_id"] = LANGUAGE

    started = time.perf_counter()
    with torch.inference_mode():
        waveform = model.generate(text, **kwargs)
    generation_seconds = time.perf_counter() - started

    waveform = waveform.detach().float().cpu()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    source_rate = int(model.sr)
    if source_rate != OUTPUT_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            source_rate,
            OUTPUT_RATE,
        )
    pcm = (
        waveform.squeeze(0)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
        .numpy()
        .astype("<i2", copy=False)
    )
    return pcm.tobytes(), generation_seconds


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, startup_error
    try:
        LOG.info("Loading Chatterbox variant=%s device=%s", VARIANT, DEVICE)
        model = await asyncio.to_thread(_load_model)
        LOG.info("Chatterbox ready: variant=%s sample_rate=%s", VARIANT, model.sr)
    except Exception as exc:
        startup_error = str(exc)
        LOG.exception("Chatterbox startup failed")
    yield
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Chatterbox TTS adapter", lifespan=lifespan)


@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(503, startup_error or "Chatterbox is loading")
    return {
        "status": "ok",
        "provider": "chatterbox",
        "variant": VARIANT,
        "device": DEVICE,
        "source_sample_rate": int(model.sr),
        "output_sample_rate": OUTPUT_RATE,
    }


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    instruction: str = Form(""),
    cfg_scale: str = Form("1"),
    seed: int = Form(42),
    ref_text: str = Form(""),
    ref_audio: UploadFile | None = File(None),
):
    del instruction, cfg_scale, ref_text
    if model is None:
        raise HTTPException(503, startup_error or "Chatterbox is loading")
    text = text.strip()
    if not text:
        raise HTTPException(400, "text is required")

    reference_path: Path | None = None
    try:
        if ref_audio is not None:
            payload = await ref_audio.read(MAX_REFERENCE_BYTES + 1)
            if len(payload) > MAX_REFERENCE_BYTES:
                raise HTTPException(413, "reference audio is too large")
            suffix = Path(ref_audio.filename or "reference.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as output:
                output.write(payload)
                reference_path = Path(output.name)

        request_started = time.perf_counter()
        async with model_lock:
            pcm, generation_seconds = await asyncio.to_thread(
                _generate,
                text,
                str(reference_path) if reference_path else None,
                seed,
            )
        total_ms = round((time.perf_counter() - request_started) * 1000)
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers={
                "X-TTS-Provider": "chatterbox",
                "X-TTS-Model": VARIANT,
                "X-Audio-Sample-Rate": str(OUTPUT_RATE),
                "X-Generation-Ms": str(round(generation_seconds * 1000)),
                "X-Total-Ms": str(total_ms),
            },
        )
    finally:
        if reference_path is not None:
            reference_path.unlink(missing_ok=True)
