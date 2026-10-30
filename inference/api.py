"""
AnimalVox AI — FastAPI REST & Streaming Endpoint (SRS v1.1)
=============================================================
Provides:
  /api/v1/translate          — Full translation (POST, JSON response)
  /api/v1/translate/stream   — Streaming translation (POST, SSE response)
  /api/v1/species            — List supported species (GET)
  /api/v1/health/weekly      — Weekly health digest (GET)
  /api/v1/emotion/weekly     — Weekly emotional digest (GET)
  /api/v1/wellbeing          — Current wellbeing score (GET)
  /api/v1/reset              — Reset session (POST)
"""

import os
import json
import asyncio
import tempfile
import base64
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import SPECIES_GROUPS, INFERENCE_CONFIG, STREAMING_CONFIG, MEDICAL_DISCLAIMER
from inference.pipeline import AnimalVoxPipeline

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def sanitize_for_json(obj):
    """Recursively convert numpy types, NaN, and tensors to standard JSON-compatible Python primitives."""
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = FastAPI(
    title="AnimalVox AI API",
    description="Behavioral bioacoustics translation & intelligence API (SRS v1.1)",
    version="1.1.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = AnimalVoxPipeline(
    device="cpu",
    groq_api_key=os.getenv("GROQ_API_KEY", None)
)

@app.on_event("startup")
async def startup_event():
    logger.info("Pre-loading models and intelligence modules...")
    pipeline.load_models()
    pipeline.load_modules()
    logger.info("All AnimalVox AI models and modules ready for serving.")


@app.get("/health")
async def root():
    return {
        "name": "AnimalVox AI",
        "version": "1.1.0",
        "status": "healthy",
        "supported_species": list(SPECIES_GROUPS.keys()),
        "modules": [
            "anti_spoofing", "animal_detection", "health_monitoring",
            "breed_identification", "emotional_tracking", "wellbeing_signals",
            "streaming_translation"
        ],
        "docs": "/docs"
    }


@app.post("/api/v1/translate")
async def translate_audio(
    audio: UploadFile = File(None),
    audio_base64: str = Form(None),
    species: str = Form(None),           # None = auto-detect
    audio_format: str = Form("wav"),
    pet_id: str = Form(None),            # Optional pet profile ID
    enable_modules: Optional[bool] = Form(True),   # Enable intelligence modules
):
    """
    Translate animal audio to human language.

    Accepts either file upload or base64-encoded audio.
    Returns behavioral analysis, natural language translation,
    and enriched intelligence module outputs.

    Set species=None for automatic animal detection (Module 05).
    Set pet_id for baseline tracking across sessions.
    """
    species = species or None
    if species and species not in SPECIES_GROUPS:
        raise HTTPException(400, f"Unsupported species: {species}. Choose from: {list(SPECIES_GROUPS.keys())}")

    modules_enabled = True if enable_modules is None else bool(enable_modules)

    try:
        if audio:
            contents = await audio.read()
            fmt = audio.filename.rsplit(".", 1)[-1] if audio.filename else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                result = await pipeline.translate(
                    audio_path=tmp_path, species=species,
                    pet_id=pet_id, enable_modules=modules_enabled
                )
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        elif audio_base64:
            audio_bytes = base64.b64decode(audio_base64)
            result = await pipeline.translate(
                audio_bytes=audio_bytes, audio_format=audio_format,
                species=species, pet_id=pet_id, enable_modules=enable_modules
            )
        else:
            raise HTTPException(400, "Provide audio file or audio_base64")

        # Remove non-serializable data and sanitize all numpy primitives
        result.pop("mel_spectrogram", None)
        return sanitize_for_json(result)

    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(500, f"Translation error: {str(e)}")


@app.post("/api/v1/translate/stream")
async def translate_audio_stream(
    audio: UploadFile = File(None),
    audio_base64: str = Form(None),
    species: str = Form(None),
    audio_format: str = Form("wav"),
    pet_id: str = Form(None),
):
    """
    Streaming translation via Server-Sent Events (SSE).

    Returns a text/event-stream with word-by-word translation tokens
    for real-time karaoke-style display with synchronized TTS.

    Event types:
    - metadata: species, emoji, soundscape
    - translation_start: translation begins
    - token: individual word/token
    - phrase: buffered phrase for TTS utterance
    - translation_end: translation complete
    - health: health monitoring results
    - emotion: emotional state results
    - wellbeing: wellbeing composite score
    - human_detected: human imitation rejected
    - no_animal: no animal detected
    - done: processing complete with timing
    - error: error occurred
    """
    species = species or None
    if species and species not in SPECIES_GROUPS:
        raise HTTPException(400, f"Unsupported species: {species}")

    async def event_generator():
        try:
            if audio:
                contents = await audio.read()
                fmt = audio.filename.rsplit(".", 1)[-1] if audio.filename else "wav"
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                    tmp.write(contents)
                    audio_path = tmp.name
            elif audio_base64:
                audio_bytes = base64.b64decode(audio_base64)
                with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    audio_path = tmp.name
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio provided'})}\n\n"
                return

            async for event in pipeline.translate_stream(
                audio_path=audio_path, species=species, pet_id=pet_id
            ):
                yield f"data: {json.dumps(event)}\n\n"

            # Cleanup temp file
            try:
                os.unlink(audio_path)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/v1/reset")
async def reset():
    """Reset call history for new session."""
    pipeline.reset_session()
    return {"status": "session_reset"}


@app.get("/api/v1/species")
async def list_species():
    """List supported species with details."""
    return {k: {"display_name": v["display_name"], "emoji": v["emoji"]}
            for k, v in SPECIES_GROUPS.items()}


@app.get("/api/v1/health/weekly/{pet_id}")
async def get_health_weekly(pet_id: str):
    """Get weekly health digest for a pet."""
    pipeline.load_modules()
    baseline = pipeline.health_monitor.baseline_mgr.get_baseline(pet_id)
    if not baseline:
        raise HTTPException(404, f"No baseline found for pet: {pet_id}")

    return {
        "pet_id": pet_id,
        "baseline": {
            "established": pipeline.health_monitor.baseline_mgr.is_established(pet_id),
            "clip_count": baseline.get("clip_count", 0),
            "created_at": baseline.get("created_at"),
            "updated_at": baseline.get("updated_at"),
            "means": baseline.get("means", {}),
        },
        "disclaimer": MEDICAL_DISCLAIMER,
    }


@app.get("/api/v1/emotion/weekly/{pet_id}")
async def get_emotion_weekly(pet_id: str):
    """Get weekly emotional digest for a pet."""
    pipeline.load_modules()
    digest = pipeline.emotional_tracker.get_weekly_digest(pet_id)
    if not digest:
        raise HTTPException(404, f"No emotional data found for pet: {pet_id}")

    return {
        "pet_id": digest.pet_id,
        "week_start": digest.week_start,
        "week_end": digest.week_end,
        "dominant_emotion": digest.dominant_emotion,
        "emotion_distribution": digest.emotion_distribution,
        "stress_episodes": digest.stress_episodes,
        "contentment_episodes": digest.contentment_episodes,
        "mood_stability": digest.mood_stability,
        "anomalies": digest.anomalies,
        "total_entries": digest.total_entries,
    }


@app.get("/api/v1/wellbeing/{pet_id}")
async def get_wellbeing(pet_id: str):
    """Get current wellbeing score and history for a pet."""
    pipeline.load_modules()
    diary_entries = pipeline.emotional_tracker.diary.get_entries(pet_id, days=7)
    if not diary_entries:
        raise HTTPException(404, f"No wellbeing data found for pet: {pet_id}")

    report = pipeline.wellbeing.assess(pet_id=pet_id, diary_entries=diary_entries)
    trend, delta = pipeline.wellbeing.history.get_trend(pet_id)
    return {
        "pet_id": pet_id,
        "composite_score": report.composite_score,
        "composite_label": report.composite_label,
        "trend": trend,
        "trend_delta": delta,
        "dimensions": {
            key: {
                "name": dimension.name,
                "score": dimension.score,
                "direction": dimension.direction,
                "trend": dimension.trend,
            }
            for key, dimension in report.dimensions.items()
        },
        "recommendations": report.recommendations,
    }


if FRONTEND_DIST.is_dir():
    @app.get("/{frontend_path:path}", include_in_schema=False)
    async def serve_frontend(frontend_path: str):
        """Serve the compiled React interface, including client-side routes."""
        requested = (FRONTEND_DIST / frontend_path).resolve()
        try:
            requested.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            raise HTTPException(404, "Not found")
        if requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
