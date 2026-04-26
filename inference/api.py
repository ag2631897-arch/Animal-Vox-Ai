"""
AnimalVox AI — FastAPI REST Endpoint
Provides /api/v1/translate for programmatic access.
"""

import os
import json
import asyncio
import tempfile
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPECIES_GROUPS, INFERENCE_CONFIG
from inference.pipeline import AnimalVoxPipeline

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="AnimalVox AI API",
    description="Behavioral bioacoustics translation API",
    version="1.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = AnimalVoxPipeline(
    device="cpu",
    groq_api_key=os.getenv("GROQ_API_KEY", None)
)


@app.get("/")
async def root():
    return {
        "name": "AnimalVox AI",
        "version": "1.0.0",
        "status": "running",
        "supported_species": list(SPECIES_GROUPS.keys()),
        "docs": "/docs"
    }


@app.post("/api/v1/translate")
async def translate_audio(
    audio: UploadFile = File(None),
    audio_base64: str = Form(None),
    species: str = Form("birds"),
    audio_format: str = Form("wav")
):
    """
    Translate animal audio to human language.
    
    Accepts either file upload or base64-encoded audio.
    Returns behavioral analysis and natural language translation.
    """
    if species not in SPECIES_GROUPS:
        raise HTTPException(400, f"Unsupported species: {species}. Choose from: {list(SPECIES_GROUPS.keys())}")

    try:
        if audio:
            contents = await audio.read()
            fmt = audio.filename.rsplit(".", 1)[-1] if audio.filename else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(contents)
                result = await pipeline.translate(audio_path=tmp.name, species=species)
                os.unlink(tmp.name)
        elif audio_base64:
            audio_bytes = base64.b64decode(audio_base64)
            result = await pipeline.translate(audio_bytes=audio_bytes, audio_format=audio_format, species=species)
        else:
            raise HTTPException(400, "Provide audio file or audio_base64")

        # Remove non-serializable data
        result.pop("mel_spectrogram", None)
        return result

    except Exception as e:
        raise HTTPException(500, f"Translation error: {str(e)}")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
