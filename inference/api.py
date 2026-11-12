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
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import init_db, get_connection
from database.auth import (
    register_user, login_user, authenticate_google_user,
    decode_access_token, get_user_by_id
)
from database.user_data import (
    verify_pet_ownership, save_user_pet, delete_user_pet,
    save_user_translation, get_user_state, sync_user_state,
    sync_pet_telemetry_to_db, restore_pet_telemetry_from_db,
)

import numpy as np

from config import SPECIES_GROUPS, INFERENCE_CONFIG, STREAMING_CONFIG, MEDICAL_DISCLAIMER
from inference.pipeline import AnimalVoxPipeline
from inference.cache import bioacoustic_cache
from inference.security import SecurityMiddleware, rate_limiter

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
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.floating, float)):
        return 0.0 if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


app = FastAPI(
    title="AnimalVox AI API",
    description="Behavioral bioacoustics translation & intelligence API (SRS v1.1)",
    version="1.1.0"
)

cors_env = os.getenv("CORS_ORIGINS", "")
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://animalvox.ai",
]
if cors_env:
    for origin in cors_env.split(","):
        origin_clean = origin.strip()
        if origin_clean and origin_clean not in allowed_origins:
            allowed_origins.append(origin_clean)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://animal-?vox-?ai(-[a-z0-9-]+)?\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityMiddleware)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "A transient bioacoustic processing error occurred. Please retry.",
        }
    )

pipeline = AnimalVoxPipeline(
    device="cpu",
    groq_api_key=os.getenv("GROQ_API_KEY", None)
)

@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("NeonDB PostgreSQL connected and initialized.")
    except Exception as e:
        logger.warning(f"NeonDB startup initialization warning: {e}")

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


def _get_optional_user_id(request: Request, authorization: Optional[str] = None) -> Optional[int]:
    """Extract authenticated user_id from Authorization Bearer header or access_token cookie."""
    token = None
    auth_header = authorization or request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif request.cookies.get("access_token"):
        token = request.cookies.get("access_token")
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        return None
    try:
        return int(payload["sub"])
    except (ValueError, TypeError):
        return None


def _require_user_id(request: Request, authorization: Optional[str] = None) -> int:
    user_id = _get_optional_user_id(request, authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user_id


def _enforce_pet_isolation(request: Request, pet_id: str, authorization: Optional[str] = None, auto_claim: bool = False) -> Optional[int]:
    """
    Ensure `pet_id` is only accessible by its owner user.
    Blocks other users (403) and unauthenticated requests to user-owned pets (401).
    """
    user_id = _get_optional_user_id(request, authorization)
    if user_id:
        if not verify_pet_ownership(user_id, pet_id, auto_claim=auto_claim):
            raise HTTPException(status_code=403, detail="Access denied: this pet profile belongs to another user.")
        return user_id

    # If unauthenticated, check if pet_id is registered to any user in NeonDB
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT user_id FROM pets WHERE id = %s;", (pet_id,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    raise HTTPException(status_code=401, detail="Authentication required to access this pet profile.")
                cur.execute("SELECT user_id FROM pet_baselines WHERE pet_id = %s;", (pet_id,))
                brow = cur.fetchone()
                if brow and brow[0] is not None:
                    raise HTTPException(status_code=401, detail="Authentication required to access this pet profile.")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception:
        pass
    return None


@app.post("/api/v1/translate")
async def translate_audio(
    request: Request,
    audio: UploadFile = File(None),
    audio_base64: str = Form(None),
    species: str = Form(None),           # None = auto-detect
    audio_format: str = Form("wav"),
    pet_id: str = Form(None),            # Optional pet profile ID
    enable_modules: Optional[Any] = Form(True),   # Enable intelligence modules
    authorization: Optional[str] = Header(None),
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
        return {
            "error": "unsupported_species",
            "species_supported": False,
            "detected_animal": species,
            "translation": "",
            "message": f"Species Not Supported Yet: {species} is not in our translation database.",
            "rejection_reason": f"Unsupported species: {species}. Supported: {list(SPECIES_GROUPS.keys())}",
            "processing_time_ms": 0,
        }

    if enable_modules is None:
        modules_enabled = True
    elif isinstance(enable_modules, bool):
        modules_enabled = enable_modules
    else:
        modules_enabled = str(enable_modules).strip().lower() not in ('false', '0', 'no', 'null', 'undefined')

    user_id = None
    if pet_id:
        user_id = await asyncio.to_thread(
            _enforce_pet_isolation, request, pet_id, authorization=authorization, auto_claim=True
        )
        await asyncio.to_thread(restore_pet_telemetry_from_db, user_id, pet_id)
    else:
        user_id = _get_optional_user_id(request, authorization)

    scoped_cache_pet = f"u{user_id}:{pet_id}" if (user_id and pet_id) else pet_id

    try:
        raw_bytes = None
        if audio:
            contents = await audio.read()
            raw_bytes = contents
            cache_key = bioacoustic_cache.compute_key(raw_bytes, species, scoped_cache_pet)
            cached_res = bioacoustic_cache.get(cache_key)
            if cached_res and not pet_id:
                return sanitize_for_json(cached_res)

            fmt = audio.filename.rsplit(".", 1)[-1] if audio.filename else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                result = await pipeline.translate(
                    audio_path=tmp_path, species=species,
                    pet_id=pet_id, enable_modules=modules_enabled,
                    filename=audio.filename
                )
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        elif audio_base64:
            audio_bytes = base64.b64decode(audio_base64)
            raw_bytes = audio_bytes
            cache_key = bioacoustic_cache.compute_key(raw_bytes, species, scoped_cache_pet)
            cached_res = bioacoustic_cache.get(cache_key)
            if cached_res and not pet_id:
                return sanitize_for_json(cached_res)

            result = await pipeline.translate(
                audio_bytes=audio_bytes, audio_format=audio_format,
                species=species, pet_id=pet_id, enable_modules=enable_modules
            )
        else:
            raise HTTPException(400, "Provide audio file or audio_base64")

        if pet_id and not result.get("error"):
            asyncio.get_running_loop().run_in_executor(
                None, sync_pet_telemetry_to_db, user_id, pet_id
            )

        # Remove non-serializable data and sanitize all numpy primitives
        result.pop("mel_spectrogram", None)
        sanitized = sanitize_for_json(result)
        if raw_bytes and not result.get("error"):
            bioacoustic_cache.set(cache_key, sanitized)
        return sanitized

    except HTTPException:
        raise
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
    """
    species = species or None
    if species and species not in SPECIES_GROUPS:
        async def unsupported_generator():
            yield f"data: {json.dumps({'type': 'unsupported_species', 'species_supported': False, 'message': f'Unsupported species: {species}'})}\n\n"
        return StreamingResponse(unsupported_generator(), media_type="text/event-stream")

    async def event_generator():
        try:
            filename = None
            if audio:
                filename = audio.filename
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
                audio_path=audio_path, species=species, pet_id=pet_id, filename=filename
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
def get_health_weekly(
    pet_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Get weekly health digest for a pet (scoped to the pet's owner)."""
    user_id = _enforce_pet_isolation(request, pet_id, authorization=authorization, auto_claim=True)
    restore_pet_telemetry_from_db(user_id, pet_id)

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
def get_emotion_weekly(
    pet_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Get weekly emotional digest for a pet (scoped to the pet's owner)."""
    user_id = _enforce_pet_isolation(request, pet_id, authorization=authorization, auto_claim=True)
    restore_pet_telemetry_from_db(user_id, pet_id)

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
def get_wellbeing(
    pet_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Get current wellbeing score and history for a pet (scoped to the pet's owner)."""
    user_id = _enforce_pet_isolation(request, pet_id, authorization=authorization, auto_claim=True)
    restore_pet_telemetry_from_db(user_id, pet_id)

    pipeline.load_modules()
    diary_entries = pipeline.emotional_tracker.diary.get_entries(pet_id, days=7)
    baseline = pipeline.health_monitor.baseline_mgr.get_baseline(pet_id)
    if not diary_entries and not baseline:
        raise HTTPException(404, f"No wellbeing data found for pet: {pet_id}")

    prosodic_means = (baseline or {}).get("means", {})
    report = pipeline.wellbeing.assess(
        pet_id=pet_id,
        prosodic_features=prosodic_means,
        diary_entries=diary_entries,
        save_to_history=False,
    )
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


# ── Per-User State & Profile Persistence Endpoints (NeonDB) ──
@app.get("/api/v1/user/state")
def api_get_user_state(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Fetch authenticated user's pets, translation history, and settings from NeonDB."""
    user_id = _require_user_id(request, authorization)
    state = get_user_state(user_id)
    return {"success": True, **sanitize_for_json(state)}


@app.post("/api/v1/user/sync")
def api_sync_user_state(
    payload: Dict[str, Any],
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Synchronize authenticated user's pets, translation history, and settings with NeonDB."""
    user_id = _require_user_id(request, authorization)
    state = sync_user_state(user_id, payload)
    return {"success": True, **sanitize_for_json(state)}


@app.post("/api/v1/pets")
def api_save_pet(
    pet: Dict[str, Any],
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Save or update a pet profile owned by the authenticated user."""
    user_id = _require_user_id(request, authorization)
    saved = save_user_pet(user_id, pet)
    if not saved:
        raise HTTPException(status_code=403, detail="Could not save pet profile.")
    return {"success": True, "pet": sanitize_for_json(saved)}


@app.delete("/api/v1/pets/{pet_id}")
def api_delete_pet(
    pet_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Delete a pet profile owned by the authenticated user."""
    user_id = _require_user_id(request, authorization)
    deleted = delete_user_pet(user_id, pet_id)
    return {"success": True, "deleted": deleted}


@app.post("/api/v1/translations")
def api_save_translation(
    entry: Dict[str, Any],
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Save a translation history entry for the authenticated user."""
    user_id = _require_user_id(request, authorization)
    saved = save_user_translation(user_id, entry)
    return {"success": True, "entry": sanitize_for_json(saved)}


# ── Authentication Endpoints (NeonDB PostgreSQL + Google OAuth) ──
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    credential: Optional[str] = None
    user_info: Optional[Dict[str, Any]] = None
    access_token: Optional[str] = None
    code: Optional[str] = None
    redirect_uri: Optional[str] = None

def _read_google_credentials() -> Dict[str, str]:
    """Helper to dynamically read Google OAuth credentials from .env files or environment."""
    from dotenv import dotenv_values
    candidates = {}
    
    # 1. Root .env
    root_env_path = PROJECT_ROOT / ".env"
    if root_env_path.exists():
        candidates.update(dotenv_values(root_env_path))
    
    # 2. Frontend .env
    fe_env_path = PROJECT_ROOT / "frontend" / ".env"
    if fe_env_path.exists():
        candidates.update(dotenv_values(fe_env_path))

    # Helper to clean strings
    def clean(val: Optional[str]) -> str:
        if not val:
            return ""
        return val.strip().strip('"').strip("'").strip()

    client_id = ""
    client_secret = ""

    # Check candidates case-insensitively
    for k, v in candidates.items():
        if not k or not v:
            continue
        k_upper = k.strip().upper()
        if k_upper in ("GOOGLE_CLIENT_ID", "VITE_GOOGLE_CLIENT_ID", "CLIENT_ID", "GOOGLE_ID"):
            client_id = clean(v)
        elif k_upper in ("GOOGLE_CLIENT_SECRET", "VITE_GOOGLE_CLIENT_SECRET", "CLIENT_SECRET", "GOOGLE_SECRET"):
            client_secret = clean(v)

    # Check os.environ if not found
    if not client_id:
        for k in ("GOOGLE_CLIENT_ID", "VITE_GOOGLE_CLIENT_ID", "CLIENT_ID"):
            val = os.getenv(k)
            if val:
                client_id = clean(val)
                break

    if not client_secret:
        for k in ("GOOGLE_CLIENT_SECRET", "CLIENT_SECRET"):
            val = os.getenv(k)
            if val:
                client_secret = clean(val)
                break

    return {"client_id": client_id, "client_secret": client_secret}

@app.get("/api/v1/auth/config")
async def api_auth_config():
    """Return public authentication configuration like Google Client ID (never secrets)."""
    creds = _read_google_credentials()
    client_id = creds["client_id"]
    return {
        "google_client_id": client_id,
        "has_google_client_id": bool(client_id),
    }

def _set_auth_cookie(response: Response, token: str):
    """Set secure HTTP-only session cookie for authentication."""
    secure_cookie = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1")
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=secure_cookie,
        max_age=7 * 24 * 3600,
        path="/",
    )

@app.post("/api/v1/auth/signup")
def api_signup(req: SignupRequest, response: Response):
    """Register a new user account with NeonDB PostgreSQL and set auth cookie."""
    res = register_user(req.name, req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Registration failed"))
    if res.get("token"):
        _set_auth_cookie(response, res["token"])
    return res

@app.post("/api/v1/auth/login")
def api_login(req: LoginRequest, response: Response):
    """Log in with email and password verified against NeonDB PostgreSQL."""
    res = login_user(req.email, req.password)
    if not res.get("success"):
        raise HTTPException(status_code=401, detail=res.get("error", "Invalid credentials"))
    if res.get("token"):
        _set_auth_cookie(response, res["token"])
    return res

@app.post("/api/v1/auth/google")
def api_google_auth(req: GoogleAuthRequest, response: Response):
    """Log in or sign up with Google OAuth, returning is_new_user and setting auth cookie."""
    res = authenticate_google_user(
        credential=req.credential,
        user_info=req.user_info,
        access_token=req.access_token,
        code=req.code,
        redirect_uri=req.redirect_uri,
    )
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Google authentication failed"))
    if res.get("token"):
        _set_auth_cookie(response, res["token"])
    return res

@app.post("/api/v1/auth/logout")
def api_logout(response: Response):
    """Log out and revoke authentication cookie."""
    response.delete_cookie(key="access_token", path="/")
    return {"success": True, "message": "Logged out successfully"}

@app.get("/api/v1/auth/me")
def api_get_me(request: Request, authorization: Optional[str] = Header(None)):
    """Retrieve currently authenticated user profile from NeonDB via header or HTTP cookie."""
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    elif request.cookies.get("access_token"):
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header or cookie")
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"success": True, "user": sanitize_for_json(user)}

@app.get("/api/v1/cache/stats")
async def api_cache_stats():
    """Retrieve bioacoustic cache performance and rate limiting telemetry."""
    return {
        "cache": bioacoustic_cache.stats(),
        "rate_limiter": {
            "active_ip_trackers": len(rate_limiter._history),
            "limits_per_minute": rate_limiter.limits,
        }
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
