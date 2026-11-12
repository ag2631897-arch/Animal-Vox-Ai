"""
AnimalVox AI — Per-User Data Persistence & Isolation Service (NeonDB PostgreSQL)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from psycopg2.extras import RealDictCursor, Json
from database.connection import get_connection
from config import PET_PROFILES_DIR, DIARY_DIR, EMOTIONAL_TRACKING_CONFIG

logger = logging.getLogger(__name__)

# In-memory caches to avoid redundant remote NeonDB queries on every API call
_verified_ownership_cache: Set[Tuple[int, str]] = set()
_restored_telemetry_pets: Set[str] = set()


def verify_pet_ownership(user_id: int, pet_id: str, auto_claim: bool = False) -> bool:
    """
    Verify that `pet_id` belongs to `user_id`.
    Returns False if `pet_id` belongs to another user.
    """
    if not pet_id or not user_id:
        return False

    if (user_id, pet_id) in _verified_ownership_cache:
        return True

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT user_id FROM pets WHERE id = %s;", (pet_id,))
            row = cur.fetchone()
            if row:
                if row["user_id"] == user_id:
                    _verified_ownership_cache.add((user_id, pet_id))
                    return True
                if row["user_id"] is None and auto_claim:
                    cur.execute("UPDATE pets SET user_id = %s WHERE id = %s;", (user_id, pet_id))
                    conn.commit()
                    _verified_ownership_cache.add((user_id, pet_id))
                    return True
                return False

            # Check pet_baselines ownership if pet row not yet synced
            cur.execute("SELECT user_id FROM pet_baselines WHERE pet_id = %s;", (pet_id,))
            b_row = cur.fetchone()
            if b_row and b_row["user_id"] is not None:
                if b_row["user_id"] == user_id:
                    _verified_ownership_cache.add((user_id, pet_id))
                    return True
                return False

            if auto_claim:
                _verified_ownership_cache.add((user_id, pet_id))
                return True
            return False
    except Exception as e:
        logger.warning(f"verify_pet_ownership error for pet {pet_id}: {e}")
        return False
    finally:
        conn.close()


def _upsert_pet_on_cursor(cur, user_id: int, pet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Upsert a pet row using an existing cursor."""
    pet_id = str(pet.get("id") or "").strip()
    name = str(pet.get("name") or "").strip()
    if not pet_id or not name:
        return None

    species = str(pet.get("species") or "Dog").strip()
    breed = str(pet.get("breed") or "").strip()
    age = str(pet.get("age") or "").strip()
    gender = str(pet.get("gender") or "Unknown").strip()
    notes = str(pet.get("notes") or "").strip()
    photo = str(pet.get("photo") or "")
    emoji = str(pet.get("emoji") or "")

    cur.execute("""
        INSERT INTO pets (id, user_id, name, species, breed, age, gender, notes, photo, emoji, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            species = EXCLUDED.species,
            breed = EXCLUDED.breed,
            age = EXCLUDED.age,
            gender = EXCLUDED.gender,
            notes = EXCLUDED.notes,
            photo = EXCLUDED.photo,
            emoji = EXCLUDED.emoji,
            updated_at = NOW()
        WHERE pets.user_id = %s OR pets.user_id IS NULL
        RETURNING id, user_id, name, species, breed, age, gender, notes, photo, emoji, created_at;
    """, (pet_id, user_id, name, species, breed, age, gender, notes, photo, emoji, user_id))
    row = cur.fetchone()
    if row:
        _verified_ownership_cache.add((user_id, pet_id))
        return _format_pet_row(row)
    return None


def save_user_pet(user_id: int, pet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Upsert a pet profile for a specific user, enforcing user isolation."""
    pet_id = str(pet.get("id") or "").strip()
    name = str(pet.get("name") or "").strip()
    if not pet_id or not name:
        return None

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            formatted = _upsert_pet_on_cursor(cur, user_id, pet)
            conn.commit()
            if formatted:
                sync_pet_telemetry_to_db(user_id, pet_id, conn=conn)
                return formatted
            return None
    finally:
        conn.close()


def delete_user_pet(user_id: int, pet_id: str) -> bool:
    """Delete a pet profile only if owned by user_id."""
    _verified_ownership_cache.discard((user_id, pet_id))
    _restored_telemetry_pets.discard(pet_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM pets WHERE id = %s AND user_id = %s;", (pet_id, user_id))
            deleted = cur.rowcount > 0
            if deleted:
                cur.execute("DELETE FROM pet_baselines WHERE pet_id = %s AND user_id = %s;", (pet_id, user_id))
                cur.execute("DELETE FROM emotion_entries WHERE pet_id = %s AND user_id = %s;", (pet_id, user_id))
                cur.execute("DELETE FROM wellbeing_reports WHERE pet_id = %s AND user_id = %s;", (pet_id, user_id))
            conn.commit()
            return deleted
    finally:
        conn.close()


def _insert_translation_on_cursor(cur, user_id: int, entry: Dict[str, Any], skip_exists_check: bool = False) -> Optional[Dict[str, Any]]:
    """Insert a translation entry using an existing cursor."""
    entry_id = str(entry.get("id") or "").strip()
    translation_text = str(entry.get("translation") or "").strip()
    if not entry_id or not translation_text:
        return None

    if not skip_exists_check:
        cur.execute(
            "SELECT id FROM translations WHERE user_id = %s AND entry_id = %s LIMIT 1;",
            (user_id, entry_id),
        )
        if cur.fetchone():
            return entry

    pet_id = str(entry.get("petId") or "").strip() or None
    pet_name = str(entry.get("petName") or "Unknown animal").strip()
    behavior = str(entry.get("behavior") or "").strip()
    confidence = entry.get("confidence")
    confidence_val = float(confidence) if isinstance(confidence, (int, float)) else None
    created_iso = str(entry.get("createdAt") or "")
    result_obj = entry.get("result") if isinstance(entry.get("result"), dict) else {}
    species = str(result_obj.get("species") or entry.get("species") or pet_name or "dogs").strip()
    intensity = result_obj.get("intensity")
    intensity_val = float(intensity) if isinstance(intensity, (int, float)) else None
    urgency = result_obj.get("urgency")
    urgency_val = float(urgency) if isinstance(urgency, (int, float)) else None

    cur.execute("""
        INSERT INTO translations (
            entry_id, user_id, pet_id, pet_name, species, behavior,
            translation, confidence, intensity, urgency, result_json, created_iso
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        entry_id, user_id, pet_id, pet_name, species, behavior,
        translation_text, confidence_val, intensity_val, urgency_val,
        Json(result_obj), created_iso
    ))
    return entry


def save_user_translation(user_id: int, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Save a translation history entry for a user."""
    entry_id = str(entry.get("id") or "").strip()
    translation_text = str(entry.get("translation") or "").strip()
    if not entry_id or not translation_text:
        return None

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            res = _insert_translation_on_cursor(cur, user_id, entry, skip_exists_check=False)
            conn.commit()
            return res
    except Exception as e:
        logger.warning(f"save_user_translation error: {e}")
        return None
    finally:
        conn.close()


def _fetch_user_state_on_cursor(cur, conn, user_id: int) -> Dict[str, Any]:
    """Fetch complete persisted state for a user using an active cursor."""
    cur.execute("""
        SELECT id, user_id, name, species, breed, age, gender, notes, photo, emoji, created_at
        FROM pets
        WHERE user_id = %s
        ORDER BY created_at ASC;
    """, (user_id,))
    pets = [_format_pet_row(r) for r in cur.fetchall()]
    for p in pets:
        _verified_ownership_cache.add((user_id, p["id"]))

    cur.execute("""
        SELECT entry_id, pet_id, pet_name, species, behavior, translation,
               confidence, result_json, created_iso, created_at
        FROM translations
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 100;
    """, (user_id,))
    translations = [_format_translation_row(r) for r in cur.fetchall()]

    cur.execute("""
        SELECT active_pet_id, theme, preset_name, voice, speed,
               auto_species_detection, has_completed_onboarding,
               account_profile, active_sessions
        FROM user_settings
        WHERE user_id = %s;
    """, (user_id,))
    settings_row = cur.fetchone()

    for p in pets:
        restore_pet_telemetry_from_db(user_id, p["id"], conn=conn)

    settings = {}
    if settings_row:
        settings = {
            "activePetId": settings_row["active_pet_id"] or "",
            "theme": settings_row["theme"],
            "presetName": settings_row["preset_name"] or "Midnight Forest",
            "voice": settings_row["voice"] or "Nova",
            "speed": float(settings_row["speed"]) if settings_row["speed"] is not None else 1.0,
            "autoSpeciesDetection": bool(settings_row["auto_species_detection"]) if settings_row["auto_species_detection"] is not None else True,
            "hasCompletedOnboarding": bool(settings_row["has_completed_onboarding"]),
            "accountProfile": settings_row["account_profile"] or {},
            "activeSessions": settings_row["active_sessions"] or {},
        }

    return {
        "pets": pets,
        "translationHistory": translations,
        "settings": settings,
    }


def get_user_state(user_id: int) -> Dict[str, Any]:
    """Fetch complete persisted state for a user from NeonDB."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            return _fetch_user_state_on_cursor(cur, conn, user_id)
    finally:
        conn.close()


def sync_user_state(user_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize client user state with NeonDB for `user_id` in a SINGLE connection
    and return the merged authoritative state.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Upsert pets in the same cursor
            incoming_pets = payload.get("pets")
            if isinstance(incoming_pets, list):
                for pet in incoming_pets:
                    if isinstance(pet, dict) and pet.get("id") and pet.get("name"):
                        _upsert_pet_on_cursor(cur, user_id, pet)

            # 2. Batch-check existing translation entry_ids in ONE query instead of 100 connections
            incoming_history = payload.get("translationHistory")
            if isinstance(incoming_history, list) and incoming_history:
                cur.execute(
                    "SELECT entry_id FROM translations WHERE user_id = %s AND entry_id IS NOT NULL;",
                    (user_id,),
                )
                existing_ids = {r["entry_id"] for r in cur.fetchall() if r.get("entry_id")}

                # Insert only genuinely new entries (oldest first)
                for entry in reversed(incoming_history[:100]):
                    if not isinstance(entry, dict):
                        continue
                    eid = str(entry.get("id") or "").strip()
                    if eid and eid not in existing_ids and entry.get("translation"):
                        _insert_translation_on_cursor(cur, user_id, entry, skip_exists_check=True)
                        existing_ids.add(eid)

            # 3. Upsert user_settings in the same cursor
            cur.execute("SELECT * FROM user_settings WHERE user_id = %s;", (user_id,))
            existing = cur.fetchone() or {}

            active_pet_id = payload.get("activePetId") if "activePetId" in payload else (existing.get("active_pet_id") or "")
            theme = payload.get("theme") if "theme" in payload else existing.get("theme")
            preset_name = payload.get("presetName") if "presetName" in payload else (existing.get("preset_name") or "Midnight Forest")
            voice = payload.get("voice") if "voice" in payload else (existing.get("voice") or "Nova")
            speed = payload.get("speed") if "speed" in payload else (existing.get("speed") if existing.get("speed") is not None else 1.0)
            auto_detect = payload.get("autoSpeciesDetection") if "autoSpeciesDetection" in payload else (
                existing.get("auto_species_detection") if existing.get("auto_species_detection") is not None else True
            )
            onboarded = bool(payload.get("hasCompletedOnboarding") or existing.get("has_completed_onboarding"))
            account_profile = payload.get("accountProfile") if "accountProfile" in payload else (existing.get("account_profile") or {})

            existing_sessions = existing.get("active_sessions") or {}
            incoming_sessions = payload.get("activeSessions")
            if isinstance(incoming_sessions, dict):
                merged_sessions = {**existing_sessions, **incoming_sessions}
            else:
                merged_sessions = existing_sessions

            cur.execute("""
                INSERT INTO user_settings (
                    user_id, active_pet_id, theme, preset_name, voice, speed,
                    auto_species_detection, has_completed_onboarding,
                    account_profile, active_sessions, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_id) DO UPDATE SET
                    active_pet_id = EXCLUDED.active_pet_id,
                    theme = COALESCE(EXCLUDED.theme, user_settings.theme),
                    preset_name = EXCLUDED.preset_name,
                    voice = EXCLUDED.voice,
                    speed = EXCLUDED.speed,
                    auto_species_detection = EXCLUDED.auto_species_detection,
                    has_completed_onboarding = EXCLUDED.has_completed_onboarding,
                    account_profile = COALESCE(EXCLUDED.account_profile, user_settings.account_profile),
                    active_sessions = COALESCE(EXCLUDED.active_sessions, user_settings.active_sessions),
                    updated_at = NOW();
            """, (
                user_id,
                str(active_pet_id or ""),
                Json(theme) if theme else None,
                str(preset_name or "Midnight Forest"),
                str(voice or "Nova"),
                float(speed if speed is not None else 1.0),
                bool(auto_detect),
                bool(onboarded),
                Json(account_profile) if account_profile is not None else Json({}),
                Json(merged_sessions),
            ))
            conn.commit()

            # 4. Fetch state on the same connection
            return _fetch_user_state_on_cursor(cur, conn, user_id)
    except Exception as e:
        logger.warning(f"sync_user_state error: {e}")
        return get_user_state(user_id)
    finally:
        conn.close()


def sync_pet_telemetry_to_db(user_id: Optional[int], pet_id: str, conn=None):
    """
    Persist local baseline JSON, SQLite emotion_entries, and wellbeing history JSON for `pet_id`
    into NeonDB under `user_id`.
    """
    if not pet_id:
        return

    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if not user_id:
                cur.execute("SELECT user_id FROM pets WHERE id = %s;", (pet_id,))
                r = cur.fetchone()
                if r:
                    user_id = r["user_id"]

            # 1. Baseline JSON -> pet_baselines
            baseline_path = PET_PROFILES_DIR / f"{pet_id}_baseline.json"
            if baseline_path.exists():
                try:
                    with open(baseline_path, "r") as f:
                        b_data = json.load(f)
                    cur.execute("""
                        INSERT INTO pet_baselines (pet_id, user_id, baseline_json, updated_at)
                        VALUES (%s, %s, %s, NOW())
                        ON CONFLICT (pet_id) DO UPDATE SET
                            user_id = COALESCE(pet_baselines.user_id, EXCLUDED.user_id),
                            baseline_json = EXCLUDED.baseline_json,
                            updated_at = NOW();
                    """, (pet_id, user_id, Json(b_data)))
                except Exception as e:
                    logger.debug(f"Baseline DB sync warning for {pet_id}: {e}")

            # 2. SQLite emotion_entries -> Postgres emotion_entries
            db_path = DIARY_DIR / EMOTIONAL_TRACKING_CONFIG.diary_db_name
            if db_path.exists():
                try:
                    s_conn = sqlite3.connect(str(db_path))
                    rows = s_conn.execute(
                        """SELECT timestamp, primary_emotion, primary_confidence,
                                  all_emotions, stress_level, prosodic_snapshot, species
                           FROM emotion_entries WHERE pet_id = ?""",
                        (pet_id,),
                    ).fetchall()
                    s_conn.close()

                    if rows:
                        cur.execute("SELECT timestamp FROM emotion_entries WHERE pet_id = %s;", (pet_id,))
                        existing_ts = {
                            r["timestamp"].isoformat()[:19] if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])[:19]
                            for r in cur.fetchall()
                        }

                        for row in rows:
                            ts_str = str(row[0])
                            if ts_str[:19] in existing_ts:
                                continue
                            all_em = json.loads(row[3]) if row[3] else []
                            pros = json.loads(row[5]) if row[5] else {}
                            stress_val = float(row[4]) if isinstance(row[4], (int, float)) else 0.0
                            cur.execute("""
                                INSERT INTO emotion_entries (
                                    user_id, pet_id, timestamp, primary_emotion,
                                    primary_confidence, all_emotions, stress_level,
                                    prosodic_snapshot, species
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """, (
                                user_id, pet_id, ts_str, str(row[1]),
                                float(row[2]), Json(all_em), stress_val,
                                Json(pros), str(row[6] or ""),
                            ))
                        if user_id:
                            cur.execute(
                                "UPDATE emotion_entries SET user_id = %s WHERE pet_id = %s AND user_id IS NULL;",
                                (user_id, pet_id),
                            )
                except Exception as e:
                    logger.debug(f"Emotion DB sync warning for {pet_id}: {e}")

            conn.commit()
    except Exception as e:
        logger.warning(f"sync_pet_telemetry_to_db error for {pet_id}: {e}")
    finally:
        if owns_conn:
            conn.close()


def restore_pet_telemetry_from_db(user_id: Optional[int], pet_id: str, conn=None):
    """
    Ensure local baseline JSON and SQLite emotion_entries are populated from NeonDB if missing.
    Skips network calls when already restored in this session or when local files are already present.
    """
    if not pet_id:
        return

    baseline_path = PET_PROFILES_DIR / f"{pet_id}_baseline.json"
    db_path = DIARY_DIR / EMOTIONAL_TRACKING_CONFIG.diary_db_name

    # Fast path: already restored during this process lifetime
    if pet_id in _restored_telemetry_pets:
        return

    # Check local disk BEFORE opening any remote DB connection
    has_local_baseline = baseline_path.exists()
    has_local_emotions = False
    try:
        s_conn = sqlite3.connect(str(db_path))
        s_conn.execute("""
            CREATE TABLE IF NOT EXISTS emotion_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pet_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                primary_emotion TEXT NOT NULL,
                primary_confidence REAL NOT NULL,
                all_emotions TEXT NOT NULL,
                stress_level REAL NOT NULL,
                prosodic_snapshot TEXT,
                species TEXT
            )
        """)
        local_count = s_conn.execute(
            "SELECT COUNT(*) FROM emotion_entries WHERE pet_id = ?", (pet_id,)
        ).fetchone()[0]
        has_local_emotions = local_count > 0
        s_conn.close()
    except Exception:
        pass

    if has_local_baseline and has_local_emotions:
        _restored_telemetry_pets.add(pet_id)
        return

    owns_conn = conn is None
    if owns_conn:
        conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Restore baseline JSON if missing locally
            if not has_local_baseline:
                cur.execute("SELECT baseline_json FROM pet_baselines WHERE pet_id = %s;", (pet_id,))
                b_row = cur.fetchone()
                if b_row and b_row["baseline_json"]:
                    with open(baseline_path, "w") as f:
                        json.dump(b_row["baseline_json"], f, indent=2)

            # 2. Restore SQLite emotion_entries if missing locally
            if not has_local_emotions:
                cur.execute("""
                    SELECT timestamp, primary_emotion, primary_confidence,
                           all_emotions, stress_level, prosodic_snapshot, species
                    FROM emotion_entries
                    WHERE pet_id = %s
                    ORDER BY timestamp ASC;
                """, (pet_id,))
                pg_rows = cur.fetchall()
                if pg_rows:
                    s_conn = sqlite3.connect(str(db_path))
                    for r in pg_rows:
                        ts_iso = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"])
                        s_conn.execute("""
                            INSERT INTO emotion_entries (
                                pet_id, timestamp, primary_emotion, primary_confidence,
                                all_emotions, stress_level, prosodic_snapshot, species
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pet_id,
                            ts_iso,
                            r["primary_emotion"],
                            float(r["primary_confidence"]),
                            json.dumps(r["all_emotions"] or []),
                            float(r["stress_level"]),
                            json.dumps(r["prosodic_snapshot"] or {}),
                            r["species"] or "",
                        ))
                    s_conn.commit()
                    s_conn.close()

            _restored_telemetry_pets.add(pet_id)
    except Exception as e:
        logger.debug(f"restore_pet_telemetry_from_db warning for {pet_id}: {e}")
    finally:
        if owns_conn:
            conn.close()


def _format_pet_row(row: Dict[str, Any]) -> Dict[str, Any]:
    created_at = row.get("created_at")
    return {
        "id": row["id"],
        "name": row["name"],
        "species": row["species"],
        "breed": row.get("breed") or "",
        "age": row.get("age") or "",
        "gender": row.get("gender") or "Unknown",
        "notes": row.get("notes") or "",
        "photo": row.get("photo") or "",
        "emoji": row.get("emoji") or "",
        "createdAt": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
    }


def _format_translation_row(row: Dict[str, Any]) -> Dict[str, Any]:
    created_at = row.get("created_at")
    created_iso = row.get("created_iso") or (
        created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or "")
    )
    return {
        "id": row.get("entry_id") or f"db-{row.get('id', '')}",
        "createdAt": created_iso,
        "petId": row.get("pet_id") or "",
        "petName": row.get("pet_name") or row.get("species") or "Unknown animal",
        "emoji": "",
        "translation": row.get("translation") or "",
        "behavior": row.get("behavior") or "",
        "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
        "result": row.get("result_json") or {},
    }
