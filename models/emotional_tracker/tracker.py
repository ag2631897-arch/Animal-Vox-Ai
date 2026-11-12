"""
AnimalVox AI — Emotional & Behavioural Tracking (SRS Module 03)
================================================================
Tracks emotional states (E01-E10) over time, stores them in a
persistent diary database, and generates weekly anomaly reports.

Key Features:
- 10 emotional states classified from acoustic features
- SQLite-based emotional diary (timestamped entries per pet)
- Weekly trend analysis with anomaly detection (> 2σ stress)
- Rolling emotion distribution and mood stability metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class EmotionPrediction:
    """A single emotion state prediction."""
    state_id: str          # E01-E10
    state_name: str
    confidence: float
    acoustic_markers: str  # What acoustic evidence supports this
    context: str           # Likely real-world context


@dataclass
class EmotionReport:
    """Emotional assessment for a single audio clip."""
    primary_emotion: EmotionPrediction
    secondary_emotions: List[EmotionPrediction]
    mood_stability: float  # 0-1 (how consistent is mood recently)
    stress_level: float    # 0-1 (composite stress indicator)
    timestamp: str
    pet_id: Optional[str]


@dataclass
class WeeklyDigest:
    """Weekly emotional summary for a pet."""
    pet_id: str
    week_start: str
    week_end: str
    dominant_emotion: str
    emotion_distribution: Dict[str, float]  # emotion_name: percentage
    stress_episodes: int
    contentment_episodes: int
    mood_stability: float
    anomalies: List[str]
    total_entries: int


try:
    from models.dual_stream_classifier import DualStreamBiometricClassifier
    _DUAL_STREAM_AVAILABLE = True
except ImportError:
    _DUAL_STREAM_AVAILABLE = False


class EmotionClassifier(nn.Module):
    """
    Multi-class emotion classifier for 10 emotional states.
    Uses DualStreamBiometricClassifier for morphological biometric gating.
    """

    def __init__(self, input_dim: int = 266, hidden_dim: int = 160,
                 num_classes: int = 10):
        super().__init__()
        if _DUAL_STREAM_AVAILABLE:
            self.model = DualStreamBiometricClassifier(
                semantic_dim=256,
                biometric_dim=10,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_residual_blocks=2,
                dropout=0.15,
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EmotionalDiary:
    """
    SQLite-backed emotional diary for longitudinal tracking.
    
    Stores timestamped emotion entries per pet, enabling:
    - Weekly/monthly trend analysis
    - Anomaly detection (unusual stress spikes)
    - Mood stability scoring
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
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
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pet_time 
            ON emotion_entries (pet_id, timestamp)
        """)
        conn.commit()
        conn.close()

    def add_entry(self, pet_id: str, emotion: EmotionPrediction,
                  all_emotions: List[EmotionPrediction], stress_level: float,
                  prosodic: Dict = None, species: str = ""):
        """Record an emotional observation."""
        conn = sqlite3.connect(str(self.db_path))
        emotions_json = json.dumps([
            {"id": str(e.state_id), "name": str(e.state_name), "conf": float(e.confidence)}
            for e in all_emotions
        ])
        clean_prosodic = None
        if prosodic:
            clean_prosodic = {
                str(k): (float(v) if isinstance(v, (np.floating, float, np.integer, int)) else v)
                for k, v in prosodic.items()
            }
        conn.execute(
            """INSERT INTO emotion_entries 
               (pet_id, timestamp, primary_emotion, primary_confidence,
                all_emotions, stress_level, prosodic_snapshot, species)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (pet_id, datetime.utcnow().isoformat(), str(emotion.state_id),
             float(emotion.confidence), emotions_json, float(stress_level),
             json.dumps(clean_prosodic) if clean_prosodic else None, species)
        )
        conn.commit()
        conn.close()

    def get_entries(self, pet_id: str, days: int = 7) -> List[Dict]:
        """Get recent diary entries for a pet, falling back to full history if no entries in window."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            """SELECT timestamp, primary_emotion, primary_confidence,
                      all_emotions, stress_level, prosodic_snapshot, species
               FROM emotion_entries
               WHERE pet_id = ? AND timestamp >= ?
               ORDER BY timestamp DESC""",
            (pet_id, since)
        )
        rows = cursor.fetchall()
        if not rows:
            cursor = conn.execute(
                """SELECT timestamp, primary_emotion, primary_confidence,
                          all_emotions, stress_level, prosodic_snapshot, species
                   FROM emotion_entries
                   WHERE pet_id = ?
                   ORDER BY timestamp DESC
                   LIMIT 100""",
                (pet_id,)
            )
            rows = cursor.fetchall()
        entries = []
        for row in rows:
            stress_val = float(row[4]) if isinstance(row[4], (int, float)) else 0.0
            prosodic_snap = {}
            if len(row) > 5 and row[5]:
                try:
                    prosodic_snap = json.loads(row[5])
                except Exception:
                    prosodic_snap = {}
            entries.append({
                "timestamp": row[0],
                "primary_emotion": row[1],
                "confidence": float(row[2]) if isinstance(row[2], (int, float)) else 0.0,
                "all_emotions": json.loads(row[3]) if row[3] else [],
                "stress_level": stress_val,
                "prosodic_snapshot": prosodic_snap,
                "species": row[6] if len(row) > 6 else "",
            })
        conn.close()
        return entries

    def compute_weekly_digest(self, pet_id: str) -> Optional[WeeklyDigest]:
        """Generate weekly emotional summary with anomaly detection."""
        entries = self.get_entries(pet_id, days=7)
        if not entries:
            return None

        # Emotion distribution
        emotion_counts = {}
        stress_values = []
        for entry in entries:
            emotion = entry["primary_emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            stress_values.append(entry["stress_level"])

        total = len(entries)
        distribution = {k: v / total for k, v in emotion_counts.items()}

        # Dominant emotion
        dominant = max(emotion_counts, key=emotion_counts.get)

        # Stress episodes (stress > 0.6)
        stress_episodes = sum(1 for s in stress_values if s > 0.6)
        contentment_episodes = sum(
            1 for e in entries if e["primary_emotion"] in ("E01", "E06", "E07")
        )

        # Mood stability (inverse of emotion switching frequency)
        if len(entries) > 1:
            switches = sum(
                1 for i in range(1, len(entries))
                if entries[i]["primary_emotion"] != entries[i-1]["primary_emotion"]
            )
            stability = 1.0 - (switches / (len(entries) - 1))
        else:
            stability = 1.0

        # Anomaly detection (stress > 2σ above mean)
        anomalies = []
        if len(stress_values) >= 5:
            mean_stress = np.mean(stress_values)
            std_stress = np.std(stress_values)
            if std_stress > 0:
                for i, s in enumerate(stress_values):
                    if (s - mean_stress) / std_stress > 2.0:
                        anomalies.append(
                            f"Stress spike at {entries[i]['timestamp']}: "
                            f"{s:.2f} (mean: {mean_stress:.2f})"
                        )

        now = datetime.utcnow()
        week_start = (now - timedelta(days=7)).isoformat()

        return WeeklyDigest(
            pet_id=pet_id,
            week_start=week_start,
            week_end=now.isoformat(),
            dominant_emotion=dominant,
            emotion_distribution=distribution,
            stress_episodes=stress_episodes,
            contentment_episodes=contentment_episodes,
            mood_stability=stability,
            anomalies=anomalies,
            total_entries=total,
        )


class EmotionalTracker:
    """
    Complete emotional tracking pipeline (Module 03).
    
    Orchestrates: Classification → Diary Entry → Trend Analysis
    """

    def __init__(self, device: str = "cpu"):
        from config import (
            EMOTIONAL_STATES, EMOTIONAL_TRACKING_CONFIG,
            MODEL_CONFIG, DIARY_DIR
        )
        self.states = EMOTIONAL_STATES
        self.config = EMOTIONAL_TRACKING_CONFIG
        self.state_ids = list(EMOTIONAL_STATES.keys())
        self.device = device

        # Emotion classifier
        input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim
        self.classifier = EmotionClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.emotion_hidden_dim,
            num_classes=self.config.num_emotion_classes
        ).to(device).eval()

        # Check for trained weights or initialize in heuristic fallback mode
        self.has_trained_weights = False
        self.metadata = {}
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_weights, self.metadata = load_checkpoint(
                self.classifier, "emotion_classifier", device=device
            )
        except Exception as e:
            logger.debug(f"Weight manager check for emotion_classifier: {e}")

        # Emotional diary
        db_path = DIARY_DIR / self.config.diary_db_name
        self.diary = EmotionalDiary(db_path)

        if self.has_trained_weights:
            logger.info(f"EmotionalTracker loaded TRAINED weights (Epoch {self.metadata.get('epoch', 'N/A')})")
        else:
            logger.info(
                "EmotionalTracker: No trained checkpoint found. "
                "Active Mode: BIOACOUSTIC HEURISTIC FALLBACK (Morton's Structural Rules)."
            )

    def save_weights(self, epoch: int = 0, optimizer=None, metrics: Dict = None) -> Path:
        """Save trained emotion classifier weights to standard checkpoint path."""
        from models.weight_manager import save_checkpoint
        return save_checkpoint(
            self.classifier, "emotion_classifier", epoch=epoch,
            optimizer=optimizer, metrics=metrics
        )

    def load_weights(self, path: Optional[Path] = None) -> bool:
        """Load weights from standard registry or custom path."""
        from models.weight_manager import load_checkpoint
        if path:
            state = torch.load(path, map_location=self.device)
            state_dict = state["state_dict"] if "state_dict" in state else state
            self.classifier.load_state_dict(state_dict, strict=False)
            self.has_trained_weights = True
            return True
        self.has_trained_weights, self.metadata = load_checkpoint(
            self.classifier, "emotion_classifier", device=self.device
        )
        return self.has_trained_weights

    def _compute_heuristic_emotion_probs(self, prosodic: Dict, species: str) -> np.ndarray:
        """
        Morton's Structural-Demonic Rules for animal emotional vocalizations:
        - High frequency & high tonality (HNR) = friendly, fearful, or excited
        - Low frequency & harsh noise (low HNR) = aggressive, threatening
        - Moderate & steady = calm, content
        """
        f0 = prosodic.get("mean_f0", 400.0)
        f0_range = prosodic.get("f0_range", 100.0)
        hnr = prosodic.get("mean_hnr", 12.0)
        rms = prosodic.get("mean_rms", 0.1)
        tempo = prosodic.get("tempo", 100.0)
        duration = prosodic.get("mean_call_duration", 0.5)

        raw_scores = np.zeros(len(self.state_ids))

        # E01: Joy / Excitement (High pitch, high tempo, good HNR)
        raw_scores[0] = (f0 / 500.0) * 0.4 + (tempo / 120.0) * 0.3 + (rms / 0.2) * 0.3

        # E02: Contentment / Calm (Moderate steady pitch, low tempo, good HNR)
        raw_scores[1] = max(0.0, 1.0 - abs(f0 - 300.0) / 300.0) * 0.5 + max(0.0, 1.0 - tempo / 120.0) * 0.5

        # E03: Curiosity / Interest (Rising pitch contour, moderate duration)
        raw_scores[2] = (f0_range / 200.0) * 0.5 + (0.5 if 0.2 < duration < 1.0 else 0.1)

        # E04: Aggression / Threat (Low pitch, low HNR / harsh rasp, high volume)
        raw_scores[3] = max(0.0, 1.0 - f0 / 300.0) * 0.4 + max(0.0, (15.0 - hnr) / 15.0) * 0.4 + (rms / 0.25) * 0.2

        # E05: Sadness / Loneliness (Low energy, falling contour, long duration)
        raw_scores[4] = max(0.0, 1.0 - rms / 0.15) * 0.4 + min(1.0, duration / 1.5) * 0.4 + max(0.0, 1.0 - tempo / 90.0) * 0.2

        # E06: Fear / Panic (Very high erratic pitch, sudden onset)
        raw_scores[5] = (f0 / 700.0) * 0.5 + (f0_range / 300.0) * 0.3 + (rms / 0.2) * 0.2

        # E07: Frustration / Demand (Sharp repetitive bursts, high tempo)
        raw_scores[6] = (tempo / 130.0) * 0.5 + min(1.0, 0.4 / max(duration, 0.05)) * 0.5

        # E08: Pain / Physical Distress (Sudden pitch peak, harsh centroid)
        raw_scores[7] = (f0 / 800.0) * 0.5 + max(0.0, (12.0 - hnr) / 12.0) * 0.5

        # E09: Affection / Bonding (Soft volume, gentle modulation)
        raw_scores[8] = max(0.0, 1.0 - rms / 0.10) * 0.5 + (hnr / 15.0) * 0.5

        # E10: Playfulness / Social Engagement (Modulated pitch, rhythmic)
        raw_scores[9] = (f0_range / 250.0) * 0.5 + (tempo / 110.0) * 0.5

        # Softmax conversion
        exp_scores = np.exp(np.clip(raw_scores, -10, 10))
        return exp_scores / np.sum(exp_scores)

    def assess(self, fused_embedding: torch.Tensor,
               prosodic_features: Dict, species: str,
               pet_id: Optional[str] = None,
               sequence_history: Optional[List[Dict]] = None,
               escalation_pattern: Optional[str] = None) -> EmotionReport:
        """
        Classify emotional state and record to diary.
        Uses trained neural classifier if available, calibrated by Multi-Bout Sequence Smoothing & Morton Bioacoustic Rules.
        """
        if self.has_trained_weights:
            try:
                # fused_embedding is already 266-dim (256 audio_embed + 10 prosodic)
                # — matches the classifier's trained input_dim exactly
                if isinstance(fused_embedding, np.ndarray):
                    fused_embedding = torch.from_numpy(fused_embedding).float()
                if fused_embedding.dim() == 1:
                    fused_embedding = fused_embedding.unsqueeze(0)
                fused_embedding = fused_embedding.to(self.device)

                if fused_embedding.shape[-1] == 256:
                    prosodic_tensor = torch.tensor([
                        prosodic_features.get("mean_f0", 0.0),
                        prosodic_features.get("f0_range", 0.0),
                        prosodic_features.get("mean_hnr", 0.0),
                        prosodic_features.get("call_rate", 0.0),
                        prosodic_features.get("mean_rms", 0.0),
                        prosodic_features.get("spectral_centroid", 0.0),
                        prosodic_features.get("mean_call_duration", 0.0),
                        prosodic_features.get("mean_zcr", 0.0),
                        prosodic_features.get("mean_spectral_rolloff", 0.0),
                        prosodic_features.get("tempo", 0.0),
                    ], dtype=torch.float32).unsqueeze(0).to(self.device)
                    fused_input = torch.cat([fused_embedding, prosodic_tensor], dim=-1)
                elif fused_embedding.shape[-1] > 266:
                    fused_input = fused_embedding[..., :266]
                else:
                    fused_input = fused_embedding

                with torch.no_grad():
                    logits = self.classifier(fused_input)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            except Exception as e:
                logger.warning(f"Neural emotion forward pass error ({e}); using heuristic fallback.")
                probs = self._compute_heuristic_emotion_probs(prosodic_features, species)
        else:
            probs = self._compute_heuristic_emotion_probs(prosodic_features, species)

        # Multi-Bout Temporal Sequence Smoothing
        if sequence_history and len(sequence_history) >= 2:
            recent_probs = []
            for past_call in sequence_history[-3:]:
                if "all_scores" in past_call and len(past_call["all_scores"]) == len(probs):
                    recent_probs.append(np.array(past_call["all_scores"]))
            if recent_probs:
                history_mean = np.mean(recent_probs, axis=0)
                probs = 0.70 * probs + 0.30 * history_mean
                probs = probs / np.sum(probs)

        # Escalation pattern modulation
        if escalation_pattern == "escalation":
            for idx in [0, 5, 6, 7]:  # High-arousal states
                if idx < len(probs):
                    probs[idx] *= 1.25
            probs = probs / np.sum(probs)
        elif escalation_pattern == "de-escalation":
            for idx in [1, 8]:  # Calming/contentment states
                if idx < len(probs):
                    probs[idx] *= 1.30
            probs = probs / np.sum(probs)

        # Build predictions sorted by confidence
        predictions = []
        for i, state_id in enumerate(self.state_ids):
            state_info = self.states[state_id]
            # Filter by species compatibility
            if species in state_info.get("species", []):
                predictions.append(EmotionPrediction(
                    state_id=state_id,
                    state_name=state_info["name"],
                    confidence=float(probs[i]),
                    acoustic_markers=state_info["acoustic_markers"],
                    context=state_info["context"],
                ))
        predictions.sort(key=lambda x: x.confidence, reverse=True)

        primary = predictions[0] if predictions else EmotionPrediction(
            state_id="E01", state_name="Unknown", confidence=0.0,
            acoustic_markers="", context=""
        )
        secondary = predictions[1:4] if len(predictions) > 1 else []

        # Compute stress level (E02, E03, E08 weighted)
        stress_ids = {"E02": 1.0, "E03": 0.7, "E08": 0.5, "E05": 0.3}
        stress_level = sum(
            probs[self.state_ids.index(sid)] * weight
            for sid, weight in stress_ids.items()
            if sid in self.state_ids
        )
        stress_level = min(stress_level, 1.0)

        # Mood stability from recent diary
        mood_stability = 1.0
        if pet_id:
            recent = self.diary.get_entries(pet_id, days=1)
            if len(recent) > 1:
                switches = sum(
                    1 for i in range(1, len(recent))
                    if recent[i]["primary_emotion"] != recent[i-1]["primary_emotion"]
                )
                mood_stability = 1.0 - (switches / (len(recent) - 1))

            # Record to diary
            self.diary.add_entry(
                pet_id=pet_id, emotion=primary,
                all_emotions=predictions, stress_level=stress_level,
                prosodic=prosodic_features, species=species
            )

        return EmotionReport(
            primary_emotion=primary,
            secondary_emotions=secondary,
            mood_stability=mood_stability,
            stress_level=stress_level,
            timestamp=datetime.utcnow().isoformat(),
            pet_id=pet_id,
        )

    def get_weekly_digest(self, pet_id: str) -> Optional[WeeklyDigest]:
        """Get the weekly emotional summary for a pet."""
        return self.diary.compute_weekly_digest(pet_id)
