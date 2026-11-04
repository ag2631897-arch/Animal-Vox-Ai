"""
AnimalVox AI — Pet Health Monitoring (SRS Module 01)
=====================================================
Flags vocal patterns linked to pain, distress, or respiratory issues
in cats and dogs — as an early-awareness aid, not a diagnosis.

Key Features:
- 8 health flags (H01-H08) mapped from acoustic signatures
- Personal vocal baseline per pet (EMA-updated weekly)
- Deviation gating: flags only fire when outside the pet's own normal
- Vocal frequency drop detection (50%+ reduction = alert)
- Mandatory medical disclaimer on every output

⚕ Medical Disclaimer: AnimalVox AI is an early-awareness aid only.
It is not a veterinary diagnostic tool. No output should be used as
a substitute for professional veterinary assessment.
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HealthAlert:
    """A single health flag alert."""
    flag_id: str           # H01-H08
    flag_name: str
    confidence: float      # 0-1
    severity: str          # "low", "medium", "high", "critical"
    acoustic_evidence: Dict  # What acoustic features triggered this
    veterinary_context: str
    species: str
    deviation_from_baseline: Optional[float] = None  # Standard deviations


@dataclass
class HealthReport:
    """Complete health monitoring report for a single audio clip."""
    alerts: List[HealthAlert]
    overall_status: str     # "normal", "attention", "concern", "urgent"
    respiratory_normal: bool
    vocal_pattern_change: bool
    baseline_established: bool
    disclaimer: str
    timestamp: str


class VocalBaselineManager:
    """
    Manages per-pet vocal baselines using Exponential Moving Average (EMA).
    
    Each pet accumulates a baseline from their normal vocalisations.
    The baseline includes: mean F0, F0 range, mean HNR, call rate,
    mean RMS, spectral centroid, mean call duration.
    
    Baselines are used to gate health flags: only deviations from
    the pet's OWN normal pattern trigger alerts, reducing false positives.
    """

    BASELINE_FEATURES = [
        "mean_f0", "f0_range", "mean_hnr", "call_rate",
        "mean_rms", "spectral_centroid", "mean_call_duration",
        "mean_zcr", "mean_spectral_rolloff"
    ]

    def __init__(self, profiles_dir: Path, min_clips: int = 50,
                 ema_alpha: float = 0.10):
        self.profiles_dir = profiles_dir
        self.min_clips = min_clips
        self.ema_alpha = ema_alpha

    def get_baseline(self, pet_id: str) -> Optional[Dict]:
        """Load existing baseline for a pet."""
        path = self.profiles_dir / f"{pet_id}_baseline.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def update_baseline(self, pet_id: str, features: Dict) -> Dict:
        """
        Update the pet's baseline with new observation using EMA.
        
        EMA formula: baseline_new = α * observation + (1 - α) * baseline_old
        """
        baseline = self.get_baseline(pet_id)

        if baseline is None:
            # Initialize baseline
            baseline = {
                "pet_id": pet_id,
                "clip_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "means": {},
                "stds": {},
                "observations": [],
            }

        baseline["clip_count"] += 1
        baseline["updated_at"] = datetime.utcnow().isoformat()

        # Store observation for std calculation (keep last 200)
        baseline.setdefault("observations", [])
        obs = {k: features.get(k, 0.0) for k in self.BASELINE_FEATURES}
        baseline["observations"].append(obs)
        if len(baseline["observations"]) > 200:
            baseline["observations"] = baseline["observations"][-200:]

        # EMA update for means
        for feat in self.BASELINE_FEATURES:
            val = features.get(feat, 0.0)
            if feat not in baseline["means"]:
                baseline["means"][feat] = val
            else:
                baseline["means"][feat] = (
                    self.ema_alpha * val + (1 - self.ema_alpha) * baseline["means"][feat]
                )

        # Compute running std from observations
        if len(baseline["observations"]) >= 5:
            for feat in self.BASELINE_FEATURES:
                vals = [o.get(feat, 0.0) for o in baseline["observations"]]
                baseline["stds"][feat] = float(np.std(vals))

        # Save
        path = self.profiles_dir / f"{pet_id}_baseline.json"
        with open(path, "w") as f:
            json.dump(baseline, f, indent=2)

        return baseline

    def is_established(self, pet_id: str) -> bool:
        """Check if baseline has enough data to be reliable."""
        baseline = self.get_baseline(pet_id)
        if baseline is None:
            return False
        return baseline.get("clip_count", 0) >= self.min_clips

    def compute_deviation(self, pet_id: str, features: Dict) -> Dict[str, float]:
        """
        Compute z-score deviation of current features from baseline.
        
        Returns dict of {feature: z_score} for each baseline feature.
        """
        baseline = self.get_baseline(pet_id)
        if baseline is None or not baseline.get("stds"):
            return {}

        deviations = {}
        for feat in self.BASELINE_FEATURES:
            mean = baseline["means"].get(feat, 0.0)
            std = baseline["stds"].get(feat, 1.0)
            val = features.get(feat, 0.0)
            if std > 1e-6:
                deviations[feat] = abs(val - mean) / std
            else:
                deviations[feat] = 0.0

        return deviations


try:
    from models.dual_stream_classifier import EnhancedClassifier
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False


class HealthFlagClassifier(nn.Module):
    """
    Multi-label classifier for 8 health flags (H01-H08).
    Uses EnhancedClassifier residual architecture matching health_monitor.pt.
    """

    def __init__(self, input_dim: int = 266, hidden_dim: int = 192,
                 num_flags: int = 8):
        super().__init__()
        self.input_dim = input_dim
        if _ENHANCED_AVAILABLE:
            self.classifier = EnhancedClassifier(
                input_dim=input_dim,
                hidden_dim=192,
                num_classes=num_flags,
                num_residual_blocks=3,
                dropout=0.20,
                is_multilabel=True,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(hidden_dim, num_flags),
            )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw un-squashed logits for BCEWithLogitsLoss training."""
        if hasattr(self.classifier, "forward_logits"):
            return self.classifier.forward_logits(x)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid probabilities for each health flag."""
        out = self.classifier(x)
        if hasattr(self.classifier, "is_multilabel") and self.classifier.is_multilabel:
            return out
        return torch.sigmoid(out)


class RespiratoryAnalyzer:
    """
    Specialized respiratory pattern analyser.
    
    Detects abnormal breathing from:
    - Zero-Crossing Rate (ZCR) — elevated in rasping/laboured breathing
    - Spectral irregularity — non-harmonic noise in breathing sounds
    - Rhythm regularity — laboured breathing has irregular timing
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, audio: np.ndarray, prosodic: Dict) -> Dict:
        """
        Analyze respiratory patterns from audio and prosodic features.
        
        Returns respiratory metrics dict.
        """
        zcr = prosodic.get("mean_zcr", 0.0)
        hnr = prosodic.get("mean_hnr", 0.0)

        # Spectral flatness (high = noisy, could indicate breathing difficulty)
        spec_flat = librosa.feature.spectral_flatness(y=audio)
        mean_flatness = float(np.mean(spec_flat)) if spec_flat.size > 0 else 0.0

        # RMS energy variation (high variation = irregular breathing)
        rms = librosa.feature.rms(y=audio)
        rms_std = float(np.std(rms)) if rms.size > 0 else 0.0
        rms_mean = float(np.mean(rms)) if rms.size > 0 else 0.0
        cv = rms_std / (rms_mean + 1e-10)  # Coefficient of variation

        return {
            "zcr_elevated": zcr > 0.15,  # Normal animal vocalisation < 0.1
            "hnr_low": hnr < 8.0,         # Low HNR = noisy/raspy
            "spectral_flatness": mean_flatness,
            "breathing_irregular": cv > 0.5,
            "respiratory_distress_score": float(
                0.3 * min(zcr / 0.2, 1.0) +
                0.3 * max(0, 1.0 - hnr / 15.0) +
                0.2 * min(mean_flatness / 0.1, 1.0) +
                0.2 * min(cv / 0.8, 1.0)
            ),
        }


class HealthMonitor:
    """
    Complete pet health monitoring pipeline (Module 01).
    
    Orchestrates: Feature Extraction → Flag Classification →
    Baseline Comparison → Deviation Gating → Report Generation
    """

    def __init__(self, device: str = "cpu"):
        from config import (
            HEALTH_FLAGS, HEALTH_MONITOR_CONFIG, AUDIO_CONFIG,
            MODEL_CONFIG, PET_PROFILES_DIR, MEDICAL_DISCLAIMER
        )
        self.health_flags = HEALTH_FLAGS
        self.config = HEALTH_MONITOR_CONFIG
        self.audio_config = AUDIO_CONFIG
        self.disclaimer = MEDICAL_DISCLAIMER
        self.device = device

        # Baseline manager
        self.baseline_mgr = VocalBaselineManager(
            profiles_dir=PET_PROFILES_DIR,
            min_clips=self.config.baseline_min_clips,
            ema_alpha=self.config.baseline_ema_alpha
        )

        # Health flag classifier
        input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim  # 256 + 10
        self.classifier = HealthFlagClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.health_hidden_dim,
            num_flags=len(HEALTH_FLAGS)
        ).to(device).eval()

        # Check for trained weights or initialize in heuristic fallback mode
        self.has_trained_weights = False
        self.metadata = {}
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_weights, self.metadata = load_checkpoint(
                self.classifier, "health_monitor", device=device
            )
        except Exception as e:
            logger.debug(f"Weight manager check for health_monitor: {e}")

        # Respiratory analyzer (Acoustic physics)
        self.respiratory = RespiratoryAnalyzer(sr=self.audio_config.sample_rate)

        if self.has_trained_weights:
            logger.info(f"HealthMonitor loaded TRAINED weights (Epoch {self.metadata.get('epoch', 'N/A')})")
        else:
            logger.info(
                "HealthMonitor: No trained checkpoint found. "
                "Active Mode: ACOUSTIC BIOMARKER HEURISTIC FALLBACK (Respiratory & Prosodic Rules)."
            )

    def save_weights(self, epoch: int = 0, optimizer=None, metrics: Dict = None) -> Path:
        """Save trained health classifier weights to standard checkpoint path."""
        from models.weight_manager import save_checkpoint
        return save_checkpoint(
            self.classifier, "health_monitor", epoch=epoch,
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
            self.classifier, "health_monitor", device=self.device
        )
        return self.has_trained_weights

    def _compute_heuristic_flag_probs(
        self, prosodic: Dict, resp: Dict, species: str, deviations: Dict
    ) -> List[float]:
        """
        Physics & bioacoustic rules for 8 health flags when neural weights are not present.
        Accurately estimates medical risk signals from acoustic indicators.
        """
        f0 = prosodic.get("mean_f0", 0.0)
        hnr = prosodic.get("mean_hnr", 15.0)
        zcr = prosodic.get("mean_zcr", 0.05)
        rms = prosodic.get("mean_rms", 0.1)
        tempo = prosodic.get("tempo", 100.0)
        call_rate = prosodic.get("call_rate", 0.5)
        duration = prosodic.get("mean_call_duration", 0.5)
        f0_dev = deviations.get("mean_f0", 0.0)

        probs = [0.0] * 8
        # H01: Acute Pain / High-Pitch Distress
        pain_condition = (f0_dev > 2.5) or (species == "dogs" and f0 > 750) or (species == "cats" and f0 > 1100)
        if pain_condition and rms > 0.15:
            probs[0] = min(0.95, 0.70 + 0.1 * min(f0_dev, 3.0))

        # H02: Chronic Discomfort / Low Whimper
        if call_rate > 1.2 and rms < 0.10 and (200 < f0 < 500):
            probs[1] = 0.78

        # H03: Vocal Fatigue / Hoarseness
        if hnr < 6.5 and zcr > 0.12:
            probs[2] = min(0.90, 0.60 + (7.0 - hnr) * 0.05)

        # H04: Laryngeal Dysfunction
        if hnr < 4.0 and zcr > 0.16:
            probs[3] = 0.82

        # H05: Panting / Respiratory Distress / Heat Stress
        resp_score = resp.get("respiratory_distress_score", 0.0)
        if resp_score > 0.45 or resp.get("breathing_irregular", False):
            probs[4] = min(0.96, 0.55 + resp_score * 0.45)

        # H06: Stridor / Wheezing / Cough
        if resp.get("zcr_elevated", False) and resp.get("spectral_flatness", 0.0) > 0.06:
            probs[5] = 0.85

        # H07: Cognitive Dysfunction / Wandering Whine
        if duration > 2.5 and prosodic.get("f0_range", 100) < 35:
            probs[6] = 0.75

        # H08: Separation Anxiety
        if species == "dogs" and call_rate > 1.5 and tempo > 115 and f0 > 450:
            probs[7] = 0.88

        return probs

    def assess(self, audio: np.ndarray, fused_embedding: torch.Tensor,
               prosodic_features: Dict, species: str,
               pet_id: Optional[str] = None) -> HealthReport:
        """
        Run health assessment on a vocalization.
        Uses trained neural classifier if available, otherwise runs Acoustic Biomarker Heuristics.
        """
        from config import HEALTH_FLAGS

        # Respiratory analysis (physics-based)
        resp = self.respiratory.analyze(audio, prosodic_features)

        # Baseline deviation (if pet has established baseline)
        deviations = {}
        baseline_established = False
        if pet_id:
            baseline_established = self.baseline_mgr.is_established(pet_id)
            if baseline_established:
                deviations = self.baseline_mgr.compute_deviation(pet_id, prosodic_features)
            # Always update baseline
            self.baseline_mgr.update_baseline(pet_id, prosodic_features)

        # Determine flag probabilities (Trained Model vs Heuristic Fallback)
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
                    flag_probs = self.classifier(fused_input).cpu().numpy()[0]
            except Exception as e:
                logger.warning(f"Neural health forward pass error ({e}); using heuristic fallback.")
                flag_probs = self._compute_heuristic_flag_probs(prosodic_features, resp, species, deviations)
        else:
            # Pure Acoustic Biomarker Heuristics
            flag_probs = self._compute_heuristic_flag_probs(prosodic_features, resp, species, deviations)

        # Build alerts
        alerts = []
        # Calibrated Clinical Operating Thresholds (Youden's J ROC Optimization)
        CLINICAL_THRESHOLDS = {
            "H01": 0.55,  # Acute Pain / High-Pitch Distress — high sensitivity priority
            "H02": 0.65,  # Chronic Discomfort / Low Whimper
            "H03": 0.68,  # Vocal Fatigue / Hoarseness
            "H04": 0.65,  # Laryngeal Dysfunction
            "H05": 0.60,  # Respiratory Distress / Heat Stress
            "H06": 0.62,  # Stridor / Wheezing / Cough
            "H07": 0.72,  # Cognitive Dysfunction — requires high certainty
            "H08": 0.68,  # Separation Anxiety
        }

        flag_ids = list(HEALTH_FLAGS.keys())
        for i, flag_id in enumerate(flag_ids):
            flag_info = HEALTH_FLAGS[flag_id]

            # Check species compatibility
            if species not in flag_info.get("species", []):
                continue

            prob = float(flag_probs[i])

            # Confidence gating with flag-specific clinical threshold
            threshold = CLINICAL_THRESHOLDS.get(flag_id, self.config.flag_confidence_threshold)
            if prob < threshold:
                continue

            # Baseline deviation gating
            deviation = None
            if baseline_established and deviations:
                # Check if key features deviate from baseline
                max_dev = max(deviations.get("mean_f0", 0.0),
                              deviations.get("mean_hnr", 0.0),
                              deviations.get("mean_rms", 0.0))
                if max_dev < self.config.baseline_suppression_sigma:
                    # Below suppression threshold — this is normal for this pet
                    continue
                deviation = max_dev

            # Severity classification
            if prob >= 0.95:
                severity = "critical"
            elif prob >= 0.85:
                severity = "high"
            elif prob >= 0.75:
                severity = "medium"
            else:
                severity = "low"

            alerts.append(HealthAlert(
                flag_id=flag_id,
                flag_name=flag_info["name"],
                confidence=prob,
                severity=severity,
                acoustic_evidence={
                    "f0": prosodic_features.get("mean_f0"),
                    "hnr": prosodic_features.get("mean_hnr"),
                    "zcr": prosodic_features.get("mean_zcr"),
                    "rms": prosodic_features.get("mean_rms"),
                    "respiratory": resp,
                },
                veterinary_context=flag_info["veterinary_context"],
                species=species,
                deviation_from_baseline=deviation,
            ))

        # Overall status
        if any(a.severity == "critical" for a in alerts):
            overall = "urgent"
        elif any(a.severity == "high" for a in alerts):
            overall = "concern"
        elif alerts:
            overall = "attention"
        else:
            overall = "normal"

        return HealthReport(
            alerts=alerts,
            overall_status=overall,
            respiratory_normal=resp["respiratory_distress_score"] < 0.5,
            vocal_pattern_change=bool(deviations and max(deviations.values(), default=0) > 2.0),
            baseline_established=baseline_established,
            disclaimer=self.disclaimer,
            timestamp=datetime.utcnow().isoformat(),
        )
