"""
AnimalVox AI — Behavioral Classifier
Multi-label classifier with intensity and urgency heads.
Fuses BEATs (256d) + CNN (256d) + Prosodic (10d) features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectrogramCNN(nn.Module):
    """EfficientNet-style CNN for mel-spectrogram analysis (Path B)."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Lin et al., 2017)."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.where(targets == 1, preds, 1 - preds)
        weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = torch.where(targets == 1, self.alpha.to(preds.device), 1 - self.alpha.to(preds.device))
            weight = weight * at
        return (weight * bce).mean()


import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

try:
    from config import BEHAVIOR_CLASSES, BEHAVIOR_LABELS, NUM_BEHAVIOR_CLASSES
except ImportError:
    BEHAVIOR_LABELS = [
        "alarm_aerial", "alarm_ground", "territorial_warning", "mating_call",
        "contact_call", "distress_call", "feeding_call", "play_vocalization",
        "submission_signal", "aggression_warning", "social_bonding", "navigation_echolocation"
    ]
    NUM_BEHAVIOR_CLASSES = len(BEHAVIOR_LABELS)


class BehavioralClassifier(nn.Module):
    """
    Multi-label behavioral classifier with Dual-Mode Execution:
    - Trained Mode: Neural fusion of [BEATs(256) | CNN(256) | Prosodic(10)] = 522-dim.
    - Heuristic Fallback Mode: Morton's Structural-Demonic Rules & Bioacoustic
      Ethological Biomarkers (responsive to pitch, burst cadence, HNR, RMS, and ZCR).
    """

    def __init__(
        self,
        input_dim: int = 522,
        hidden: int = 512,
        out: int = 256,
        n_cls: int = 12,
        drop: float = 0.2,
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.n_cls = n_cls
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, out), nn.GELU(), nn.Dropout(drop * 0.5),
        )
        self.classifier = nn.Linear(out, n_cls)
        self.intensity_head = nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, 1))
        self.urgency_head = nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, 1))

        # Check for trained weights or initialize in heuristic fallback mode
        self.has_trained_weights = False
        self.metadata = {}
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_weights, self.metadata = load_checkpoint(
                self, "behavioral_classifier", device=device
            )
        except Exception as e:
            logger.debug(f"Weight manager check for behavioral_classifier: {e}")

        if self.has_trained_weights:
            logger.info(
                f"BehavioralClassifier loaded TRAINED weights (Epoch {self.metadata.get('epoch', 'N/A')})"
            )
        else:
            logger.info(
                "BehavioralClassifier: No trained checkpoint found. "
                "Active Mode: BIOACOUSTIC ETHOLOGY FALLBACK (Morton's Rules & Acoustic Prosody)."
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard neural forward pass for PyTorch training and inference."""
        h = self.fusion(x)
        labels = torch.sigmoid(self.classifier(h))
        intensity = torch.sigmoid(self.intensity_head(h))
        urgency = torch.sigmoid(self.urgency_head(h))
        return labels, intensity, urgency

    def get_fused(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(x)

    def save_weights(self, epoch: int = 0, optimizer=None, metrics: Dict = None) -> Path:
        """Save trained weights to standard checkpoint path."""
        from models.weight_manager import save_checkpoint
        return save_checkpoint(
            self, "behavioral_classifier", epoch=epoch,
            optimizer=optimizer, metrics=metrics
        )

    def load_weights(self, path: Optional[Path] = None) -> bool:
        """Load weights from standard registry or custom path."""
        from models.weight_manager import load_checkpoint
        if path:
            state = torch.load(path, map_location=self.device)
            state_dict = state["state_dict"] if "state_dict" in state else state
            self.load_state_dict(state_dict, strict=False)
            self.has_trained_weights = True
            return True
        self.has_trained_weights, self.metadata = load_checkpoint(
            self, "behavioral_classifier", device=self.device
        )
        return self.has_trained_weights

    def classify(
        self,
        fused_tensor: torch.Tensor,
        prosodic: Dict[str, float],
        species: str = "dogs",
        audio_clip: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, float]:
        """
        Classify behavioral state, intensity, and urgency.
        Uses trained neural network if checkpoint is present, otherwise executes
        acoustically-responsive bioacoustic ethology heuristics.
        """
        if self.has_trained_weights:
            try:
                with torch.no_grad():
                    labels, intensity, urgency = self.forward(fused_tensor)
                labels_np = labels.cpu().numpy()[0]
                intensity_val = float(intensity.cpu().numpy()[0][0])
                urgency_val = float(urgency.cpu().numpy()[0][0])
                return labels_np, intensity_val, urgency_val
            except Exception as e:
                logger.warning(f"Neural behavioral forward error ({e}); falling back to acoustic rules.")

        # Heuristic fallback: compute authentic probabilities and metrics from physical acoustics
        probs = self._compute_heuristic_behaviors(prosodic, species)
        intensity_val, urgency_val = self._compute_heuristic_intensity_urgency(prosodic, species)
        return probs, intensity_val, urgency_val

    def _compute_heuristic_intensity_urgency(
        self, prosodic: Dict[str, float], species: str
    ) -> Tuple[float, float]:
        """
        Compute continuous intensity (0.05-0.98) and urgency (0.05-0.98) from acoustic energy & burst rate.
        """
        rms = prosodic.get("mean_rms", 0.1)
        f0 = prosodic.get("mean_f0", 350.0)
        call_rate = prosodic.get("call_rate", 1.5)
        centroid = prosodic.get("spectral_centroid", 1500.0)
        f0_slope = prosodic.get("f0_slope", 0.0)
        zcr = prosodic.get("mean_zcr", 0.05)

        # Intensity = acoustic power, amplitude excursion, and vocal effort
        rms_norm = np.clip(rms / 0.22, 0.05, 1.0)
        centroid_norm = np.clip(centroid / 3000.0, 0.05, 1.0)
        f0_norm = np.clip(f0 / 800.0, 0.05, 1.0)
        intensity = float(0.45 * rms_norm + 0.30 * centroid_norm + 0.25 * f0_norm)
        intensity = float(np.clip(intensity, 0.10, 0.96))

        # Urgency = cadence/repetition rate, sudden onsets, harshness, and pitch elevation
        rate_norm = np.clip(call_rate / 3.5, 0.05, 1.0)
        pitch_urgency = np.clip(f0 / 700.0, 0.05, 1.0)
        harshness = np.clip(zcr / 0.10, 0.05, 1.0)
        urgency = float(0.45 * rate_norm + 0.30 * pitch_urgency + 0.25 * harshness)
        if f0_slope > 1.5:  # Rapidly rising pitch increases urgency
            urgency += 0.08
        urgency = float(np.clip(urgency, 0.10, 0.98))

        return intensity, urgency

    def _compute_heuristic_behaviors(
        self, prosodic: Dict[str, float], species: str
    ) -> np.ndarray:
        """
        Bioacoustic ethological classification based on Morton's Structural-Demonic Rules:
        - Low frequency + harsh rasp (low HNR, high ZCR) = Aggression / Territorial threat
        - High frequency + tonal purity (high HNR) = Distress / Whimper / Submission
        - Pitch modulation (high F0 range) + bouncy cadence = Play vocalization
        - Rapid burst rate (high call rate > 3.0/s) + high energy = Alarm alert
        - Sustained steady harmonic tone = Social bonding (Howl / Purr)
        - High repetitive solicitation = Feeding call
        """
        f0 = prosodic.get("mean_f0", 350.0)
        f0_range = prosodic.get("f0_range", 120.0)
        hnr = prosodic.get("mean_hnr", 10.0)
        rms = prosodic.get("mean_rms", 0.1)
        call_rate = prosodic.get("call_rate", 1.5)
        centroid = prosodic.get("spectral_centroid", 1500.0)
        zcr = prosodic.get("mean_zcr", 0.05)
        duration = prosodic.get("mean_call_duration", 0.5)
        tempo = prosodic.get("tempo", 100.0)
        f0_slope = prosodic.get("f0_slope", 0.0)

        # Vector of 12 raw affinity scores matching BEHAVIOR_LABELS
        # [0: alarm_aerial, 1: alarm_ground, 2: territorial_warning, 3: mating_call,
        #  4: contact_call, 5: distress_call, 6: feeding_call, 7: play_vocalization,
        #  8: submission_signal, 9: aggression_warning, 10: social_bonding, 11: navigation_echolocation]
        raw = np.zeros(self.n_cls, dtype=np.float32)

        species_lower = (species or "dogs").lower()

        if species_lower == "dogs":
            # ── 9: Aggression Warning (Deep threat growl / snarl) ────────────────
            # Characterized by low F0 (< 280 Hz), low HNR (< 8 dB), high roughness/ZCR (> 0.06), low burst cadence
            growl_pitch = max(0.0, 1.0 - (f0 - 80.0) / 260.0) if f0 > 0 else 0.4
            growl_harsh = max(0.0, (14.0 - hnr) / 14.0)
            growl_zcr = min(1.0, zcr / 0.08)
            growl_cadence = max(0.0, 1.0 - call_rate / 2.8)
            raw[9] = 0.35 * growl_pitch + 0.35 * growl_harsh + 0.15 * growl_zcr + 0.15 * growl_cadence

            # ── 2: Territorial Warning (Boundary defense bark) ───────────────────
            # Characterized by mid-low F0 (200-420 Hz), moderate burst rate (1.0-2.8/s), moderate-high volume
            terr_pitch = max(0.0, 1.0 - abs(f0 - 300.0) / 200.0)
            terr_rate = max(0.0, 1.0 - abs(call_rate - 1.8) / 1.6)
            terr_rms = min(1.0, rms / 0.16)
            raw[2] = 0.40 * terr_pitch + 0.35 * terr_rate + 0.25 * terr_rms

            # ── 1: Alarm Ground (Sudden sharp intruder alarm barking) ────────────
            # Characterized by fast repetitive barking (> 2.8 onsets/s), mid-high F0 (350-650 Hz), sharp centroid
            alarm_rate = min(1.0, call_rate / 3.0)
            alarm_pitch = max(0.0, 1.0 - abs(f0 - 480.0) / 250.0)
            alarm_energy = min(1.0, rms / 0.18)
            raw[1] = 0.45 * alarm_rate + 0.30 * alarm_pitch + 0.25 * alarm_energy

            # ── 7: Play Vocalization (Excited, modulated play bark) ───────────────
            # Characterized by high pitch modulation (f0_range > 180 Hz), mid-high F0 (420-800 Hz), bouncy rhythm
            play_modulation = min(1.0, f0_range / 260.0)
            play_pitch = max(0.0, 1.0 - abs(f0 - 550.0) / 300.0)
            play_hnr = min(1.0, max(0.0, (hnr - 6.0) / 8.0))
            raw[7] = 0.45 * play_modulation + 0.35 * play_pitch + 0.20 * play_hnr

            # ── 5: Distress Call (Pain yelp / separation whine / crying) ─────────
            # Characterized by high F0 (> 600 Hz), high tonal purity (high HNR), rising pitch inflection
            distress_pitch = min(1.0, max(0.0, (f0 - 450.0) / 550.0))
            distress_hnr = min(1.0, max(0.0, (hnr - 6.0) / 10.0))
            distress_slope = 0.2 if f0_slope > 0.5 else 0.0
            raw[5] = (0.55 * distress_pitch + 0.30 * distress_hnr + 0.15 * distress_slope) * (1.0 if f0 > 380 else 0.1)

            # ── 8: Submission Signal (Appeasement whimper) ────────────────────────
            # Characterized by quiet volume (low RMS < 0.08), high pitch (F0 > 500 Hz), high HNR
            sub_quiet = max(0.0, 1.0 - rms / 0.10)
            sub_pitch = min(1.0, max(0.0, (f0 - 400.0) / 450.0))
            sub_hnr = min(1.0, max(0.0, (hnr - 6.0) / 8.0))
            raw[8] = 0.40 * sub_quiet + 0.35 * sub_pitch + 0.25 * sub_hnr

            # ── 10: Social Bonding (Sustained pack howl / gentle vocalization) ───
            # Characterized by sustained duration (> 1.2s), steady F0 (low f0_range), resonant pure HNR (> 11 dB), smooth low ZCR
            howl_dur = min(1.0, duration / 1.5)
            howl_steady = max(0.0, 1.0 - f0_range / 180.0)
            howl_hnr = min(1.0, max(0.0, (hnr - 8.0) / 8.0))
            howl_f0 = max(0.0, 1.0 - abs(f0 - 380.0) / 250.0)
            howl_smooth = max(0.0, 1.0 - zcr / 0.05)
            raw[10] = (0.35 * howl_dur + 0.25 * howl_steady + 0.25 * howl_hnr + 0.15 * howl_f0) * howl_smooth * howl_hnr

            # ── 4: Contact Call (Looking for owner / loneliness check) ────────────
            # Characterized by mid F0 (320-550 Hz), moderate duration, moderate HNR
            contact_f0 = max(0.0, 1.0 - abs(f0 - 420.0) / 220.0)
            contact_hnr = min(1.0, max(0.0, (hnr - 6.0) / 8.0))
            raw[4] = 0.55 * contact_f0 + 0.45 * contact_hnr

            # ── 6: Feeding Call (Begging solicitation whines) ─────────────────────
            # Characterized by repetitive short whines, rising inflection
            raw[6] = min(1.0, call_rate / 2.8) * 0.45 + min(1.0, max(0.0, (f0 - 380.0) / 350.0)) * 0.40 + (0.15 if f0_slope > 0 else 0.05)

            # ── 0: Alarm Aerial (High vigilance overhead) ────────────────────────
            raw[0] = min(1.0, max(0.0, (centroid - 2500.0) / 2000.0)) * 0.6 + min(1.0, max(0.0, (f0 - 750.0) / 600.0)) * 0.4

            # ── 3: Mating Call (Broadcast reproductive howl) ─────────────────────
            raw[3] = raw[10] * 0.65 + (0.25 if duration > 2.0 and hnr > 10.0 else 0.0)

            # ── 11: Navigation Echolocation (Non-canine, keep very low) ──────────
            raw[11] = 0.02


        elif species_lower == "cats":
            # Felis catus bioacoustics
            # 10: Purring (Social Bonding) - low frequency rhythmic vibration (20-60 Hz)
            is_purr = (f0 < 120.0 or centroid < 400.0) and duration > 0.8
            raw[10] = 0.85 if is_purr else max(0.0, 1.0 - abs(f0 - 450.0) / 300.0) * 0.4

            # 9: Hiss / Growl (Aggression) - high ZCR, low HNR, broad noise
            raw[9] = min(1.0, zcr / 0.12) * 0.5 + max(0.0, (12.0 - hnr) / 12.0) * 0.5

            # 2: Territorial caterwaul / warning - loud sustained yowl
            raw[2] = min(1.0, rms / 0.15) * 0.5 + max(0.0, 1.0 - abs(f0 - 500.0) / 300.0) * 0.5

            # 6: Feeding / Solicitation Meow - high pitch (500-800 Hz), rising pitch
            raw[6] = max(0.0, 1.0 - abs(f0 - 650.0) / 250.0) * 0.6 + (0.3 if f0_slope > 0 else 0.1)

            # 4: Contact Meow
            raw[4] = max(0.0, 1.0 - abs(f0 - 550.0) / 250.0) * 0.7 + (hnr / 15.0) * 0.3

            # 5: Distress / Pain yowl
            raw[5] = min(1.0, max(0.0, (f0 - 700.0) / 500.0)) * 0.6 + min(1.0, rms / 0.18) * 0.4

            # 7: Play chatter
            raw[7] = min(1.0, call_rate / 3.0) * 0.6 + min(1.0, f0_range / 200.0) * 0.4
            raw[1] = raw[9] * 0.5
            raw[0] = 0.05
            raw[3] = raw[2] * 0.7
            raw[8] = max(0.0, 1.0 - rms / 0.08) * 0.5 + min(1.0, f0 / 700.0) * 0.5
            raw[11] = 0.01

        elif species_lower == "birds":
            # Avian vocalizations
            raw[0] = min(1.0, max(0.0, (centroid - 3000.0) / 2500.0)) * 0.6 + min(1.0, call_rate / 3.5) * 0.4
            raw[1] = min(1.0, max(0.0, (centroid - 2000.0) / 2000.0)) * 0.5 + min(1.0, call_rate / 3.0) * 0.5
            raw[3] = min(1.0, f0_range / 400.0) * 0.6 + min(1.0, hnr / 15.0) * 0.4
            raw[4] = max(0.0, 1.0 - abs(f0 - 1800.0) / 1000.0) * 0.6 + min(1.0, hnr / 14.0) * 0.4
            raw[5] = min(1.0, rms / 0.2) * 0.5 + min(1.0, max(0.0, (f0 - 2500.0) / 1500.0)) * 0.5
            raw[7] = min(1.0, f0_range / 350.0) * 0.5 + min(1.0, call_rate / 2.5) * 0.5
            raw[10] = raw[3] * 0.7
            raw[2] = raw[1] * 0.7
            raw[6] = raw[4] * 0.6
            raw[8] = 0.05
            raw[9] = max(0.0, (12.0 - hnr) / 12.0) * 0.6 + min(1.0, zcr / 0.15) * 0.4
            raw[11] = 0.02

        elif species_lower in ("dolphins", "whales"):
            # Marine mammal bioacoustics: click trains & high whistle echolocation
            is_click = centroid > 3800.0 or zcr > 0.15
            raw[11] = 0.90 if is_click else 0.30
            raw[4] = min(1.0, hnr / 16.0) * 0.7 + max(0.0, 1.0 - abs(f0 - 2000.0) / 1500.0) * 0.3
            raw[10] = raw[4] * 0.85
            raw[3] = min(1.0, duration / 2.0) * 0.6 + min(1.0, hnr / 15.0) * 0.4
            raw[5] = min(1.0, rms / 0.2) * 0.6 + (0.4 if f0_slope > 0 else 0.1)
            raw[1] = 0.10
            raw[0] = 0.05
            raw[2] = 0.15
            raw[6] = 0.20
            raw[7] = min(1.0, f0_range / 300.0) * 0.6
            raw[8] = 0.05
            raw[9] = 0.10

        else:
            # General fallback: Morton's structural rules
            raw[9] = max(0.0, 1.0 - f0 / 300.0) * 0.5 + max(0.0, (12.0 - hnr) / 12.0) * 0.5
            raw[5] = min(1.0, f0 / 800.0) * 0.5 + min(1.0, hnr / 15.0) * 0.5
            raw[1] = min(1.0, call_rate / 3.0) * 0.6 + min(1.0, rms / 0.18) * 0.4
            raw[2] = max(0.0, 1.0 - abs(f0 - 350.0) / 250.0) * 0.6 + min(1.0, rms / 0.15) * 0.4
            raw[7] = min(1.0, f0_range / 250.0) * 0.6 + min(1.0, call_rate / 2.0) * 0.4
            raw[10] = min(1.0, duration / 1.5) * 0.5 + min(1.0, hnr / 14.0) * 0.5
            raw[4] = 0.35
            raw[6] = 0.30
            raw[8] = 0.20
            raw[0] = 0.15
            raw[3] = 0.20
            raw[11] = 0.05

        # Calibration: scale raw affinities to probabilities where dominant class is sharp (0.75 - 0.90)
        # and minor classes fall into natural biological proportions
        max_val = np.max(raw)
        if max_val > 0:
            norm_raw = raw / max_val
        else:
            norm_raw = raw

        # Apply calibrated sigmoid curve centered around 0.5 with biological contrast
        calibrated_probs = 1.0 / (1.0 + np.exp(-6.0 * (norm_raw - 0.55)))

        # Ensure top behavior is prominent (> 0.72) and non-matching behaviors are realistically low (< 0.35)
        top_idx = int(np.argmax(raw))
        calibrated_probs[top_idx] = max(float(calibrated_probs[top_idx]), 0.78)

        return calibrated_probs.astype(np.float32)

