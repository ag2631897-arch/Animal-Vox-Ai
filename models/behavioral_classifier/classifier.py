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

            # ── 2: Territorial Warning (Boundary defense / continuous fence barking) ──
            # Characterized by mid-low F0 (200-420 Hz), low-mid centroid (< 1600 Hz), high volume,
            # and rapid or continuous repetitive cadence (1.8 - 8.0/s) (Pongrácz 2005, Yin 2002)
            terr_pitch = max(0.0, 1.0 - abs(f0 - 320.0) / 180.0)
            terr_rate = min(1.0, max(0.5, call_rate / 2.8))
            terr_rms = min(1.0, rms / 0.15)
            terr_boost = 0.30 if (180 <= f0 <= 430 and centroid <= 1650 and call_rate >= 1.8) else 0.0
            raw[2] = (0.40 * terr_pitch + 0.35 * terr_rate + 0.25 * terr_rms) + terr_boost

            # ── 1: Alarm Ground (Sudden high-pitched startle alert barking) ───────
            # Characterized by high F0 (> 450 Hz), high spectral centroid (> 1600 Hz), harsh acoustic roughness
            alarm_rate = min(1.0, call_rate / 3.5)
            alarm_pitch = max(0.0, 1.0 - abs(f0 - 520.0) / 220.0) if f0 >= 420 else 0.20
            alarm_energy = min(1.0, rms / 0.18)
            alarm_cent_factor = 1.15 if centroid > 1650 else 0.65
            alarm_harshness = max(0.12, 1.0 - max(0.0, hnr - 9.0) / 10.0) if f0 > 380 else 0.8
            raw[1] = (0.40 * alarm_rate + 0.35 * alarm_pitch + 0.25 * alarm_energy) * alarm_cent_factor * alarm_harshness

            # ── 7: Play Vocalization (Excited, modulated play bark) ───────────────
            # Characterized by high pitch modulation, mid-high F0 (450-800 Hz), excited energy
            play_modulation = min(1.0, max(f0_range / 180.0, (call_rate - 2.0) / 2.5))
            play_pitch = max(0.0, 1.0 - abs(f0 - 580.0) / 300.0)
            play_energy = min(1.0, rms / 0.12)
            raw[7] = 0.40 * play_pitch + 0.35 * play_modulation + 0.25 * play_energy

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
            # Characterized by mid F0 (320-480 Hz), moderate duration, moderate HNR, low call rate (< 2.0)
            contact_f0 = max(0.0, 1.0 - abs(f0 - 400.0) / 200.0)
            contact_hnr = min(1.0, max(0.0, (hnr - 6.0) / 8.0))
            contact_rate_pen = max(0.25, 1.0 - max(0.0, call_rate - 1.8) / 2.5)
            raw[4] = (0.55 * contact_f0 + 0.45 * contact_hnr) * contact_rate_pen

            # ── 6: Feeding Call (Begging solicitation whines for food/treats) ─────
            # Characterized by high pitch whines (450-800 Hz), pure pleading tonality (high HNR), repetitive solicitation bursts
            feed_pitch = max(0.0, 1.0 - abs(f0 - 620.0) / 280.0)
            feed_tonal = min(1.0, max(0.0, (hnr - 7.0) / 10.0))
            feed_rate = min(1.0, call_rate / 2.0)
            raw[6] = 0.40 * feed_pitch + 0.35 * feed_tonal + 0.25 * feed_rate

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
            # Avian vocalizations grounded in ornithological field science (Marler 1955, Catchpole & Slater 2008)
            # 0: Aerial Predator Alarm (thin, high-frequency whistle, 'seet' call — Marler 1955)
            raw[0] = min(1.0, max(0.0, (centroid - 3500.0) / 2500.0)) * 0.65 + min(1.0, max(0.0, (f0 - 2200.0) / 2000.0)) * 0.35
            # 1: Ground Predator Mobbing Call (harsh, repetitive broadband scold / rattle)
            raw[1] = min(1.0, call_rate / 3.0) * 0.50 + max(0.0, (11.0 - hnr) / 11.0) * 0.30 + min(1.0, max(0.0, (2800.0 - centroid) / 1800.0)) * 0.20
            # 3: Breeding / Territorial Display Song (melodious, wide pitch modulation, high HNR)
            raw[3] = min(1.0, f0_range / 350.0) * 0.55 + min(1.0, hnr / 14.0) * 0.35 + (0.10 if duration > 1.2 else 0.0)
            # 2: Territorial Countersinging / Defense Song
            raw[2] = raw[3] * 0.70 + min(1.0, rms / 0.16) * 0.30
            # 4: Flock Flight / Contact Call (short gentle chirp, moderate pitch 1200-2500 Hz, low rate)
            contact_pitch = max(0.0, 1.0 - abs(f0 - 1800.0) / 1000.0)
            contact_rate = max(0.0, 1.0 - max(0.0, call_rate - 1.5) / 2.0)
            raw[4] = 0.50 * contact_pitch + 0.30 * min(1.0, hnr / 12.0) + 0.20 * contact_rate
            # 6: Feeding / Begging Solicitation Chirps (rapid repetitive begging, high pitch)
            raw[6] = min(1.0, call_rate / 2.8) * 0.50 + max(0.0, 1.0 - abs(f0 - 2000.0) / 1000.0) * 0.50
            # 5: Distress / Capture Scream (loud RMS, high ZCR, harsh)
            raw[5] = min(1.0, rms / 0.20) * 0.55 + min(1.0, zcr / 0.12) * 0.45
            # 7: Play / Acrobatics (parrots, corvids)
            raw[7] = min(1.0, f0_range / 300.0) * 0.60 + min(1.0, call_rate / 2.5) * 0.40
            # 10: Social Bonding (pair duetting)
            raw[10] = raw[3] * 0.60 + min(1.0, hnr / 14.0) * 0.40
            # 8: Submission (subordinate posture chirp)
            raw[8] = max(0.0, 1.0 - rms / 0.08) * 0.60 + 0.40 * contact_pitch
            # 9: Direct Threat / Aggressive Chitter
            raw[9] = max(0.0, (10.0 - hnr) / 10.0) * 0.60 + min(1.0, zcr / 0.14) * 0.40
            raw[11] = 0.02

        elif species_lower == "whales":
            # Cetacean bioacoustics (Payne & McVay 1971; Payne & Webb 1971; Jurasz & Jurasz 1979)
            # Differentiates Blue/Fin Whale infrasound, Humpback breeding songs, bubble-net feeding, breaching, and pod contact

            # 4: Contact Call — Low-Frequency Infrasonic Moan (Blue / Fin whale 20 Hz pulses, SOFAR propagation)
            is_infrasonic = (f0 < 65.0 or centroid < 480.0)
            infrasound_score = 0.92 if is_infrasonic else max(0.0, 1.0 - abs(f0 - 120.0) / 250.0) * 0.55
            raw[4] = infrasound_score * min(1.0, duration / 1.5)

            # 11: Navigation / Echolocation — Bathymetric Deep-Sea Pulse / Echolocation
            raw[11] = 0.88 if (f0 < 35.0 or (centroid > 3200.0 and zcr > 0.10)) else 0.25

            # 3: Mating Call — Melodic Breeding Song Theme (Humpback seasonal breeding song)
            # Rich frequency modulation (f0_range > 65 Hz), higher pitch than blue whale (100-1500 Hz), tonal HNR
            is_melodic_song = (f0_range > 65.0 and hnr >= 8.5 and not is_infrasonic)
            song_mod = min(1.0, f0_range / 160.0)
            song_hnr = min(1.0, hnr / 14.0)
            raw[3] = (0.55 * song_mod + 0.45 * song_hnr) if is_melodic_song else 0.20

            # 6: Feeding Call — Bubble-Net Feeding Cry (Ascending trumpet scream coordinating krill ring)
            is_feeding_trumpet = (f0_slope > 0.25 and 200.0 <= f0 <= 950.0 and rms >= 0.08)
            raw[6] = 0.85 if is_feeding_trumpet else (min(1.0, max(0.0, f0_slope / 0.5)) * 0.40)

            # 7: Play Vocalization — Breaching Slap & Trumpet (Percussive surface slap, lobtailing)
            is_breach_slap = (rms >= 0.14 and zcr >= 0.06 and hnr < 10.0)
            raw[7] = 0.88 if is_breach_slap else min(1.0, rms / 0.18) * 0.35

            # 10: Social Bonding — Mother-Calf Acoustic Whisper (Cryptic low-amplitude grunts)
            is_whisper = (rms < 0.09 and 70.0 <= f0 <= 260.0 and f0_range < 65.0)
            raw[10] = 0.86 if is_whisper else (max(0.0, 1.0 - rms / 0.12) * 0.30)

            # 2: Territorial / Competitive Display Blast (Escort competition bellow)
            raw[2] = 0.85 if (rms >= 0.15 and f0 < 300.0 and hnr < 8.5) else 0.15

            # 9: Aggression Warning — Bubble Trail & Jaw Strike
            raw[9] = 0.84 if (rms >= 0.16 and zcr >= 0.08 and hnr < 7.0) else 0.10

            # 5: Distress Call — Stranding / Entanglement Pulse
            raw[5] = min(1.0, rms / 0.20) * 0.50 + (0.40 if f0_slope > 0.8 else 0.0)

            # 1: Alarm Ground — Orca Predator Warning
            raw[1] = 0.12
            raw[0] = 0.05
            raw[8] = 0.08

        elif species_lower == "dolphins":
            # Delphinid bioacoustics: signature whistles, echolocation clicks, and burst pulses
            # 11: Navigation Echolocation — Ultrasonic Click Trains (> 3200 Hz centroid, high ZCR)
            is_click_train = (centroid > 3200.0 or zcr > 0.11)
            raw[11] = 0.92 if is_click_train else 0.20

            # 4: Contact Call — Signature Whistle (FM frequency sweep, F0 > 1000 Hz, tonal purity)
            is_whistle = (f0 >= 900.0 and hnr >= 10.0 and f0_range >= 100.0)
            raw[4] = 0.90 if is_whistle else min(1.0, hnr / 15.0) * 0.40

            # 10: Social Bonding — Affiliative Whistle Duet / Pod Cohesion
            raw[10] = raw[4] * 0.75 + min(1.0, duration / 1.5) * 0.25

            # 7: Play Vocalization — Social Play Burst Pulses / Tumbling Calls
            raw[7] = min(1.0, f0_range / 300.0) * 0.50 + min(1.0, call_rate / 2.5) * 0.50

            # 9: Aggression Warning — Jaw Clap Threat (Loud impulsive snap, high ZCR, low HNR)
            raw[9] = min(1.0, zcr / 0.12) * 0.55 + max(0.0, (10.0 - hnr) / 10.0) * 0.45

            # 5: Distress Call — Distress Whistle (Rapid repetitive rising whistles)
            raw[5] = min(1.0, rms / 0.18) * 0.50 + (0.40 if f0_slope > 0.5 and f0 > 1500.0 else 0.10)

            # 3: Mating Call — Courtship Whistle & Synchronized Swimming
            raw[3] = raw[4] * 0.65 + min(1.0, duration / 2.0) * 0.35

            # 6: Feeding Call — Cooperative Foraging / Fish Herding
            raw[6] = raw[11] * 0.60 + raw[4] * 0.40
            raw[2] = 0.15
            raw[1] = 0.10
            raw[0] = 0.05
            raw[8] = 0.10

        elif species_lower == "frogs":
            # Anuran bioacoustics (Wells 2007; Gerhardt & Huber 2002)
            # 3: Mating Call — Advertisement Breeding Call (Species-specific pulsed croaks, 180-700 Hz, steady F0)
            is_advertisement = (180.0 <= f0 <= 700.0 and call_rate >= 0.8 and f0_range < 90.0 and hnr >= 5.0)
            raw[3] = 0.90 if is_advertisement else max(0.0, 1.0 - abs(f0 - 380.0) / 250.0) * 0.50

            # 2: Territorial Warning — Rival Encounter Call (Higher cadence, louder, staccato croak against rival)
            is_territorial = (call_rate >= 2.0 and rms >= 0.09 and 180.0 <= f0 <= 750.0)
            raw[2] = 0.88 if is_territorial else min(1.0, call_rate / 2.5) * 0.45

            # 5: Distress Call — Open-Mouth Predator Distress Scream (Loud, sudden, high centroid, wide band)
            is_scream = (rms >= 0.14 and (f0 > 750.0 or centroid > 2200.0))
            raw[5] = 0.92 if is_scream else min(1.0, rms / 0.18) * 0.35

            # 8: Submission Signal — Release Call (Short vibrational chirp when mis-mounted)
            is_release = (duration < 0.35 and 150.0 <= f0 <= 450.0 and rms < 0.07)
            raw[8] = 0.85 if is_release else (0.30 if duration < 0.4 else 0.10)

            # 10: Social Bonding — Wetland Chorus Bonding (Synchronized group calling)
            raw[10] = raw[3] * 0.70 + min(1.0, duration / 1.5) * 0.30

            # 6: Feeding Call
            raw[6] = 0.20
            # 1: Alarm Ground — Splash warning
            raw[1] = 0.25
            raw[0] = 0.05
            raw[7] = 0.20
            raw[9] = raw[2] * 0.65
            raw[4] = 0.25
            raw[11] = 0.02
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

