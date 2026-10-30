"""
AnimalVox AI — Animal Detection & Auto-Routing (SRS Module 05)
================================================================
Automatically detects what animal is present from ambient audio.
Removes the need for manual species selection.

Pipeline:
1. Human Voice Exclusion (VAD-based frame suppression)
2. Soundscape Classification (indoor/outdoor/forest/marine)
3. Animal Presence Detection (fast binary CNN)
4. Species Group Identification (bird/dog/cat/marine/frog)
5. Source Count Estimation (how many individuals)
6. Auto-routing to the correct translation pipeline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from the animal detection pipeline."""
    has_animal: bool
    soundscape: str                    # indoor, outdoor_garden, etc.
    species_detections: List[Dict]     # [{species, confidence, ...}]
    num_individuals: int
    human_speech_removed: bool
    processing_time_ms: float
    # Auto-routing is deliberately marked as tentative unless the leading
    # species is sufficiently separated from the next candidate.  A species
    # label must never be mistaken for ground truth.
    routing_confident: bool = False
    detector_mode: str = "heuristic"


class VADFilter:
    """
    Voice Activity Detection filter using energy-based analysis.
    Suppresses human speech frames from the audio before animal analysis.
    
    Uses energy band analysis: human speech concentrates energy in
    300-3400 Hz with characteristic spectral envelope. Animal calls
    have different spectral distributions.
    """

    def __init__(self, sr: int = 16000, frame_duration_ms: int = 30,
                 aggressiveness: int = 3):
        self.sr = sr
        self.frame_size = int(sr * frame_duration_ms / 1000)
        self.aggressiveness = aggressiveness

    def remove_human_speech(self, audio: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Human voice exclusion filter.
        Primary human speech & imitation rejection is handled authoritatively by
        Stage 0 (AntiSpoofingGatekeeper) with trained ECAPA-TDNN / LPC physics.
        Preserves audio to avoid zeroing out real animal vocalization bursts.
        """
        return audio, False

    def _is_speech_frame(self, frame: np.ndarray) -> bool:
        """
        Detect if a frame contains human speech using spectral analysis.
        
        Human speech characteristics:
        - Spectral centroid typically 300-3400 Hz
        - Strong energy in F1-F2 bands (300-2500 Hz)
        - Harmonic structure with regular spacing
        """
        if np.max(np.abs(frame)) < 0.01:  # Silence
            return False

        # Compute FFT
        fft = np.fft.rfft(frame)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(frame), 1.0 / self.sr)

        # Energy in speech band (300-3400 Hz) vs total
        speech_mask = (freqs >= 300) & (freqs <= 3400)
        total_energy = np.sum(magnitude ** 2)
        speech_energy = np.sum(magnitude[speech_mask] ** 2)

        if total_energy < 1e-10:
            return False

        speech_ratio = speech_energy / total_energy

        # Spectral centroid
        centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)

        # Human speech: high speech_ratio + centroid in speech band
        is_speech = (speech_ratio > 0.6 and 200 < centroid < 4000)

        return is_speech


class SoundscapeClassifier(nn.Module):
    """
    Classifies the recording environment from audio characteristics.
    
    Categories: indoor, outdoor_garden, urban_park, forest_wilderness, marine
    
    Uses mel-spectrogram features to identify background noise signatures:
    - Indoor: low background, room reverb
    - Garden: birdsong backdrop, moderate noise
    - Urban: traffic, sirens, human activity
    - Forest: wind, dense wildlife, crickets
    - Marine: wave noise, underwater acoustics
    """

    def __init__(self, n_mels: int = 128, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class AnimalPresenceDetector(nn.Module):
    """
    Fast binary classifier: is any animal vocalizing in this audio?
    
    Lightweight CNN designed to be < 1.5 MB and respond in < 150ms.
    Acts as a fast gate before running heavier species classifiers.
    """

    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Binary: animal present or not
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(self.features(x)))


try:
    from models.dual_stream_classifier import EnhancedClassifier
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False


class SpeciesGroupClassifier(nn.Module):
    """
    Classifies audio into species group: bird, dog, cat, dolphin, whale, frog.
    Uses EnhancedClassifier residual architecture matching animal_detector.pt.
    """

    def __init__(self, input_dim: int = 256, num_species: int = 6):
        super().__init__()
        self.input_dim = input_dim
        if _ENHANCED_AVAILABLE:
            self.classifier = EnhancedClassifier(
                input_dim=input_dim,
                hidden_dim=160,
                num_classes=num_species,
                num_residual_blocks=2,
                dropout=0.15,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_species)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.input_dim:
            x = x[..., :self.input_dim]
        return self.classifier(x)


class SourceCountEstimator(nn.Module):
    """
    Estimates the number of individual animals vocalizing simultaneously.
    
    Input: Mel-spectrogram
    Output: Count prediction (1, 2, 3, 4+)
    """

    def __init__(self, n_mels: int = 128, max_sources: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_sources)  # Classes: 1, 2, 3, 4+
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class AnimalDetectionPipeline:
    """
    Complete animal detection and auto-routing pipeline (Module 05).
    
    Orchestrates: VAD → Soundscape → Presence → Species ID → Count → Route
    """

    def __init__(self, device: str = "cpu"):
        from config import AUDIO_CONFIG, ANIMAL_DETECTION_CONFIG, SOUNDSCAPE_CATEGORIES
        self.audio_config = AUDIO_CONFIG
        self.det_config = ANIMAL_DETECTION_CONFIG
        self.device = device
        self.soundscape_labels = SOUNDSCAPE_CATEGORIES

        # Initialize components
        self.vad = VADFilter(sr=self.audio_config.sample_rate,
                             aggressiveness=self.det_config.vad_aggressiveness)

        self.soundscape_model = SoundscapeClassifier(
            n_mels=self.audio_config.n_mels,
            num_classes=self.det_config.num_soundscapes
        ).to(device).eval()

        self.presence_model = AnimalPresenceDetector(
            n_mels=self.audio_config.n_mels
        ).to(device).eval()

        self.species_model = SpeciesGroupClassifier(
            input_dim=256,
            num_species=self.det_config.num_species_groups
        ).to(device).eval()

        self.count_model = SourceCountEstimator(
            n_mels=self.audio_config.n_mels,
            max_sources=self.det_config.max_sources
        ).to(device).eval()

        # The checkpoint currently contains weights for *species_model* only.
        # Do not use this flag to run the independently initialised soundscape,
        # presence, or source-count heads: doing so creates convincing-looking
        # but random labels such as "forest wilderness" for a home recording.
        self.has_trained_species_weights = False
        self.has_trained_soundscape_weights = False
        self.has_trained_count_weights = False
        # Kept as a compatibility alias for the existing status UI.
        self.has_trained_weights = False
        self.metadata = {}
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_species_weights, self.metadata = load_checkpoint(
                self.species_model, "animal_detector", device=device
            )
            self.has_trained_weights = self.has_trained_species_weights
        except Exception as e:
            logger.debug(f"Weight manager check for animal_detector: {e}")

        if self.has_trained_species_weights:
            logger.info(
                "AnimalDetectionPipeline loaded trained species-routing weights "
                f"(Epoch {self.metadata.get('epoch', 'N/A')}); "
                "soundscape and source count remain unavailable."
            )
        else:
            logger.info(
                "AnimalDetectionPipeline: No trained checkpoint found. "
                "Active Mode: SPECTRAL ACOUSTIC HEURISTIC FALLBACK (Pitch/Centroid/VAD Rules)."
            )

    def save_weights(self, epoch: int = 0, optimizer=None, metrics: Dict = None) -> Path:
        """Save trained species classifier weights to standard checkpoint path."""
        from models.weight_manager import save_checkpoint
        return save_checkpoint(
            self.species_model, "animal_detector", epoch=epoch,
            optimizer=optimizer, metrics=metrics
        )

    def load_weights(self, path: Optional[Path] = None) -> bool:
        """Load weights from standard registry or custom path."""
        from models.weight_manager import load_checkpoint
        if path:
            state = torch.load(path, map_location=self.device)
            state_dict = state["state_dict"] if "state_dict" in state else state
            self.species_model.load_state_dict(state_dict, strict=False)
            self.has_trained_species_weights = True
            self.has_trained_weights = True
            return True
        self.has_trained_species_weights, self.metadata = load_checkpoint(
            self.species_model, "animal_detector", device=self.device
        )
        self.has_trained_weights = self.has_trained_species_weights
        return self.has_trained_species_weights

    def _compute_heuristic_species_probs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract acoustic bio-markers to classify species without neural weights.
        Analyzes pitch (F0), spectral centroid, and zero-crossing rate.
        Order: ['birds', 'dogs', 'cats', 'dolphins', 'whales', 'frogs']
        """
        if len(audio) == 0:
            return np.ones(6) / 6.0

        # Select active segment (peak RMS 1-second window) to avoid analyzing silent pre-roll
        if len(audio) > sr:
            window_len = sr
            hop = sr // 4
            best_start = 0
            best_rms = -1.0
            for start_idx in range(0, len(audio) - window_len + 1, hop):
                chunk = audio[start_idx : start_idx + window_len]
                chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))
                if chunk_rms > best_rms:
                    best_rms = chunk_rms
                    best_start = start_idx
            analysis_audio = audio[best_start : best_start + window_len]
        else:
            analysis_audio = audio

        # RMS energy check
        rms = float(np.sqrt(np.mean(analysis_audio ** 2))) if len(analysis_audio) > 0 else 0.0
        # Spectral centroid
        try:
            cent = librosa.feature.spectral_centroid(y=analysis_audio, sr=sr)
            mean_cent = float(np.mean(cent)) if cent.size > 0 else 1500.0
        except Exception:
            mean_cent = 1500.0

        # Pitch estimation using autocorrelation
        try:
            autocorr = np.correlate(analysis_audio, analysis_audio, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            # Look for peak in animal vocal range (50 Hz to 4000 Hz)
            min_lag = int(sr / 4000)
            max_lag = int(sr / 50)
            if max_lag < len(autocorr):
                search_window = autocorr[min_lag:max_lag]
                peak_idx = int(np.argmax(search_window))
                if peak_idx == 0 or search_window[peak_idx] < 0.15 * autocorr[0]:
                    est_f0 = 250.0  # Unvoiced / noisy bark or growl
                else:
                    peak_lag = min_lag + peak_idx
                    est_f0 = float(sr / peak_lag) if peak_lag > 0 else 300.0
            else:
                est_f0 = 300.0
        except Exception:
            est_f0 = 300.0

        scores = np.zeros(6)
        # Birds: High pitch (> 1100 Hz) AND high spectral centroid (> 2200 Hz), or very high pitch (> 1400 Hz)
        if (est_f0 > 1100 and mean_cent > 2200) or est_f0 > 1400:
            scores[0] = 3.6
        # Dogs: Low to mid pitch (80 - 750 Hz), moderate centroid (600 - 2800 Hz)
        if 80 <= est_f0 <= 750 and mean_cent < 2800:
            scores[1] = 3.8
        # Cats: Mid to high pitch (350 - 1100 Hz) or very low purr (< 100 Hz)
        if (350 <= est_f0 <= 1100) or (est_f0 < 90 and rms > 0.02):
            scores[2] = 3.0
        # Dolphins: High frequency ultrasonic / clicks (> 3500 Hz)
        if mean_cent > 3800:
            scores[3] = 2.0
        # Whales: Infrasonic or low frequency moans (< 150 Hz) with long sustained envelope
        if est_f0 < 150 and len(audio) / sr > 2.0:
            scores[4] = 2.0
        # Frogs: Croak pulses, mid-range (200 - 700 Hz)
        if 200 <= est_f0 <= 700 and mean_cent < 1800:
            scores[5] = 2.0

        # Ensure at least a default dog/cat distribution if inconclusive
        if np.max(scores) == 0:
            scores[1] = 1.5  # Dog default
            scores[2] = 1.2  # Cat default

        exp = np.exp(scores)
        return exp / np.sum(exp)

    def detect(self, audio: np.ndarray, audio_embed: torch.Tensor = None) -> DetectionResult:
        """
        Run full detection pipeline on audio.
        Uses trained neural networks when weights exist, otherwise uses robust acoustic signal heuristics.
        """
        import time
        start = time.time()
        sr = self.audio_config.sample_rate

        # Step 1: Human voice exclusion
        filtered_audio, had_speech = self.vad.remove_human_speech(audio)

        # Step 2: Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=filtered_audio, sr=sr,
            n_mels=self.audio_config.n_mels,
            n_fft=self.audio_config.n_fft,
            hop_length=self.audio_config.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.from_numpy(mel_db).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Step 3: Soundscape classification.  Its weights are not packaged in
        # the species checkpoint, so report it as unavailable instead of using
        # random, untrained neural outputs.
        if self.has_trained_soundscape_weights:
            try:
                with torch.no_grad():
                    soundscape_logits = self.soundscape_model(mel_tensor)
                    soundscape_probs = torch.softmax(soundscape_logits, dim=-1).cpu().numpy()[0]
                soundscape_idx = int(np.argmax(soundscape_probs))
                soundscape = self.soundscape_labels[soundscape_idx]
            except Exception:
                soundscape = "not_assessed"
        else:
            soundscape = "not_assessed"

        # Step 4: Animal presence detection (VAD + RMS energy check)
        rms = float(np.sqrt(np.mean(filtered_audio ** 2))) if len(filtered_audio) > 0 else 0.0
        has_animal = rms > 0.003  # Audible vocalization exists

        if not has_animal:
            elapsed = (time.time() - start) * 1000
            return DetectionResult(
                has_animal=False, soundscape=soundscape,
                species_detections=[], num_individuals=0,
                human_speech_removed=had_speech, processing_time_ms=elapsed,
                routing_confident=False,
                detector_mode="species_model" if self.has_trained_species_weights else "heuristic"
            )

        # Step 5: Species group classification
        from config import SPECIES_GROUPS
        species_list = list(SPECIES_GROUPS.keys())

        if self.has_trained_species_weights:
            try:
                # Use provided embedding if compatible (256-dim), otherwise extract standard mel embedding
                if audio_embed is not None and isinstance(audio_embed, (torch.Tensor, np.ndarray)):
                    embed = audio_embed if isinstance(audio_embed, torch.Tensor) else torch.tensor(audio_embed)
                    if embed.dim() == 1:
                        embed = embed.unsqueeze(0)
                    if embed.shape[-1] > 256:
                        embed = embed[..., :256]
                    embed = embed.to(self.device).float()
                else:
                    from training.feature_extractor import extract_mel_embedding
                    mel_feat = extract_mel_embedding(filtered_audio, sr=sr, embed_dim=256)
                    embed = torch.from_numpy(mel_feat).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    species_logits = self.species_model(embed)
                    neural_probs = torch.softmax(species_logits, dim=-1).cpu().numpy()[0]

                # Blend neural prediction with acoustic bio-marker heuristic for maximum robustness.
                # Note: Neural model (Epoch 12) was trained on limited data, so heuristics
                # (physically-grounded pitch/centroid analysis) get higher weight until
                # the model is retrained on a larger real-world dataset.
                heuristic_probs = self._compute_heuristic_species_probs(filtered_audio, sr)

                # Confidence gate: if neural model is uncertain (top class <40%),
                # fall back entirely to acoustic heuristics which are physics-based.
                neural_top_conf = float(np.max(neural_probs))
                if neural_top_conf < 0.40:
                    logger.debug(f"Neural species confidence low ({neural_top_conf:.2f}); using pure heuristic.")
                    species_probs = heuristic_probs
                else:
                    # Heuristic-dominant blend: 70% acoustic physics + 30% neural
                    species_probs = 0.30 * neural_probs + 0.70 * heuristic_probs
                species_probs = species_probs / np.sum(species_probs)
            except Exception as e:
                logger.warning(f"Neural species forward pass error ({e}); using heuristic fallback.")
                species_probs = self._compute_heuristic_species_probs(filtered_audio, sr)
        else:
            # Pure Acoustic Bio-Marker Heuristics
            species_probs = self._compute_heuristic_species_probs(filtered_audio, sr)

        # Step 6: Source count estimation.  As with soundscape, only run this
        # head if it has its own trained checkpoint.
        if self.has_trained_count_weights:
            try:
                with torch.no_grad():
                    count_logits = self.count_model(mel_tensor)
                    count_probs = torch.softmax(count_logits, dim=-1).cpu().numpy()[0]
                num_individuals = int(np.argmax(count_probs)) + 1
            except Exception:
                num_individuals = 1
        else:
            num_individuals = 1

        # Build species detections
        detections = []
        for i, sp in enumerate(species_list):
            if i < len(species_probs) and species_probs[i] > 0.15:
                sp_info = SPECIES_GROUPS.get(sp, {"display_name": sp.capitalize(), "emoji": "🐾"})
                detections.append({
                    "species": sp,
                    "display_name": sp_info["display_name"],
                    "emoji": sp_info["emoji"],
                    "confidence": float(species_probs[i]),
                })
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        top_confidence = detections[0]["confidence"] if detections else 0.0
        runner_up_confidence = detections[1]["confidence"] if len(detections) > 1 else 0.0
        routing_confident = (
            top_confidence >= 0.70 and
            (top_confidence - runner_up_confidence) >= 0.15
        )

        elapsed = (time.time() - start) * 1000

        return DetectionResult(
            has_animal=True, soundscape=soundscape,
            species_detections=detections,
            num_individuals=num_individuals,
            human_speech_removed=had_speech,
            processing_time_ms=elapsed,
            routing_confident=routing_confident,
            detector_mode="species_model+acoustic_heuristic" if self.has_trained_species_weights else "acoustic_heuristic"
        )
