"""
AnimalVox AI — Prosodic Feature Extractor
============================================
Extracts 10 prosodic features from animal audio clips:
pitch, rhythm, pace, tone, and harmonic structure.

These "musical qualities" of sound are critical for
distinguishing emotional states and urgency levels.

Part of Stage 2, Path C of the AnimalVox AI pipeline.
"""

import numpy as np
import librosa
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Feature names and descriptions for documentation
PROSODIC_FEATURE_NAMES = [
    "f0_mean",          # Fundamental frequency — base pitch
    "f0_std",           # Pitch variability — complex vs. simple signal
    "f0_range",         # Pitch range — wide = complex communication
    "voiced_ratio",     # Fraction with clear pitch — low = rough vocalization
    "rms_mean",         # Average volume — sudden loud = alarm
    "rms_std",          # Volume variation — high = urgent/emotional
    "zcr_mean",         # Zero crossing rate — roughness/noisiness
    "call_rate",        # Calls per second — fast = panic
    "hnr_mean",         # Harmonics-to-Noise Ratio — purity of tone
    "f0_slope",         # Pitch direction — rising = alert, falling = calm
]


def extract_prosodic_features(
    audio: np.ndarray,
    sr: int = 16000,
    use_parselmouth: bool = True
) -> np.ndarray:
    """
    Extract 10 prosodic features from an audio clip.

    Features capture the "musical qualities" of animal sounds:
    pitch, rhythm, amplitude, and tonal purity. These are
    biologically meaningful — rising pitch indicates alertness,
    rough tones indicate aggression, fast repetition indicates panic.

    Args:
        audio: Preprocessed audio clip (16kHz mono, normalized)
        sr: Sample rate
        use_parselmouth: Use Praat via parselmouth for HNR (more accurate)

    Returns:
        10-dimensional prosodic feature vector
    """
    features = {}

    # ── Fundamental Frequency (Pitch) ──────────────────────
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
            sr=sr
        )

        # Handle NaN values in f0
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        if len(f0_valid) == 0:
            f0_valid = np.array([0.0])

        features['f0_mean'] = float(np.mean(f0_valid))
        features['f0_std'] = float(np.std(f0_valid))
        features['f0_range'] = float(np.ptp(f0_valid))  # max - min

        # Voiced ratio
        if voiced_flag is not None:
            features['voiced_ratio'] = float(np.mean(voiced_flag))
        else:
            features['voiced_ratio'] = 0.0

        # F0 slope (direction of pitch change)
        if len(f0_valid) >= 2:
            # Linear regression slope of pitch contour
            x = np.arange(len(f0_valid))
            slope = np.polyfit(x, f0_valid, 1)[0]
            features['f0_slope'] = float(slope)
        else:
            features['f0_slope'] = 0.0

    except Exception as e:
        logger.warning(f"Pitch extraction failed: {e}")
        features['f0_mean'] = 0.0
        features['f0_std'] = 0.0
        features['f0_range'] = 0.0
        features['voiced_ratio'] = 0.0
        features['f0_slope'] = 0.0

    # ── Amplitude (RMS Energy) ──────────────────────────────
    rms = librosa.feature.rms(y=audio)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # ── Zero Crossing Rate (Roughness) ──────────────────────
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = float(np.mean(zcr))

    # ── Call Rate (Onsets Per Second) ────────────────────────
    try:
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        duration = len(audio) / sr
        features['call_rate'] = float(len(onset_frames) / max(duration, 0.01))
    except Exception:
        features['call_rate'] = 0.0

    # ── Harmonics-to-Noise Ratio ────────────────────────────
    if use_parselmouth:
        try:
            import parselmouth
            snd = parselmouth.Sound(audio, sr)
            hnr = snd.to_harmonicity()
            hnr_values = hnr.values[hnr.values != -200]
            features['hnr_mean'] = float(
                np.mean(hnr_values) if len(hnr_values) > 0 else 0.0
            )
        except ImportError:
            logger.warning("parselmouth not available — using spectral flatness for HNR proxy")
            sf = librosa.feature.spectral_flatness(y=audio)[0]
            features['hnr_mean'] = float(-10 * np.log10(np.mean(sf) + 1e-8))
        except Exception as e:
            logger.warning(f"HNR extraction failed: {e}")
            features['hnr_mean'] = 0.0
    else:
        sf = librosa.feature.spectral_flatness(y=audio)[0]
        features['hnr_mean'] = float(-10 * np.log10(np.mean(sf) + 1e-8))

    # ── Assemble Feature Vector ─────────────────────────────
    # Order matches PROSODIC_FEATURE_NAMES
    feature_vector = np.array([
        features['f0_mean'],
        features['f0_std'],
        features['f0_range'],
        features['voiced_ratio'],
        features['rms_mean'],
        features['rms_std'],
        features['zcr_mean'],
        features['call_rate'],
        features['hnr_mean'],
        features['f0_slope'],
    ], dtype=np.float32)

    return feature_vector


def extract_prosodic_features_batch(
    clips: list,
    sr: int = 16000
) -> np.ndarray:
    """
    Extract prosodic features for a batch of audio clips.

    Args:
        clips: List of audio clip arrays
        sr: Sample rate

    Returns:
        Feature matrix of shape (N, 10)
    """
    features = []
    for clip in clips:
        feat = extract_prosodic_features(clip, sr=sr)
        features.append(feat)
    return np.stack(features, axis=0)


def get_feature_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions of each prosodic feature."""
    return {
        "f0_mean": "Base pitch — rising = alertness, falling = submission",
        "f0_std": "Pitch variability — high = complex communication",
        "f0_range": "Pitch range — wide = complex, narrow = simple signal",
        "voiced_ratio": "Fraction of clear tonal sound — low = rough/noisy",
        "rms_mean": "Average volume — sudden loud = alarm, gradual = contact",
        "rms_std": "Volume variation — high = urgent/emotional",
        "zcr_mean": "Roughness — high = growl/aggression, low = pure tone",
        "call_rate": "Calls per second — fast = high urgency/panic",
        "hnr_mean": "Tonal purity — low HNR = roughness = aggression/distress",
        "f0_slope": "Pitch direction — rising = question/alert, falling = end",
    }
