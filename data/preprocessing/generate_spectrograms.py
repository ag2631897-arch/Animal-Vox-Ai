"""
AnimalVox AI — Mel-Spectrogram Generator
==========================================
Converts preprocessed audio clips into mel-spectrograms
for visual frequency analysis via EfficientNet-B3 CNN.

Part of Stage 2, Path B of the AnimalVox AI pipeline.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Tuple
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUDIO_CONFIG

logger = logging.getLogger(__name__)


def generate_mel_spectrogram(
    audio: np.ndarray,
    sr: int = None,
    n_mels: int = None,
    n_fft: int = None,
    hop_length: int = None,
    fmin: int = None,
    fmax: int = None,
    to_db: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a mel-spectrogram from an audio clip.

    A mel-spectrogram is a visual representation of sound:
    - Horizontal axis: time
    - Vertical axis: frequency (mel-scaled to match human perception)
    - Brightness/color: intensity (amplitude)

    Different animal behaviors produce distinctly different spectrogram
    shapes — alarm calls look different from mating songs.

    Args:
        audio: Preprocessed audio array (16kHz mono, normalized)
        sr: Sample rate (default: 16000)
        n_mels: Number of mel bands (default: 128)
        n_fft: FFT window size in samples (default: 400 = 25ms @ 16kHz)
        hop_length: Hop size in samples (default: 160 = 10ms @ 16kHz)
        fmin: Minimum frequency in Hz (default: 50)
        fmax: Maximum frequency in Hz (default: 8000)
        to_db: Convert power to dB scale (default: True)
        normalize: Normalize to [0, 1] range (default: True)

    Returns:
        Mel-spectrogram array of shape (n_mels, T) where T = time frames
    """
    if sr is None:
        sr = AUDIO_CONFIG.sample_rate
    if n_mels is None:
        n_mels = AUDIO_CONFIG.n_mels
    if n_fft is None:
        n_fft = AUDIO_CONFIG.n_fft
    if hop_length is None:
        hop_length = AUDIO_CONFIG.hop_length
    if fmin is None:
        fmin = AUDIO_CONFIG.fmin
    if fmax is None:
        fmax = AUDIO_CONFIG.fmax

    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )

    if to_db:
        # Convert to log scale (dB) — better for visual analysis
        mel = librosa.power_to_db(mel, ref=np.max)

    if normalize:
        # Normalize to [0, 1] for CNN input
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max - mel_min > 1e-8:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

    return mel  # Shape: (n_mels, T)


def mel_to_image(
    mel: np.ndarray,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Convert mel-spectrogram to 3-channel image for EfficientNet-B3 input.

    EfficientNet expects RGB images of size (224, 224, 3).
    We replicate the single-channel spectrogram across 3 channels.

    Args:
        mel: Mel-spectrogram of shape (n_mels, T)
        target_size: Target image size (H, W) for the CNN

    Returns:
        Image array of shape (3, H, W) normalized to [0, 1]
    """
    from PIL import Image

    # Resize to target dimensions
    # mel shape is (n_mels, T) — treat as grayscale image
    img = Image.fromarray((mel * 255).astype(np.uint8), mode='L')
    img = img.resize(target_size, Image.Resampling.BILINEAR)

    # Convert back to numpy and create 3-channel
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_3ch = np.stack([img_array] * 3, axis=0)  # (3, H, W)

    return img_3ch


def batch_generate_spectrograms(
    clips: list,
    sr: int = None,
    as_images: bool = True,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Generate mel-spectrograms for a batch of audio clips.

    Args:
        clips: List of audio clip arrays
        sr: Sample rate
        as_images: If True, convert to 3-channel images for CNN
        target_size: Target image size if as_images=True

    Returns:
        Batch array of shape (N, n_mels, T) or (N, 3, H, W)
    """
    spectrograms = []
    for clip in clips:
        mel = generate_mel_spectrogram(clip, sr=sr)
        if as_images:
            mel = mel_to_image(mel, target_size=target_size)
        spectrograms.append(mel)

    return np.stack(spectrograms, axis=0)
