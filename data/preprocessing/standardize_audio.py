"""
AnimalVox AI — Audio Preprocessing Pipeline
=============================================
Standardizes raw audio files to 16kHz mono WAV format,
segments into 5-second clips with 50% overlap, applies
noise gating, and normalizes amplitude.

This module handles Stage 1 of the AnimalVox AI pipeline.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Optional
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUDIO_CONFIG

logger = logging.getLogger(__name__)


def load_and_standardize(
    file_path: str,
    target_sr: int = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and standardize to target sample rate and mono.

    Args:
        file_path: Path to audio file (.wav, .mp3, .flac, .ogg, .m4a)
        target_sr: Target sample rate (default: 16000 Hz)
        mono: Convert to mono (default: True)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if target_sr is None:
        target_sr = AUDIO_CONFIG.sample_rate

    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=mono)
        logger.info(f"Loaded {file_path}: {len(audio)} samples @ {sr}Hz")
        return audio, sr
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to [-1, 1] range.

    Uses peak normalization to ensure consistent volume
    regardless of recording quality.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def remove_silence(
    audio: np.ndarray,
    top_db: int = None,
    sr: int = None
) -> np.ndarray:
    """
    Remove silent segments from audio using energy-based gating.

    Args:
        audio: Input audio array
        top_db: Threshold in dB below peak for silence detection
        sr: Sample rate

    Returns:
        Audio with silent segments removed
    """
    if top_db is None:
        top_db = AUDIO_CONFIG.top_db
    if sr is None:
        sr = AUDIO_CONFIG.sample_rate

    # Detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=top_db)

    if len(intervals) == 0:
        logger.warning("No non-silent segments found — returning original audio")
        return audio

    # Concatenate non-silent segments
    non_silent = np.concatenate([audio[start:end] for start, end in intervals])
    logger.debug(
        f"Silence removal: {len(audio)} -> {len(non_silent)} samples "
        f"({100 * len(non_silent) / len(audio):.1f}% retained)"
    )
    return non_silent


def segment_audio(
    audio: np.ndarray,
    clip_duration: float = None,
    overlap_ratio: float = None,
    sr: int = None
) -> List[np.ndarray]:
    """
    Segment audio into fixed-length clips with overlap.

    Args:
        audio: Input audio array (16kHz mono)
        clip_duration: Duration of each clip in seconds (default: 5.0)
        overlap_ratio: Overlap between clips (default: 0.5 = 50%)
        sr: Sample rate

    Returns:
        List of audio clips (numpy arrays)
    """
    if clip_duration is None:
        clip_duration = AUDIO_CONFIG.clip_duration
    if overlap_ratio is None:
        overlap_ratio = AUDIO_CONFIG.overlap_ratio
    if sr is None:
        sr = AUDIO_CONFIG.sample_rate

    clip_len = int(sr * clip_duration)
    hop_len = int(clip_len * (1 - overlap_ratio))

    clips = []
    for i in range(0, max(1, len(audio) - clip_len + 1), hop_len):
        clip = audio[i:i + clip_len]
        # Pad last clip if necessary
        if len(clip) < clip_len:
            clip = np.pad(clip, (0, clip_len - len(clip)), mode='constant')
        clips.append(clip)

    # Handle case where audio is shorter than one clip
    if len(clips) == 0 and len(audio) > 0:
        clip = np.pad(audio, (0, clip_len - len(audio)), mode='constant')
        clips.append(clip)

    logger.debug(f"Segmented into {len(clips)} clips of {clip_duration}s each")
    return clips


def preprocess_audio(
    file_path: str,
    target_sr: int = None,
    clip_duration: float = None,
    remove_silence_flag: bool = True
) -> List[np.ndarray]:
    """
    Full preprocessing pipeline: load → normalize → silence removal → segment.

    This is the main entry point for Stage 1 processing.

    Args:
        file_path: Path to raw audio file
        target_sr: Target sample rate (default: 16000)
        clip_duration: Clip duration in seconds (default: 5.0)
        remove_silence_flag: Whether to remove silent segments

    Returns:
        List of preprocessed audio clips (normalized, 16kHz mono)
    """
    if target_sr is None:
        target_sr = AUDIO_CONFIG.sample_rate
    if clip_duration is None:
        clip_duration = AUDIO_CONFIG.clip_duration

    # Step 1: Load and resample
    audio, sr = load_and_standardize(file_path, target_sr=target_sr)

    # Step 2: Normalize amplitude
    audio = normalize_amplitude(audio)

    # Step 3: Remove silence
    if remove_silence_flag:
        audio = remove_silence(audio, sr=sr)

    # Step 4: Segment into clips
    clips = segment_audio(audio, clip_duration=clip_duration, sr=sr)

    logger.info(
        f"Preprocessed {file_path}: {len(clips)} clips @ {sr}Hz, "
        f"{clip_duration}s each"
    )
    return clips


def preprocess_audio_bytes(
    audio_bytes: bytes,
    format: str = "wav",
    target_sr: int = None,
    clip_duration: float = None
) -> List[np.ndarray]:
    """
    Preprocess audio from raw bytes (for web/API input).

    Args:
        audio_bytes: Raw audio bytes
        format: Audio format string
        target_sr: Target sample rate
        clip_duration: Clip duration in seconds

    Returns:
        List of preprocessed audio clips
    """
    import io
    import tempfile
    import os

    if target_sr is None:
        target_sr = AUDIO_CONFIG.sample_rate

    # Write bytes to temp file for librosa to read
    suffix = f".{format}" if not format.startswith(".") else format
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        clips = preprocess_audio(
            temp_path,
            target_sr=target_sr,
            clip_duration=clip_duration
        )
    finally:
        os.unlink(temp_path)

    return clips
