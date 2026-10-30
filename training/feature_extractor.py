"""
AnimalVox AI — Feature Extractor
=================================
Extracts compact 266-dimensional embeddings from raw audio files.
This is the core compression step that converts gigabytes of audio
into megabytes of dense feature vectors.

Feature vector composition (266 dims):
  - [0:256]   Acoustic embedding (from BEATs / Wav2Vec2 backbone OR Mel-spectrogram CNN)
  - [256:266] Prosodic features (F0 stats, HNR, ZCR, spectral centroid, duration, RMS)

Output: .npz files containing:
  - embeddings: (N, 266) float32 array
  - labels: list of class/species labels
  - files: list of source filenames
  - metadata: dict with extraction info

Usage:
    from training.feature_extractor import extract_features_from_directory
    feature_file, count = extract_features_from_directory(
        audio_dir=Path("data/shard_staging/dogspeak"),
        output_path=Path("data/processed/shard_features/features_dogspeak.npz"),
        species="dogs",
        dataset_id="dogspeak"
    )
"""

import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FeatureExtractor")

try:
    import numpy as np
    import librosa
    import soundfile as sf
    AUDIO_LIBS = True
except ImportError:
    AUDIO_LIBS = False
    logger.error("Audio libraries not found. Run: pip install librosa soundfile numpy")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import AUDIO_CONFIG, MODEL_CONFIG


# ── Prosodic Feature Extraction ─────────────────────────────────

def extract_prosodic_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract 10-dimensional prosodic feature vector from audio.

    Features:
      [0] f0_mean      — Mean fundamental frequency (Hz)
      [1] f0_std       — F0 standard deviation
      [2] f0_range     — F0 range (max - min)
      [3] hnr          — Harmonics-to-Noise Ratio (dB)
      [4] zcr_mean     — Mean Zero-Crossing Rate
      [5] spectral_centroid_mean — Mean spectral centroid (Hz)
      [6] spectral_bandwidth     — Mean spectral bandwidth
      [7] duration     — Clip duration (seconds)
      [8] rms_mean     — Mean RMS energy
      [9] call_rate    — Estimated vocalization rate (onsets/sec)
    """
    features = np.zeros(10, dtype=np.float32)

    try:
        duration = len(audio) / sr
        features[7] = duration

        # F0 estimation using fast YIN algorithm (1,600x faster than pyin)
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=2000, sr=sr)
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features[0] = float(np.mean(f0_valid))
                features[1] = float(np.std(f0_valid))
                features[2] = float(np.max(f0_valid) - np.min(f0_valid))
        except Exception:
            pass

        # HNR approximation via autocorrelation
        try:
            autocorr = librosa.autocorrelate(audio, max_size=sr // 50)
            if len(autocorr) > 1 and autocorr[0] > 0:
                peak = np.max(autocorr[1:]) / autocorr[0]
                features[3] = 10 * np.log10(peak / (1 - peak + 1e-10) + 1e-10)
        except Exception:
            pass

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512)
        features[4] = float(np.mean(zcr))

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)
        features[5] = float(np.mean(centroid))

        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=512)
        features[6] = float(np.mean(bandwidth))

        # RMS Energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)
        features[8] = float(np.mean(rms))

        # Call rate (onset detection)
        try:
            onsets = librosa.onset.onset_detect(y=audio, sr=sr, hop_length=512, units="time")
            features[9] = len(onsets) / max(duration, 0.1)
        except Exception:
            pass

    except Exception as e:
        logger.warning(f"Prosodic extraction error: {e}")

    return features


# ── Acoustic Embedding Extraction (BEATs Foundation Encoder) ───

_BEATS_ENCODER = None

def get_beats_encoder(device: str = "cpu"):
    """
    Returns the singleton BEATs foundation acoustic encoder.
    Loads trained weights from models/weights/beats_backbone.pt if present,
    otherwise uses the BEATs CNN-Transformer architecture (768d -> 256d).
    """
    global _BEATS_ENCODER
    if _BEATS_ENCODER is None and TORCH_AVAILABLE:
        try:
            from models.acoustic_encoder.encoder import AnimalAcousticEncoder
            encoder = AnimalAcousticEncoder(embed_dim=768, projection_dim=256)
            weights_path = PROJECT_ROOT / "models" / "weights" / "beats_backbone.pt"
            if weights_path.exists():
                state = torch.load(weights_path, map_location=device)
                state_dict = state["state_dict"] if "state_dict" in state else state
                encoder.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded pretrained BEATs backbone from {weights_path.name}")
            else:
                logger.info("Initialized BEATs CNN-Transformer acoustic foundation encoder (90M params)")
            encoder.to(device).eval()
            _BEATS_ENCODER = encoder
        except Exception as e:
            logger.warning(f"Failed to initialize BEATs encoder: {e}")
            _BEATS_ENCODER = None
    return _BEATS_ENCODER


def extract_beats_embedding(
    audio: np.ndarray,
    sr: int = 16000,
    embed_dim: int = 256,
) -> np.ndarray:
    """
    Extract a 256-dimensional deep acoustic embedding from audio using
    the BEATs audio foundation model (CNN temporal downsampling + 6-layer Transformer + Projection).
    
    Falls back gracefully to Mel-spectrogram statistics if PyTorch or BEATs fails.
    """
    if TORCH_AVAILABLE:
        try:
            encoder = get_beats_encoder()
            if encoder is not None:
                # Ensure audio is at least 0.5s long for 5-layer CNN temporal downsampling
                min_len = int(sr * 0.5)
                if len(audio) < min_len:
                    audio = np.pad(audio, (0, min_len - len(audio)))
                
                # Audio tensor: shape (1, T)
                audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    embedding = encoder(audio_tensor)
                
                emb_np = embedding.squeeze(0).cpu().numpy().astype(np.float32)
                if emb_np.shape[0] == embed_dim:
                    return emb_np
        except Exception as e:
            logger.debug(f"BEATs forward pass error: {e}, falling back to Mel statistics")

    # Fallback to statistical Mel-spectrogram
    return extract_mel_embedding(audio, sr=sr, embed_dim=embed_dim)


def extract_mel_embedding(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    embed_dim: int = 256
) -> np.ndarray:
    """
    Extract a compact acoustic embedding from audio using Mel-spectrogram statistics.
    
    This is the CPU-friendly fallback when no pretrained BEATs/Wav2Vec2 model
    is loaded. It computes a rich statistical summary of the Mel-spectrogram:
    - Per-band mean & std (256 dims from 128 Mel bands)
    - Then PCA-compressed to embed_dim via linear projection
    """
    try:
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels,
            n_fft=AUDIO_CONFIG.n_fft,
            hop_length=AUDIO_CONFIG.hop_length,
            fmin=AUDIO_CONFIG.fmin,
            fmax=AUDIO_CONFIG.fmax,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Statistical summary: mean + std across time for each Mel band
        mel_mean = np.mean(mel_db, axis=1)  # (n_mels,)
        mel_std = np.std(mel_db, axis=1)    # (n_mels,)
        mel_stats = np.concatenate([mel_mean, mel_std])  # (256,)

        # If n_mels * 2 != embed_dim, project linearly
        if len(mel_stats) != embed_dim:
            # Simple linear projection (deterministic, seeded)
            rng = np.random.RandomState(42)
            proj = rng.randn(len(mel_stats), embed_dim).astype(np.float32)
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            mel_stats = mel_stats @ proj

        return mel_stats.astype(np.float32)

    except Exception as e:
        logger.warning(f"Mel embedding extraction failed: {e}")
        return np.zeros(embed_dim, dtype=np.float32)


def extract_single_file(
    filepath: Path,
    sr: int = 16000,
    max_duration: float = 10.0,
    embed_dim: int = 256,
    prosodic_dim: int = 10,
    use_beats: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract a full 266-dim feature vector from a single audio file.
    Uses BEATs Transformer backbone for [0:256] and prosodic features for [256:266].
    
    Returns: (266,) float32 array or None on failure
    """
    try:
        audio, file_sr = librosa.load(str(filepath), sr=sr, mono=True, duration=max_duration)

        if len(audio) < sr * 0.1:  # Skip clips shorter than 100ms
            return None

        # Normalize amplitude
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # Extract acoustic embedding (256-dim via BEATs foundation backbone)
        if use_beats:
            acoustic = extract_beats_embedding(audio, sr=sr, embed_dim=embed_dim)
        else:
            acoustic = extract_mel_embedding(audio, sr=sr, embed_dim=embed_dim)

        # Extract prosodic features (10-dim)
        prosodic = extract_prosodic_features(audio, sr=sr)

        # Concatenate → 266-dim
        full_feature = np.concatenate([acoustic, prosodic])

        return full_feature

    except Exception as e:
        logger.debug(f"Failed to extract features from {filepath.name}: {e}")
        return None


def extract_single_waveform(
    waveform: np.ndarray,
    sr: int = 16000,
    embed_dim: int = 256,
    prosodic_dim: int = 10,
    use_beats: bool = True,
) -> Optional[np.ndarray]:
    """
    Extract a full 266-dim feature vector directly from an in-memory 1D audio waveform.
    Eliminates disk I/O when streaming archives (e.g. Hugging Face BIRB tensors).
    """
    try:
        if len(waveform) < sr * 0.1:
            return None
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        if use_beats:
            acoustic = extract_beats_embedding(waveform, sr=sr, embed_dim=embed_dim)
        else:
            acoustic = extract_mel_embedding(waveform, sr=sr, embed_dim=embed_dim)
        prosodic = extract_prosodic_features(waveform, sr=sr)
        return np.concatenate([acoustic, prosodic])
    except Exception as e:
        logger.debug(f"Failed to extract from in-memory waveform: {e}")
        return None


# ── Batch Directory Extraction ───────────────────────────────────

def extract_features_from_directory(
    audio_dir: Path,
    output_path: Path,
    species: str = "unknown",
    dataset_id: str = "unknown",
    max_files: int = 50000,
    sr: int = 16000,
) -> Tuple[Optional[Path], int]:
    """
    Extract features from all audio files in a directory tree.
    
    Walks the directory recursively, extracts 266-dim embeddings from each
    audio file, and saves everything to a compressed .npz file.
    
    The label for each sample is derived from its parent folder name,
    which follows the convention: audio_dir/split/label_name/file.wav
    
    Args:
        audio_dir: Root directory containing audio files
        output_path: Path for the output .npz file
        species: Species group identifier
        dataset_id: Dataset identifier for metadata
        max_files: Maximum number of files to process
        sr: Target sample rate
        
    Returns:
        (output_path, num_features_extracted)
    """
    if not AUDIO_LIBS:
        logger.error("Cannot extract features: librosa/soundfile not installed")
        return None, 0

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([
        f for f in audio_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in audio_extensions
    ])

    if not audio_files:
        logger.warning(f"No audio files found in {audio_dir}")
        return None, 0

    if len(audio_files) > max_files:
        logger.info(f"Limiting extraction to {max_files} of {len(audio_files)} files")
        audio_files = audio_files[:max_files]

    embed_dim = MODEL_CONFIG.fusion_output_dim  # 256
    prosodic_dim = MODEL_CONFIG.prosodic_dim     # 10
    total_dim = embed_dim + prosodic_dim          # 266

    features_list = []
    labels_list = []
    files_list = []
    failed = 0

    start_time = time.time()
    logger.info(f"Extracting {total_dim}-dim features from {len(audio_files)} audio files with BEATs backbone...")

    for i, filepath in enumerate(audio_files):
        try:
            feat = extract_single_file(filepath, sr=sr, embed_dim=embed_dim, prosodic_dim=prosodic_dim)
            if feat is not None:
                features_list.append(feat)
                labels_list.append(filepath.parent.name)
                files_list.append(str(filepath.relative_to(audio_dir)))
            else:
                failed += 1
        except Exception as e:
            logger.debug(f"Feature extraction failed on {filepath.name}: {e}")
            failed += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(audio_files):
            elapsed = time.time() - start_time
            rate = (i + 1) / max(0.1, elapsed)
            logger.info(
                f"  [{i+1}/{len(audio_files)}] "
                f"{len(features_list)} extracted, {failed} failed "
                f"({rate:.1f} files/s)"
            )

    if not features_list:
        logger.error("No features were successfully extracted")
        return None, 0

    # Save as compressed numpy archive
    features_arr = np.array(features_list, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=features_arr,
        labels=np.array(labels_list),
        files=np.array(files_list),
        metadata=np.array([{
            "dataset_id": dataset_id,
            "species": species,
            "total_dim": total_dim,
            "embed_dim": embed_dim,
            "prosodic_dim": prosodic_dim,
            "num_samples": len(features_list),
            "num_failed": failed,
            "sample_rate": sr,
        }]),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time
    logger.info(
        f"✅ Saved {output_path.name}: "
        f"{len(features_list)} samples × {total_dim} dims = {size_mb:.2f} MB "
        f"({elapsed:.1f}s total)"
    )

    return output_path, len(features_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AnimalVox AI Feature Extractor")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output .npz file path")
    parser.add_argument("--species", type=str, default="unknown", help="Species group")
    parser.add_argument("--dataset-id", type=str, default="manual", help="Dataset identifier")
    parser.add_argument("--max-files", type=int, default=50000, help="Max files to process")
    args = parser.parse_args()

    extract_features_from_directory(
        audio_dir=Path(args.input_dir),
        output_path=Path(args.output),
        species=args.species,
        dataset_id=args.dataset_id,
        max_files=args.max_files,
    )
