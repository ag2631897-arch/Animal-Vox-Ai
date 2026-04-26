"""
AnimalVox AI — Global Configuration
====================================
Central configuration for all components: audio processing,
model architectures, training hyperparameters, and deployment.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Project Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
LABEL_MAPS_DIR = DATA_DIR / "label_maps"
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
KB_DIR = PROJECT_ROOT / "knowledge_base"
KB_SEED_DIR = KB_DIR / "seed_data"
TRAINING_DIR = PROJECT_ROOT / "training"
CONFIGS_DIR = TRAINING_DIR / "configs"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, LABEL_MAPS_DIR,
          WEIGHTS_DIR, KB_SEED_DIR, CONFIGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Audio Processing Constants ─────────────────────────────────
@dataclass
class AudioConfig:
    """Audio preprocessing configuration."""
    sample_rate: int = 16000          # Standard for audio AI models
    clip_duration: float = 5.0        # Seconds per clip
    overlap_ratio: float = 0.5        # 50% overlap between clips
    n_mels: int = 128                 # Mel-spectrogram bands
    n_fft: int = 400                  # 25ms window at 16kHz
    hop_length: int = 160             # 10ms hop at 16kHz
    fmin: int = 50                    # Minimum frequency (Hz)
    fmax: int = 8000                  # Maximum frequency (Hz)
    top_db: int = 20                  # Silence threshold for noise gate
    max_duration: int = 60            # Maximum recording duration (seconds)

    @property
    def clip_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration)

    @property
    def hop_samples(self) -> int:
        return int(self.clip_samples * self.overlap_ratio)


# ── Behavior Label Schema ──────────────────────────────────────
BEHAVIOR_CLASSES = {
    "B00": "alarm_aerial",
    "B01": "alarm_ground",
    "B02": "territorial_warning",
    "B03": "mating_call",
    "B04": "contact_call",
    "B05": "distress_call",
    "B06": "feeding_call",
    "B07": "play_vocalization",
    "B08": "submission_signal",
    "B09": "aggression_warning",
    "B10": "social_bonding",
    "B11": "navigation_echolocation",
}

BEHAVIOR_LABELS = list(BEHAVIOR_CLASSES.values())
NUM_BEHAVIOR_CLASSES = len(BEHAVIOR_CLASSES)

# Reverse mapping
BEHAVIOR_ID_TO_NAME = BEHAVIOR_CLASSES
BEHAVIOR_NAME_TO_ID = {v: k for k, v in BEHAVIOR_CLASSES.items()}


# ── Species Configuration ──────────────────────────────────────
SPECIES_GROUPS = {
    "birds": {
        "id": 0,
        "display_name": "Birds",
        "emoji": "🐦",
        "encoder": "beats_finetuned_birds",
        "supported_behaviors": ["B00", "B01", "B02", "B03", "B04", "B05", "B06"],
    },
    "dogs": {
        "id": 1,
        "display_name": "Dogs",
        "emoji": "🐕",
        "encoder": "beats_finetuned_dogs",
        "supported_behaviors": ["B05", "B07", "B08", "B09", "B10"],
    },
    "cats": {
        "id": 2,
        "display_name": "Cats",
        "emoji": "🐈",
        "encoder": "beats_finetuned_dogs",  # Shared encoder with dogs
        "supported_behaviors": ["B03", "B05", "B09", "B10"],
    },
    "dolphins": {
        "id": 3,
        "display_name": "Dolphins",
        "emoji": "🐬",
        "encoder": "beats_finetuned_marine",
        "supported_behaviors": ["B04", "B05", "B07", "B10", "B11"],
    },
    "whales": {
        "id": 4,
        "display_name": "Whales",
        "emoji": "🐋",
        "encoder": "beats_finetuned_marine",  # Shared encoder with dolphins
        "supported_behaviors": ["B03", "B04", "B05", "B10", "B11"],
    },
    "frogs": {
        "id": 5,
        "display_name": "Frogs",
        "emoji": "🐸",
        "encoder": "beats_finetuned_birds",  # Fine-tuned from bird encoder
        "supported_behaviors": ["B02", "B03", "B05"],
    },
}

NUM_SPECIES_GROUPS = len(SPECIES_GROUPS)


# ── Model Architecture Config ──────────────────────────────────
@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # BEATs encoder
    beats_embed_dim: int = 768
    beats_projection_dim: int = 256

    # EfficientNet CNN
    cnn_feature_dim: int = 256

    # Prosodic features
    prosodic_dim: int = 10

    # Fusion layer
    fusion_input_dim: int = 522  # 256 + 256 + 10
    fusion_hidden_dim: int = 512
    fusion_output_dim: int = 256

    # Classifier
    num_behavior_classes: int = NUM_BEHAVIOR_CLASSES
    classifier_dropout: float = 0.2

    # Sequence Transformer
    seq_embed_dim: int = 276  # 12 + 1 + 1 + 256 + 6
    seq_num_heads: int = 4
    seq_num_layers: int = 4
    seq_feedforward_dim: int = 512
    seq_dropout: float = 0.1
    seq_length: int = 5  # Last 5 calls

    # Context output
    context_dim: int = 128


# ── Training Config ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # General
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True

    # Birds
    birds_lr: float = 1e-4
    birds_batch_size: int = 32
    birds_epochs: int = 30
    birds_weight_decay: float = 0.01

    # Dogs/Cats
    dogs_lr: float = 2e-4
    dogs_batch_size: int = 16
    dogs_epochs: int = 50
    dogs_weight_decay: float = 0.01

    # Marine
    marine_lr: float = 1e-4
    marine_batch_size: int = 16
    marine_epochs: int = 40
    marine_weight_decay: float = 0.01

    # Common
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    scheduler: str = "cosine"  # cosine | onecycle
    focal_loss_gamma: float = 2.0


# ── Inference Config ────────────────────────────────────────────
@dataclass
class InferenceConfig:
    """Inference and deployment configuration."""
    # Groq API
    groq_model: str = "mixtral-8x7b-32768"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_max_tokens: int = 512
    groq_temperature: float = 0.7

    # Performance
    max_inference_time: float = 3.0  # seconds
    batch_size: int = 1
    device: str = "cpu"  # cpu | cuda

    # Web App
    server_port: int = 7860
    max_file_size_mb: int = 10
    allowed_formats: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    )


# ── Singleton Instances ─────────────────────────────────────────
AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
INFERENCE_CONFIG = InferenceConfig()
