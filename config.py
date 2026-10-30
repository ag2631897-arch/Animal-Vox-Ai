"""
AnimalVox AI — Global Configuration
====================================
Central configuration for all components: audio processing,
model architectures, training hyperparameters, deployment,
and the 5 advanced intelligence modules (SRS v1.1).
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

# New paths for expanded modules
PET_PROFILES_DIR = DATA_DIR / "pet_profiles"
DIARY_DIR = DATA_DIR / "diary"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, LABEL_MAPS_DIR,
          WEIGHTS_DIR, KB_SEED_DIR, CONFIGS_DIR, PET_PROFILES_DIR, DIARY_DIR]:
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


# ── Anti-Spoofing Configuration (SRS Feature 0) ───────────────
@dataclass
class AntiSpoofingConfig:
    """Human imitation detection thresholds and parameters."""
    # Human vocal tract formant ranges (Hz) — used as spoof indicators
    human_f1_range: tuple = (300, 800)
    human_f2_range: tuple = (1000, 2500)
    human_f3_range: tuple = (2500, 3500)
    human_vocal_tract_length_cm: tuple = (14.0, 18.0)  # Adult human range
    human_f0_max: float = 450.0  # Max typical human fundamental freq (Hz)

    # Decision thresholds
    spoof_rejection_threshold: float = 0.70   # Reject if P(human) > 0.70
    formant_match_tolerance_hz: float = 150.0  # Hz tolerance for formant matching
    min_audio_duration_sec: float = 0.5  # Minimum clip length for spoof check

    # ECAPA-TDNN embedding dimension
    ecapa_embed_dim: int = 192


# ── Health Monitoring Flags (SRS Module 01) ────────────────────
HEALTH_FLAGS = {
    "H01": {
        "name": "Pain Vocalization — Acute",
        "species": ["dogs", "cats"],
        "description": "High-pitched cry, sudden onset, short duration, elevated F0, high HNR",
        "veterinary_context": "Injury, post-surgical pain, acute illness",
    },
    "H02": {
        "name": "Pain Vocalization — Chronic",
        "species": ["dogs", "cats"],
        "description": "Low-pitched repetitive whimper, reduced amplitude, compressed F0 range",
        "veterinary_context": "Arthritis, chronic disease, persistent pain",
    },
    "H03": {
        "name": "Respiratory Distress",
        "species": ["dogs", "cats"],
        "description": "Laboured breathing sounds, rasping quality, elevated ZCR, irregular rhythm",
        "veterinary_context": "Asthma (cats), pneumonia, heart failure, obstruction",
    },
    "H04": {
        "name": "Distress Cry — Separation",
        "species": ["dogs"],
        "description": "Repetitive high-pitched bark/whine, escalating amplitude, tight call intervals",
        "veterinary_context": "Separation anxiety — not medical but welfare concern",
    },
    "H05": {
        "name": "Fear-Pain Combination",
        "species": ["dogs", "cats"],
        "description": "Hiss/growl combined with pain cry markers — mixed acoustic signature",
        "veterinary_context": "Pain combined with fear of handling",
    },
    "H06": {
        "name": "Cat Purr — Paradoxical",
        "species": ["cats"],
        "description": "Purring during known stress context (vet visit, post-injury) — abnormal purr",
        "veterinary_context": "Cats purr under stress and pain, not just contentment",
    },
    "H07": {
        "name": "Breathing Abnormality — Open Mouth",
        "species": ["cats"],
        "description": "Mouth-breathing click/rasp sounds — extremely abnormal in cats",
        "veterinary_context": "Cardiac or respiratory emergency in cats",
    },
    "H08": {
        "name": "Nighttime Vocalisation — Cognitive",
        "species": ["dogs", "cats"],
        "description": "Disoriented calling at night, irregular pitch, no obvious trigger",
        "veterinary_context": "Cognitive dysfunction syndrome (CDS) — dementia-like",
    },
}

NUM_HEALTH_FLAGS = len(HEALTH_FLAGS)
HEALTH_FLAG_IDS = list(HEALTH_FLAGS.keys())

MEDICAL_DISCLAIMER = (
    "⚕ Medical Disclaimer: AnimalVox AI is an early-awareness aid only. "
    "It is not a veterinary diagnostic tool. Do not use outputs as a "
    "substitute for professional veterinary assessment."
)

@dataclass
class HealthMonitorConfig:
    """Pet health monitoring thresholds and baseline configuration."""
    # Confidence and deviation gating
    flag_confidence_threshold: float = 0.75
    deviation_threshold_sigma: float = 1.5
    respiratory_deviation_sigma: float = 2.0

    # Personal baseline system
    baseline_min_clips: int = 50        # Minimum clips to establish baseline
    baseline_ema_alpha: float = 0.10    # Weekly EMA smoothing factor
    baseline_suppression_sigma: float = 1.5  # Suppress flags below this

    # Vocal frequency drop detection
    vocal_drop_threshold_pct: float = 0.50  # Flag if call freq drops 50%+
    vocal_drop_window_hours: int = 24


# ── Breed Identification (SRS Module 02) ──────────────────────
BREED_ACOUSTIC_PROFILES = {
    "brachycephalic": {
        "display_name": "Brachycephalic Dogs",
        "f0_range_hz": (200, 400),
        "breathiness": "high",
        "hnr_threshold_db": 8.0,  # HNR < 8 dB = raspy
        "call_duration": "short",
        "examples": ["French Bulldog", "Pug", "Bulldog", "Boston Terrier"],
    },
    "working_herding": {
        "display_name": "Working / Herding Dogs",
        "f0_range_hz": (600, 900),
        "breathiness": "low",
        "hnr_threshold_db": 18.0,  # HNR > 18 dB = clear tonal
        "call_duration": "medium",
        "examples": ["German Shepherd", "Border Collie", "Malinois"],
    },
    "hound": {
        "display_name": "Hound Dogs",
        "f0_range_hz": (150, 500),
        "breathiness": "low",
        "hnr_threshold_db": 12.0,
        "call_duration": "long",  # Bay duration > 1.5s
        "examples": ["Beagle", "Bloodhound", "Basset Hound", "Coonhound"],
    },
    "toy": {
        "display_name": "Toy Breeds",
        "f0_range_hz": (900, 1500),
        "breathiness": "low",
        "hnr_threshold_db": 15.0,
        "call_duration": "very_short",
        "examples": ["Chihuahua", "Yorkshire Terrier", "Pomeranian"],
    },
    "nordic_spitz": {
        "display_name": "Nordic / Spitz Dogs",
        "f0_range_hz": (300, 800),
        "breathiness": "low",
        "hnr_threshold_db": 14.0,
        "call_duration": "long",
        "examples": ["Husky", "Malamute", "Samoyed", "Akita"],
    },
    "siamese_oriental": {
        "display_name": "Siamese / Oriental Cats",
        "f0_range_hz": (800, 1200),
        "breathiness": "low",
        "hnr_threshold_db": 16.0,
        "call_duration": "long",
        "examples": ["Siamese", "Balinese", "Oriental Shorthair"],
    },
    "persian_quiet": {
        "display_name": "Persian / Quiet Cats",
        "f0_range_hz": (200, 500),
        "breathiness": "high",
        "hnr_threshold_db": 10.0,
        "call_duration": "short",
        "examples": ["Persian", "Himalayan", "Exotic Shorthair"],
    },
    "wild_derived": {
        "display_name": "Wild-Derived Cats",
        "f0_range_hz": (400, 900),
        "breathiness": "low",
        "hnr_threshold_db": 14.0,
        "call_duration": "short",
        "examples": ["Bengal", "Savannah", "Maine Coon"],
    },
}

NUM_BREED_GROUPS = len(BREED_ACOUSTIC_PROFILES)

@dataclass
class BreedIDConfig:
    """Breed identification model parameters."""
    # Audio branch
    audio_breed_embed_dim: int = 128
    num_breed_groups: int = NUM_BREED_GROUPS

    # Visual branch (EfficientNet-B4)
    visual_feature_dim: int = 1792  # EfficientNet-B4 penultimate
    visual_projection_dim: int = 256
    num_visual_breeds: int = 120  # Stanford Dogs breed count

    # Multimodal fusion weights
    audio_fusion_weight: float = 0.40
    visual_fusion_weight: float = 0.60

    # Output
    top_k_breeds: int = 3


# ── Emotional States (SRS Module 03) ──────────────────────────
EMOTIONAL_STATES = {
    "E01": {
        "name": "Contentment",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Low-frequency purr (cats), soft low bark/whimper, slow call rate, high HNR, smooth F0",
        "context": "Post-feeding, social grooming, lap time, play aftermath",
    },
    "E02": {
        "name": "Stress — Acute",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Rapid high-pitched repetition, elevated amplitude, compressed call intervals, rising F0",
        "context": "Thunderstorm, vet visit, loud noise, stranger entry",
    },
    "E03": {
        "name": "Stress — Chronic",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Slightly elevated call rate, subtle F0 elevation above baseline, reduced HNR vs. baseline",
        "context": "Persistent environmental stress, new household member, schedule change",
    },
    "E04": {
        "name": "Hunger",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Rhythmic whine/meow, regular inter-call interval, moderate amplitude",
        "context": "Pre-feeding time, food bowl empty, feeding schedule disruption",
    },
    "E05": {
        "name": "Territorial Signalling",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Loud sharp bark/hiss, high amplitude, sudden onset, addressed outward",
        "context": "Stranger at door, other animal visible, property boundary",
    },
    "E06": {
        "name": "Social Bonding",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Soft trill/whimper, low amplitude, short duration, follows human vocalization",
        "context": "Greeting owner, close physical contact, play initiation",
    },
    "E07": {
        "name": "Playfulness",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Short rapid bursts, rising F0, irregular rhythm, mid-amplitude",
        "context": "Play sessions, toy interaction, invitation to play",
    },
    "E08": {
        "name": "Frustration",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Repetitive whine without relief, escalating amplitude, irregular F0",
        "context": "Blocked access, delayed feeding, ignored attention seeking",
    },
    "E09": {
        "name": "Excitement",
        "species": ["dogs"],
        "acoustic_markers": "High-amplitude rapid bark clusters, very short inter-call interval, rising pitch",
        "context": "Owner return, leash appearance, car ride anticipation",
    },
    "E10": {
        "name": "Loneliness / Isolation",
        "species": ["dogs", "cats"],
        "acoustic_markers": "Low-amplitude calling without response, slowing over time, reduced vocal energy",
        "context": "Home alone, separated from bonded companion",
    },
}

NUM_EMOTIONAL_STATES = len(EMOTIONAL_STATES)
EMOTIONAL_STATE_IDS = list(EMOTIONAL_STATES.keys())

@dataclass
class EmotionalTrackingConfig:
    """Emotional tracking and diary configuration."""
    # Database
    diary_db_name: str = "pet_diary.db"

    # Anomaly detection
    anomaly_sigma_threshold: float = 2.0  # Flag if weekly stress > 2 sigma
    weekly_report_day: str = "Sunday"

    # Emotion classifier
    emotion_embed_dim: int = 128
    num_emotion_classes: int = NUM_EMOTIONAL_STATES


# ── Wellbeing Signals (SRS Module 04) ─────────────────────────
WELLBEING_DIMENSIONS = {
    "physical_distress": {
        "name": "Physical Distress",
        "weight": 0.35,
        "description": "Pain cry patterns, respiratory sounds, reduced vocalisation, abnormal call duration",
        "domain": "Domain 1 & 3 (Nutrition + Health)",
        "direction": "lower_is_better",  # 0 = no distress, 100 = severe
    },
    "environmental_stress": {
        "name": "Environmental Stress",
        "weight": 0.20,
        "description": "Noise-triggered stress, stranger alerts, routine disruption, confinement",
        "domain": "Domain 2 (Environment)",
        "direction": "lower_is_better",
    },
    "isolation_stress": {
        "name": "Isolation Stress",
        "weight": 0.20,
        "description": "Separation bursts, sustained quiet calling, silence-then-burst cycling",
        "domain": "Domain 4 (Behaviour)",
        "direction": "lower_is_better",
    },
    "contentment": {
        "name": "Contentment Baseline",
        "weight": 0.25,
        "description": "Contentment purr, social trill, play frequency, post-feed contentment, greeting vocalisation",
        "domain": "Domain 5 (Mental State)",
        "direction": "higher_is_better",  # 0 = no contentment, 100 = thriving
    },
}

@dataclass
class WellbeingConfig:
    """Wellbeing signal scoring configuration."""
    # Composite formula weights (sum to 1.0)
    physical_weight: float = 0.35
    environmental_weight: float = 0.20
    isolation_weight: float = 0.20
    contentment_weight: float = 0.25

    # Scoring bounds
    min_score: float = 0.0
    max_score: float = 100.0

    # Trend detection
    trend_window_weeks: int = 4
    trend_significant_delta: float = 5.0  # Points change to show arrow


# ── Soundscape Categories (SRS Module 05) ─────────────────────
SOUNDSCAPE_CATEGORIES = [
    "indoor",
    "outdoor_garden",
    "urban_park",
    "forest_wilderness",
    "marine",
]

NUM_SOUNDSCAPES = len(SOUNDSCAPE_CATEGORIES)
NUM_SPECIES_GROUPS = 6

@dataclass
class AnimalDetectionConfig:
    """Animal detection and auto-routing configuration."""
    # Presence detector
    presence_model_max_size_mb: float = 1.5
    presence_latency_target_ms: float = 150.0

    # Species group classification
    num_species_groups: int = NUM_SPECIES_GROUPS
    num_soundscapes: int = NUM_SOUNDSCAPES

    # Source separation (Conv-TasNet)
    max_sources: int = 4  # Max simultaneous animal calls
    separation_sample_rate: int = 16000

    # Human voice exclusion (WebRTC VAD)
    vad_aggressiveness: int = 3  # 0-3, higher = more aggressive filtering


# ── Streaming Translation Config (SRS Feature 3) ──────────────
@dataclass
class StreamingConfig:
    """Real-time streaming translation and TTS configuration."""
    # Phrase buffering
    phrase_buffer_max_words: int = 4  # Max words before flushing
    phrase_punctuation_triggers: str = ".,!?;:—–"  # Chars that trigger flush

    # SSE (Server-Sent Events)
    sse_retry_ms: int = 1000

    # TTS tier
    tts_default: str = "web_speech_api"  # Free forever, browser-native
    tts_premium: str = "elevenlabs"      # eleven_turbo_v2
    elevenlabs_model: str = "eleven_turbo_v2"

    # Latency targets
    target_first_word_ms: int = 2500   # Max ms from recording stop to first word
    sonar_animation_duration_ms: int = 2000  # "Listening to..." animation


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
    # BEATs encoder (Tier 1 — 90M params)
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

    # Sequence Transformer (Tier 4 — 1.8M params)
    seq_embed_dim: int = 276  # 12 + 1 + 1 + 256 + 6
    seq_num_heads: int = 4
    seq_num_layers: int = 4
    seq_feedforward_dim: int = 512
    seq_dropout: float = 0.1
    seq_length: int = 5  # Last 5 calls

    # Context output
    context_dim: int = 128

    # Anti-Spoofing (Tier 0 — 6.2M params, ECAPA-TDNN)
    anti_spoof_embed_dim: int = 192
    anti_spoof_channels: List[int] = field(
        default_factory=lambda: [512, 512, 512, 512, 1536]
    )

    # Health Monitor heads (part of Tier 3 — 3.5M shared)
    num_health_flags: int = NUM_HEALTH_FLAGS
    health_hidden_dim: int = 128

    # Emotion heads (part of Tier 3)
    num_emotion_classes: int = NUM_EMOTIONAL_STATES
    emotion_hidden_dim: int = 160

    # Breed Acoustic heads (part of Tier 3)
    num_breed_groups: int = NUM_BREED_GROUPS
    breed_hidden_dim: int = 160

    # Breed Visual (Tier 2 — 19M params, EfficientNet-B4)
    visual_backbone: str = "efficientnet_b4"
    visual_feature_dim: int = 1792
    visual_projection_dim: int = 256


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
    # Groq API (Tier 5 — Llama 3.1 8B, free tier)
    groq_model: str = "openai/gpt-oss-20b"
    groq_fallback_model: str = "qwen/qwen3.8-27b"
    # A translation is a concise behavioural interpretation, not creative
    # prose.  Keep sampling and output length constrained accordingly.
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.2

    # Performance
    max_inference_time: float = 3.0  # seconds
    batch_size: int = 1
    device: str = "cpu"  # cpu | cuda

    # Web App
    server_port: int = 7860
    max_file_size_mb: int = 10
    allowed_formats: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"]
    )


# ── Singleton Instances ─────────────────────────────────────────
AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
INFERENCE_CONFIG = InferenceConfig()
ANTI_SPOOFING_CONFIG = AntiSpoofingConfig()
HEALTH_MONITOR_CONFIG = HealthMonitorConfig()
BREED_ID_CONFIG = BreedIDConfig()
EMOTIONAL_TRACKING_CONFIG = EmotionalTrackingConfig()
WELLBEING_CONFIG = WellbeingConfig()
ANIMAL_DETECTION_CONFIG = AnimalDetectionConfig()
STREAMING_CONFIG = StreamingConfig()
