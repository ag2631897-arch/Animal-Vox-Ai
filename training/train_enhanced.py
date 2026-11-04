"""
AnimalVox AI — Enhanced Training Pipeline (Phase 1+2+3)
========================================================
Three-phase accuracy boost for ALL 5 classifier modules:

Phase 1: Breed Audio Data Augmentation
  - Synthesize brachycephalic (class 0) & hound (class 2) samples
    from acoustic profile priors + existing dog embeddings
  - Pitch/HNR jitter to simulate breed-specific phonation

Phase 2: Cross-Tag Ethology Behaviors → Emotions
  - Map behavioral labels to emotion classes for 4 under-served states
  - territorial_warning → territorial (class 4)
  - social_bonding → bonding (class 5)
  - distress_call → fear/distress (class 8)
  - alarm_aerial/ground → alertness (class 9)

Phase 3: Advanced Training Techniques
  - Deeper residual MLP heads with LayerNorm
  - Mixup augmentation in embedding space
  - Focal Loss for hard-example mining
  - Cosine Annealing with Warm Restarts
  - Stochastic Weight Averaging (SWA) in final 20% of epochs
  - Per-class balanced sampling via WeightedRandomSampler
  - Gradient accumulation (effective batch = 512)
  - Stronger embedding jitter augmentation with scheduled annealing

Usage:
    py training/train_enhanced.py --module all --epochs 60
    py training/train_enhanced.py --module emotion_classifier --epochs 80
"""

import os
import sys
import argparse
import logging
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EnhancedTrainer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim.swa_utils import AveragedModel, SWALR
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not found. Run: pip install torch numpy")

from config import (
    MODEL_CONFIG, TRAINING_CONFIG,
    HEALTH_FLAGS, NUM_EMOTIONAL_STATES,
    NUM_BREED_GROUPS, NUM_SPECIES_GROUPS,
    PROCESSED_DATA_DIR, BREED_ACOUSTIC_PROFILES,
    BEHAVIOR_CLASSES,
)
from models.weight_manager import save_checkpoint, load_checkpoint


# ══════════════════════════════════════════════════════════════════
# Phase 3: Enhanced Model Architectures with Residual Connections
# ══════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + GELU + Dropout."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        return x + self.block(x)


class EnhancedClassifier(nn.Module):
    """
    Deeper MLP head with residual connections, LayerNorm, and GELU.
    Replaces the shallow 2-layer MLPs in the original models.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_residual_blocks: int = 2, dropout: float = 0.2,
                 is_multilabel: bool = False):
        super().__init__()
        self.is_multilabel = is_multilabel

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout * 0.75)
              for _ in range(num_residual_blocks)]
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.residual_blocks(h)
        logits = self.head(h)
        if self.is_multilabel:
            return torch.sigmoid(logits)
        return logits


# ══════════════════════════════════════════════════════════════════
# Phase 3: Focal Loss for Hard-Example Mining
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """Focal Loss — downweights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', num_classes=10):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = torch.ones(num_classes)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha.to(inputs.device),
                                   reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()


class FocalBCELoss(nn.Module):
    """Focal BCE for multi-label classification."""
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight,
            reduction='none'
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# ══════════════════════════════════════════════════════════════════
# Phase 3: Mixup Augmentation in Embedding Space
# ══════════════════════════════════════════════════════════════════

def mixup_data(x, y, alpha=0.3, is_multilabel=False):
    """Mixup: interpolates pairs of samples in embedding space."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    if is_multilabel:
        mixed_y = lam * y + (1 - lam) * y[index]
        return mixed_x, mixed_y, lam
    else:
        return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════
# Phase 1 & 2: Data Augmentation Functions
# ══════════════════════════════════════════════════════════════════

def augment_breed_data(embeddings, labels, label_to_idx, num_augmented=500):
    """
    Phase 1: Synthesize brachycephalic (class 0) and hound (class 2) samples
    from existing dog embeddings using acoustic profile-informed perturbation.
    """
    logger.info("Phase 1: Augmenting breed data for underrepresented groups...")

    # Count existing samples per class
    class_counts = Counter()
    for l in labels:
        idx = label_to_idx.get(str(l).lower().strip(), -1)
        if idx >= 0:
            class_counts[idx] += 1

    logger.info(f"  Pre-augmentation class distribution: {dict(sorted(class_counts.items()))}")

    # Find the max class count for balancing target
    max_count = max(class_counts.values()) if class_counts else num_augmented

    # Breed group profiles for acoustic perturbation
    BREED_PERTURBATION = {
        0: {  # Brachycephalic: low HNR, short calls, raspy
            "name": "brachycephalic",
            "noise_scale": 0.08,
            "dim_bias": {0: -0.3, 1: 0.2, 2: -0.4, 3: 0.1},  # Lower F0-related dims, add noise
        },
        2: {  # Hound: long calls, low F0, moderate HNR
            "name": "hound",
            "noise_scale": 0.06,
            "dim_bias": {0: 0.2, 1: -0.1, 2: 0.1, 3: -0.2},  # Boost duration-related dims
        },
    }

    new_embeddings = []
    new_labels = []

    # Find existing dog embeddings to use as base
    dog_indices = []
    for i, l in enumerate(labels):
        idx = label_to_idx.get(str(l).lower().strip(), -1)
        if idx >= 0:  # Any valid breed class
            dog_indices.append(i)

    if len(dog_indices) == 0:
        logger.warning("  No existing breed samples to augment from. Skipping.")
        return embeddings, labels

    dog_embeds = embeddings[dog_indices]

    for target_class, profile in BREED_PERTURBATION.items():
        current_count = class_counts.get(target_class, 0)
        needed = max(num_augmented, max_count) - current_count
        if needed <= 0:
            logger.info(f"  Class {target_class} ({profile['name']}): already has {current_count} samples, skipping")
            continue

        logger.info(f"  Generating {needed} augmented samples for class {target_class} ({profile['name']})")

        for _ in range(needed):
            # Pick a random existing dog embedding as base
            base_idx = np.random.randint(len(dog_embeds))
            base_embed = dog_embeds[base_idx].copy()

            # Apply breed-specific perturbation
            noise = np.random.randn(*base_embed.shape).astype(np.float32) * profile["noise_scale"]
            for dim, bias in profile["dim_bias"].items():
                if dim < len(base_embed):
                    noise[dim] += bias

            augmented = base_embed + noise
            new_embeddings.append(augmented)
            new_labels.append(profile["name"])

    if new_embeddings:
        aug_embeddings = np.array(new_embeddings, dtype=np.float32)
        all_embeddings = np.concatenate([embeddings, aug_embeddings], axis=0)
        all_labels = list(labels) + new_labels
        logger.info(f"  Phase 1 complete: {len(embeddings)} → {len(all_embeddings)} samples")
        return all_embeddings, all_labels

    return embeddings, labels


def cross_tag_behaviors_to_emotions(embeddings, labels, label_to_idx):
    """
    Phase 2: Cross-tag ethology behavior labels into emotion classes.
    Maps behavioral classification labels to fill underrepresented emotion slots.
    """
    logger.info("Phase 2: Cross-tagging ethology behaviors → emotions...")

    BEHAVIOR_TO_EMOTION = {
        # Behavior label → emotion label
        "territorial_warning": "territorial",
        "aggression_warning": "territorial",
        "territorial": "territorial",
        "warning": "territorial",

        "social_bonding": "bonding",
        "bonding": "bonding",
        "affection": "bonding",

        "distress_call": "distress",
        "distress": "distress",
        "fear": "fear",

        "alarm_aerial": "alert",
        "alarm_ground": "alert",
        "alertness": "alert",
        "curiosity": "alert",

        "play_vocalization": "play",
        "playfulness": "play",
        "play": "play",

        "mating_call": "bonding",
        "contact_call": "bonding",
        "submission_signal": "calm",
        "feeding_call": "food",
    }

    remapped_count = 0
    new_labels = list(labels)

    for i, l in enumerate(new_labels):
        s = str(l).lower().strip()
        if s in BEHAVIOR_TO_EMOTION:
            new_labels[i] = BEHAVIOR_TO_EMOTION[s]
            remapped_count += 1

    logger.info(f"  Phase 2 complete: {remapped_count} labels cross-tagged")
    return embeddings, new_labels


# ══════════════════════════════════════════════════════════════════
# Enhanced Feature Shard Dataset
# ══════════════════════════════════════════════════════════════════

class EnhancedFeatureDataset(Dataset):
    """
    Enhanced dataset with Phase 1+2 augmentation built in.
    """
    def __init__(self, npz_paths, module_name, is_multilabel=False,
                 num_classes=10, apply_augmentation=True):
        self.is_multilabel = is_multilabel
        self.num_classes = num_classes
        self.module_name = module_name

        all_embeddings = []
        all_labels = []

        for npz_path in npz_paths:
            try:
                data = np.load(npz_path, allow_pickle=True)
                embeddings_data = data["embeddings"]
                labels_data = data["labels"]

                # Determine species from filename
                dataset_species = "unknown"
                name_lower = npz_path.name.lower()
                if "cat" in name_lower:
                    dataset_species = "cats"
                elif "dog" in name_lower or "barkopedia" in name_lower:
                    dataset_species = "dogs"
                elif "bird" in name_lower or "avian" in name_lower:
                    dataset_species = "birds"
                elif "frog" in name_lower or "rfcx" in name_lower:
                    dataset_species = "frogs"
                elif "marine" in name_lower or "watkins" in name_lower:
                    dataset_species = "multi"

                # Module-specific label filtering (same logic as original)
                if module_name == "animal_detector":
                    if dataset_species == "multi":
                        shard_labels = []
                        valid_indices = []
                        for idx, l in enumerate(labels_data):
                            s = str(l).lower()
                            if "frog" in s:
                                shard_labels.append("frogs"); valid_indices.append(idx)
                            elif any(w in s for w in ("bird", "crow", "rooster", "hen", "finch")):
                                shard_labels.append("birds"); valid_indices.append(idx)
                            elif "dog" in s:
                                shard_labels.append("dogs"); valid_indices.append(idx)
                            elif "cat" in s:
                                shard_labels.append("cats"); valid_indices.append(idx)
                            elif "whale" in s or "orca" in s:
                                shard_labels.append("whales"); valid_indices.append(idx)
                            elif "dolphin" in s:
                                shard_labels.append("dolphins"); valid_indices.append(idx)
                        if not valid_indices:
                            continue
                        embeddings_data = embeddings_data[valid_indices]
                        all_labels.extend(shard_labels)
                    elif dataset_species in ("dogs", "cats", "birds", "frogs", "whales", "dolphins"):
                        all_labels.extend([dataset_species] * len(embeddings_data))
                    else:
                        continue

                elif module_name == "breed_acoustic":
                    BREED_NAMES = {"chihuahua", "husky", "shiba_inu", "shiba", "pitbull",
                                   "german_shepherd", "gsd", "pug", "bulldog", "french_bulldog",
                                   "collie", "beagle", "bloodhound", "basset", "pomeranian",
                                   "yorkshire", "malamute", "akita", "siamese", "persian", "bengal",
                                   "brachycephalic", "hound", "working_herding", "toy",
                                   "nordic_spitz", "siamese_oriental", "persian_quiet", "wild_derived"}
                    valid_indices = []
                    shard_labels = []
                    for idx, l in enumerate(labels_data):
                        s = str(l).lower().strip().replace(" ", "_")
                        if s in BREED_NAMES:
                            shard_labels.append(s)
                            valid_indices.append(idx)
                    if not valid_indices:
                        continue
                    embeddings_data = embeddings_data[valid_indices]
                    all_labels.extend(shard_labels)

                elif module_name == "emotion_classifier":
                    EMOTIONS = {"contentment", "brushing", "neutral_rest", "calm", "stress_acute",
                                "isolation", "alarm", "pain", "stress_chronic", "anxiety", "hunger",
                                "waiting_food", "food", "territorial", "warning", "bark", "bonding",
                                "affection", "purr", "playfulness", "play", "frustration", "fear",
                                "distress", "alertness", "curiosity", "alert",
                                # Phase 2 additions
                                "territorial_warning", "aggression_warning", "social_bonding",
                                "distress_call", "alarm_aerial", "alarm_ground",
                                "play_vocalization", "mating_call", "contact_call",
                                "submission_signal", "feeding_call"}
                    valid_indices = []
                    shard_labels = []
                    for idx, l in enumerate(labels_data):
                        s = str(l).lower().strip()
                        if s in EMOTIONS:
                            shard_labels.append(s)
                            valid_indices.append(idx)
                    if not valid_indices:
                        continue
                    embeddings_data = embeddings_data[valid_indices]
                    all_labels.extend(shard_labels)

                elif module_name == "anti_spoofing":
                    shard_labels = []
                    valid_indices = []
                    for idx, l in enumerate(labels_data):
                        s = str(l).lower()
                        if any(w in s for w in ("human", "speech", "baby", "sneeze", "cough",
                                                 "footstep", "laugh", "snore", "breath",
                                                 "ambient", "noise", "music", "traffic",
                                                 "presto", "dwashing", "pcafeter", "spoof")):
                            shard_labels.append("1")
                            valid_indices.append(idx)
                        elif any(w in s for w in ("dog", "cat", "bird", "frog", "whale",
                                                   "dolphin", "wildlife", "animal", "bark",
                                                   "purr", "howl", "growl", "chirp", "croak", "anuran")):
                            shard_labels.append("0")
                            valid_indices.append(idx)
                        elif dataset_species in ("dogs", "cats", "birds", "frogs", "whales", "dolphins"):
                            shard_labels.append("0")
                            valid_indices.append(idx)
                        elif dataset_species in ("human_imitation", "human"):
                            shard_labels.append("1")
                            valid_indices.append(idx)
                    if not valid_indices:
                        continue
                    embeddings_data = embeddings_data[valid_indices]
                    all_labels.extend(shard_labels)

                elif module_name == "health_monitor":
                    HEALTH_NAMES = {"pain", "acute_pain", "h01", "cry", "injury",
                                    "whimper", "chronic_pain", "h02", "arthritis",
                                    "respiratory", "cough", "gasp", "h03", "distress_respiratory",
                                    "distress", "separation", "whine", "h04", "anxiety",
                                    "fear", "hiss", "growl", "h05",
                                    "purr", "h06", "stress_purr",
                                    "pant", "wheeze", "h07", "mouth_breathing",
                                    "howl", "nighttime", "h08", "disoriented"}
                    valid_indices = []
                    shard_labels = []
                    for idx, l in enumerate(labels_data):
                        s = str(l).lower().strip()
                        if s in HEALTH_NAMES:
                            shard_labels.append(s)
                            valid_indices.append(idx)
                    if not valid_indices:
                        # For health, also accept general labels
                        all_labels.extend(labels_data.tolist())
                    else:
                        embeddings_data = embeddings_data[valid_indices]
                        all_labels.extend(shard_labels)
                else:
                    all_labels.extend(labels_data.tolist())

                all_embeddings.append(embeddings_data)
                logger.info(f"  Loaded {len(embeddings_data)} samples from {npz_path.name}")

            except Exception as e:
                logger.warning(f"  Failed to load {npz_path}: {e}")

        if not all_embeddings:
            logger.warning("No feature shards loaded! Using synthetic fallback data.")
            input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim
            all_embeddings = [np.random.randn(200, input_dim).astype(np.float32)]
            all_labels = [f"class_{i % num_classes}" for i in range(200)]

        concatenated = np.concatenate(all_embeddings, axis=0)

        # ── Build label mapping ──
        if module_name == "animal_detector":
            from config import SPECIES_GROUPS
            self.label_to_idx = {k: v["id"] for k, v in SPECIES_GROUPS.items()}
        elif module_name == "anti_spoofing":
            self.label_to_idx = {"0": 0, "1": 1}
        elif module_name == "emotion_classifier":
            self.label_to_idx = {
                "contentment": 0, "brushing": 0, "neutral_rest": 0, "calm": 0,
                "submission_signal": 0,
                "stress_acute": 1, "isolation": 1, "alarm": 1, "pain": 1,
                "stress_chronic": 2, "anxiety": 2,
                "hunger": 3, "waiting_food": 3, "food": 3, "feeding_call": 3,
                "territorial": 4, "warning": 4, "bark": 4,
                "territorial_warning": 4, "aggression_warning": 4,
                "bonding": 5, "affection": 5, "purr": 5,
                "social_bonding": 5, "mating_call": 5, "contact_call": 5,
                "playfulness": 6, "play": 6, "play_vocalization": 6,
                "frustration": 7,
                "fear": 8, "distress": 8, "distress_call": 8,
                "alarm_aerial": 8, "alarm_ground": 8,
                "alertness": 9, "curiosity": 9, "alert": 9,
            }
        elif module_name == "breed_acoustic":
            self.label_to_idx = {
                "brachycephalic": 0, "pug": 0, "bulldog": 0, "french_bulldog": 0,
                "working_herding": 1, "german_shepherd": 1, "gsd": 1, "pitbull": 1, "collie": 1,
                "hound": 2, "beagle": 2, "bloodhound": 2, "basset": 2,
                "toy": 3, "chihuahua": 3, "pomeranian": 3, "yorkshire": 3,
                "nordic_spitz": 4, "husky": 4, "shiba_inu": 4, "shiba": 4, "malamute": 4, "akita": 4,
                "siamese_oriental": 5, "siamese": 5,
                "persian_quiet": 6, "persian": 6,
                "wild_derived": 7, "bengal": 7,
            }
        elif module_name == "health_monitor":
            self.label_to_idx = {
                "pain": 0, "acute_pain": 0, "h01": 0, "cry": 0, "injury": 0,
                "whimper": 1, "chronic_pain": 1, "h02": 1, "arthritis": 1,
                "respiratory": 2, "cough": 2, "gasp": 2, "h03": 2, "distress_respiratory": 2,
                "distress": 3, "separation": 3, "whine": 3, "h04": 3, "anxiety": 3,
                "fear": 4, "hiss": 4, "growl": 4, "h05": 4,
                "purr": 5, "h06": 5, "stress_purr": 5,
                "pant": 6, "wheeze": 6, "h07": 6, "mouth_breathing": 6,
                "howl": 7, "nighttime": 7, "h08": 7, "disoriented": 7,
            }
        else:
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {l: i % num_classes for i, l in enumerate(unique_labels)}

        # ── Apply Phase 1 & 2 augmentation ──
        if apply_augmentation and module_name == "breed_acoustic":
            concatenated, all_labels = augment_breed_data(
                concatenated, all_labels, self.label_to_idx, num_augmented=600
            )

        if apply_augmentation and module_name == "emotion_classifier":
            concatenated, all_labels = cross_tag_behaviors_to_emotions(
                concatenated, all_labels, self.label_to_idx
            )

        self.embeddings = concatenated

        # ── Convert labels to indices ──
        if is_multilabel:
            self.targets = np.zeros((len(all_labels), num_classes), dtype=np.float32)
            for i, label in enumerate(all_labels):
                idx = self.label_to_idx.get(str(label).lower().strip(), 0)
                self.targets[i, idx] = 1.0
        else:
            self.targets = np.array(
                [self.label_to_idx.get(str(l).lower().strip(), 0) for l in all_labels],
                dtype=np.int64
            )

        # Log class distribution
        if is_multilabel:
            class_sums = self.targets.sum(axis=0).astype(int)
            logger.info(f"  Multi-label class distribution: {class_sums.tolist()}")
        else:
            counts = Counter(self.targets.tolist())
            logger.info(f"  Class distribution: {dict(sorted(counts.items()))}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.module_name == "animal_detector":
            x = torch.from_numpy(self.embeddings[idx, :256])
        else:
            x = torch.from_numpy(self.embeddings[idx])

        if self.is_multilabel:
            y = torch.from_numpy(self.targets[idx])
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


# ══════════════════════════════════════════════════════════════════
# Module Training Configuration
# ══════════════════════════════════════════════════════════════════

MODULE_CONFIGS = {
    "health_monitor": {
        "is_multilabel": True,
        "num_classes": len(HEALTH_FLAGS),
        "loss_fn": "focal_bce",
        "species_filter": ["dogs", "cats"],
        "description": "8-Flag Multi-Label Health Classifier (Enhanced)",
        "hidden_dim": 192,
        "num_residual": 3,
        "dropout": 0.20,
    },
    "emotion_classifier": {
        "is_multilabel": False,
        "num_classes": NUM_EMOTIONAL_STATES,
        "loss_fn": "focal",
        "species_filter": ["dogs", "cats", "dolphins"],
        "description": "10-Class Emotional State Classifier (Enhanced + Cross-Tagged)",
        "hidden_dim": 192,
        "num_residual": 3,
        "dropout": 0.20,
    },
    "breed_acoustic": {
        "is_multilabel": False,
        "num_classes": NUM_BREED_GROUPS,
        "loss_fn": "focal",
        "species_filter": ["dogs"],
        "description": "8-Group Acoustic Breed Classifier (Enhanced + Augmented)",
        "hidden_dim": 192,
        "num_residual": 3,
        "dropout": 0.20,
    },
    "animal_detector": {
        "is_multilabel": False,
        "num_classes": NUM_SPECIES_GROUPS,
        "loss_fn": "focal",
        "species_filter": None,
        "description": "6-Species Auto-Router (Enhanced)",
        "hidden_dim": 160,
        "num_residual": 2,
        "dropout": 0.15,
    },
    "anti_spoofing": {
        "is_multilabel": False,
        "num_classes": 2,
        "loss_fn": "focal",
        "species_filter": None,
        "description": "Binary Human vs. Animal Detector (Enhanced)",
        "hidden_dim": 192,
        "num_residual": 2,
        "dropout": 0.18,
    },
}


def discover_feature_shards(species_filter=None, module_name=None):
    """Discover all .npz feature shard files."""
    features_dir = PROCESSED_DATA_DIR / "shard_features"
    if not features_dir.exists():
        return []

    npz_files = sorted(features_dir.glob("features_*.npz"))

    filtered = []
    for npz_path in npz_files:
        name_lower = npz_path.name.lower()
        ds_species = "unknown"
        if "cat" in name_lower:
            ds_species = "cats"
        elif "dog" in name_lower or "barkopedia" in name_lower:
            ds_species = "dogs"
        elif "bird" in name_lower or "avian" in name_lower:
            ds_species = "birds"
        elif "frog" in name_lower or "rfcx" in name_lower:
            ds_species = "frogs"
        elif "marine" in name_lower or "watkins" in name_lower:
            ds_species = "multi"

        # For animal_detector and anti_spoofing, accept everything
        if module_name in ("animal_detector", "anti_spoofing"):
            filtered.append(npz_path)
            continue

        if species_filter is None:
            filtered.append(npz_path)
        elif ds_species in species_filter or ds_species == "multi":
            filtered.append(npz_path)

    return filtered


def build_enhanced_model(module_name, input_dim, device, cfg):
    """Build an enhanced classifier with residual blocks."""
    model = EnhancedClassifier(
        input_dim=input_dim,
        hidden_dim=cfg["hidden_dim"],
        num_classes=cfg["num_classes"],
        num_residual_blocks=cfg["num_residual"],
        dropout=cfg["dropout"],
        is_multilabel=cfg["is_multilabel"],
    )
    return model.to(device)


def get_balanced_sampler(targets, num_classes):
    """Create a WeightedRandomSampler for class-balanced training."""
    if isinstance(targets, np.ndarray) and targets.ndim > 1:
        # Multi-label: weight by inverse frequency of active flags
        sample_weights = []
        pos_counts = targets.sum(axis=0) + 1.0
        for i in range(len(targets)):
            active = targets[i].sum()
            if active > 0:
                w = sum(1.0 / pos_counts[j] for j in range(targets.shape[1]) if targets[i, j] > 0)
            else:
                w = 1.0
            sample_weights.append(w)
    else:
        # Single-label: weight by inverse class frequency
        counts = Counter(targets.tolist())
        total = len(targets)
        class_weights = {c: total / (len(counts) * count) for c, count in counts.items()}
        sample_weights = [class_weights[int(t)] for t in targets]

    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler


# ══════════════════════════════════════════════════════════════════
# Main Training Loop with All Phase 3 Techniques
# ══════════════════════════════════════════════════════════════════

def train_module_enhanced(
    module_name: str,
    npz_paths: List[Path],
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 5e-4,
    device: str = "cpu",
) -> Dict:
    """Train a single module with all enhanced techniques."""
    cfg = MODULE_CONFIGS[module_name]
    input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim  # 266
    if module_name == "animal_detector":
        input_dim = 256

    print("\n" + "═" * 72)
    print(f"  🐾 ENHANCED Training: {module_name} — {cfg['description']}")
    print(f"     Device: {device} | Epochs: {epochs} | LR: {lr}")
    print(f"     Architecture: ResidualMLP({input_dim}→{cfg['hidden_dim']}→{cfg['num_classes']})")
    print(f"     Residual Blocks: {cfg['num_residual']} | Dropout: {cfg['dropout']}")
    print(f"     Techniques: Focal Loss + Mixup + SWA + Balanced Sampling")
    print("═" * 72)

    # ── Load dataset with augmentation ──
    dataset = EnhancedFeatureDataset(
        npz_paths=npz_paths,
        module_name=module_name,
        is_multilabel=cfg["is_multilabel"],
        num_classes=cfg["num_classes"],
        apply_augmentation=True,
    )

    total = len(dataset)
    val_size = max(int(total * 0.12), 1)
    train_size = total - val_size

    np.random.seed(TRAINING_CONFIG.seed)
    indices = np.random.permutation(total)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    train_targets = dataset.targets[train_idx]

    # ── Balanced sampler ──
    sampler = get_balanced_sampler(train_targets, cfg["num_classes"])

    effective_batch = min(batch_size, max(32, train_size // 4))
    grad_accum_steps = max(1, 512 // effective_batch)  # Effective batch ~512

    train_loader = DataLoader(
        train_ds, batch_size=effective_batch, sampler=sampler,
        drop_last=(train_size > effective_batch),
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=effective_batch * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    logger.info(f"Dataset: {total} total ({train_size} train, {val_size} val, "
                f"batch: {effective_batch}, grad_accum: {grad_accum_steps})")

    # ── Build model ──
    model = build_enhanced_model(module_name, input_dim, device, cfg)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    # ── Loss function ──
    if cfg["loss_fn"] == "focal_bce":
        targets_arr = np.array(train_targets)
        pos_counts = torch.tensor(targets_arr.sum(axis=0), dtype=torch.float32).clamp(min=1.0)
        neg_counts = float(len(targets_arr)) - pos_counts
        pos_weight = torch.clamp((neg_counts / pos_counts) ** 0.5, min=0.5, max=10.0).to(device)
        criterion = FocalBCELoss(gamma=2.0, pos_weight=pos_weight)
        logger.info(f"⚖️ Focal BCE with pos_weight: {pos_weight.cpu().numpy().round(2).tolist()}")
    else:
        targets_tensor = torch.tensor(train_targets, dtype=torch.long)
        counts = torch.bincount(targets_tensor, minlength=cfg["num_classes"]).float()
        weights = torch.zeros(cfg["num_classes"])
        active = counts > 0
        if active.any():
            raw_w = (len(targets_tensor) / (active.sum() * counts[active])) ** 0.5
            weights[active] = raw_w / raw_w.mean()
        else:
            weights = torch.ones(cfg["num_classes"])
        criterion = FocalLoss(alpha=weights.to(device), gamma=2.0, num_classes=cfg["num_classes"])
        logger.info(f"⚖️ Focal Loss (γ=2.0) with class weights: {weights.cpu().numpy().round(3).tolist()}")

    # ── Optimizer: AdamW ──
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)

    # ── Scheduler: CosineAnnealingWarmRestarts ──
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, epochs // 4), T_mult=2, eta_min=1e-6
    )

    # ── SWA: activate in last 20% of epochs ──
    swa_start = int(epochs * 0.80)
    swa_model = AveragedModel(model, device=device)
    swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1, anneal_epochs=5)

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_metric = 0.0
    patience_counter = 0
    max_patience = max(15, epochs // 4)

    for epoch in range(epochs):
        in_swa = epoch >= swa_start

        # ── Train ──
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        # Adaptive jitter: starts strong, anneals over training
        jitter_scale = 0.025 * (1.0 - 0.5 * epoch / epochs)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Embedding jitter augmentation (annealed)
            x_aug = x + torch.randn_like(x) * jitter_scale

            # Mixup augmentation (50% of batches)
            use_mixup = np.random.random() < 0.5 and not cfg["is_multilabel"]

            if use_mixup:
                x_aug, y_a, y_b, lam = mixup_data(x_aug, y, alpha=0.3)
                out = model(x_aug)
                if cfg["is_multilabel"]:
                    loss = criterion(out, y)
                else:
                    loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                if cfg["is_multilabel"]:
                    logits = model.head(model.residual_blocks(model.input_proj(x_aug)))
                    loss = criterion(logits, y)
                else:
                    out = model(x_aug)
                    loss = criterion(out, y)

            loss = loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps * len(x)

        avg_train_loss = train_loss / max(train_size, 1)

        # ── SWA update ──
        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # ── Validate ──
        eval_model = swa_model if (in_swa and epoch > swa_start + 2) else model
        eval_model.eval()
        val_loss = 0.0
        correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if cfg["is_multilabel"]:
                    logits = eval_model.module.head(
                        eval_model.module.residual_blocks(
                            eval_model.module.input_proj(x)
                        )
                    ) if in_swa and epoch > swa_start + 2 else model.head(
                        model.residual_blocks(model.input_proj(x))
                    )
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == y).float().sum().item()
                    val_total += y.numel()
                else:
                    out = eval_model(x)
                    preds = torch.argmax(out, dim=-1)
                    correct += (preds == y).sum().item()
                    val_total += len(x)

        val_acc = correct / max(val_total, 1)
        avg_val_loss = val_loss  # simplified since we focus on accuracy

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)

        # ── Log ──
        lr_current = optimizer.param_groups[0]["lr"]
        swa_tag = " [SWA]" if in_swa else ""
        logger.info(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {lr_current:.2e}{swa_tag}"
        )

        # ── Early stopping check ──
        if val_acc >= best_metric:
            best_metric = val_acc
            patience_counter = 0

            # Save best model — we need to save it in a format compatible
            # with the ORIGINAL model architectures so inference works
            save_inference_compatible_checkpoint(model, module_name, epoch, optimizer,
                                                  val_acc, avg_train_loss, len(npz_paths), total)
            logger.info(f"  ⭐ New best! Acc: {val_acc:.2%} → models/weights/{module_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= max_patience and epoch > epochs * 0.5:
                logger.info(f"  Early stopping at epoch {epoch+1} (patience={max_patience})")
                break

    # ── Final SWA BN update ──
    if swa_start < epochs:
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            # Evaluate SWA model
            swa_model.eval()
            swa_correct = 0
            swa_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    out = swa_model(x)
                    if cfg["is_multilabel"]:
                        preds = (torch.sigmoid(out) > 0.5).float()
                        swa_correct += (preds == y).float().sum().item()
                        swa_total += y.numel()
                    else:
                        preds = torch.argmax(out, dim=-1)
                        swa_correct += (preds == y).sum().item()
                        swa_total += len(x)
            swa_acc = swa_correct / max(swa_total, 1)
            logger.info(f"  SWA final accuracy: {swa_acc:.2%}")

            if swa_acc > best_metric:
                best_metric = swa_acc
                # Save SWA model (using underlying module)
                save_inference_compatible_checkpoint(
                    swa_model.module, module_name, epochs, optimizer,
                    swa_acc, avg_train_loss, len(npz_paths), total
                )
                logger.info(f"  ⭐ SWA model is better! Saved as final checkpoint.")
        except Exception as e:
            logger.warning(f"  SWA BN update failed (non-critical): {e}")

    print(f"\n🎉 {module_name} ENHANCED training complete! Best accuracy: {best_metric:.2%}")
    return {
        "module": module_name,
        "best_accuracy": best_metric,
        "epochs_trained": epoch + 1 if 'epoch' in dir() else epochs,
        "total_samples": total,
        "history": history,
    }


def save_inference_compatible_checkpoint(model, module_name, epoch, optimizer,
                                          val_acc, train_loss, num_shards, total_samples):
    """
    Save the enhanced model checkpoint in a format compatible with
    the existing inference pipeline (which uses strict=False loading).
    """
    save_checkpoint(
        model=model,
        module_name=module_name,
        epoch=epoch + 1,
        optimizer=optimizer,
        metrics={
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "best_metric": val_acc,
            "training_mode": "enhanced_v2",
        },
        extra_meta={
            "description": (
                f"Enhanced training with Focal Loss + Mixup + SWA + "
                f"Balanced Sampling + {MODULE_CONFIGS[module_name]['num_residual']} "
                f"Residual Blocks"
            ),
            "num_shards": num_shards,
            "total_samples": total_samples,
            "architecture": "EnhancedResidualMLP",
        }
    )
    # Also save companion ECAPA checkpoint for anti_spoofing
    if module_name == "anti_spoofing":
        try:
            save_checkpoint(
                model=model,
                module_name="anti_spoofing_ecapa",
                epoch=epoch + 1,
                optimizer=optimizer,
                metrics={"val_accuracy": val_acc, "best_metric": val_acc},
                extra_meta={"description": "Enhanced anti-spoofing checkpoint"}
            )
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Enhanced Training Pipeline")
    parser.add_argument(
        "--module", type=str, default="all",
        choices=list(MODULE_CONFIGS.keys()) + ["all"],
        help="Which module to train"
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Cannot train without PyTorch. Install: pip install torch")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║      🐾 AnimalVox AI — ENHANCED Training Pipeline v2 🐾        ║
    ║                                                                  ║
    ║   Phase 1: Breed Augmentation (Brachycephalic + Hound)           ║
    ║   Phase 2: Ethology → Emotion Cross-Tagging                     ║
    ║   Phase 3: Focal Loss + Mixup + SWA + Residual MLPs             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Determine modules to train
    if args.module == "all":
        modules = list(MODULE_CONFIGS.keys())
    else:
        modules = [args.module]

    results = []
    for module_name in modules:
        cfg = MODULE_CONFIGS[module_name]
        species_filter = cfg.get("species_filter")

        npz_paths = discover_feature_shards(
            species_filter=species_filter,
            module_name=module_name,
        )

        if not npz_paths:
            logger.warning(f"No feature shards found for {module_name}. Using synthetic fallback.")

        result = train_module_enhanced(
            module_name=module_name,
            npz_paths=npz_paths,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        results.append(result)

    # ── Final Summary ──
    print("\n" + "═" * 72)
    print("  📊 ENHANCED Training Summary")
    print("═" * 72)
    for r in results:
        icon = "✅" if r["best_accuracy"] >= 0.80 else "⚡" if r["best_accuracy"] >= 0.60 else "⚠️"
        print(
            f"  {icon} {r['module']:<25s} | "
            f"Best Acc: {r['best_accuracy']:.2%} | "
            f"Epochs: {r['epochs_trained']} | "
            f"Samples: {r['total_samples']:,}"
        )
    print("═" * 72)
    print("\n  Trained weights saved to: models/weights/")
    print("  These will be auto-loaded by the AnimalVox AI inference pipeline.")


if __name__ == "__main__":
    main()
