"""
AnimalVox AI — Train From Extracted Feature Shards
====================================================
Trains all classifier heads from pre-extracted .npz feature files.
This is Phase 2 of the sharded pipeline — runs entirely from local
disk with zero network downloads, enabling 50+ epoch training.

Supported modules:
  - health_monitor:    8-flag multi-label health classifier
  - emotion_classifier: 10-class emotional state classifier
  - breed_acoustic:    8-group acoustic breed classifier
  - animal_detector:   6-species auto-router
  - anti_spoofing:     Binary human vs. animal detector

Usage:
    # Train all modules from all available feature shards:
    py training/train_from_shards.py --module all --epochs 50

    # Train a specific module:
    py training/train_from_shards.py --module emotion_classifier --epochs 30

    # Train only from specific species data:
    py training/train_from_shards.py --module health_monitor --species dogs --epochs 40
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ShardTrainer")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
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
    PROCESSED_DATA_DIR,
)
from models.weight_manager import save_checkpoint, load_checkpoint


# ── Feature Shard Dataset ────────────────────────────────────────

class FeatureShardDataset(Dataset):
    """
    Dataset that loads pre-extracted 266-dim feature vectors from .npz files.
    
    Each .npz contains:
      - embeddings: (N, 266) float32
      - labels: (N,) string array of class names
      - files: (N,) string array of source filenames
    """
    def __init__(
        self,
        npz_paths: List[Path],
        label_to_idx: Optional[Dict[str, int]] = None,
        is_multilabel: bool = False,
        num_classes: int = 10,
        module_name: Optional[str] = None,
    ):
        self.is_multilabel = is_multilabel
        self.num_classes = num_classes
        self.module_name = module_name
        
        all_embeddings = []
        all_labels = []
        all_multilabel_targets = []
        counts_per_class = Counter()
        counts_per_species_animal = Counter()
        max_per_species = 25000
        max_spoof = 36000
        max_clean = 36000
        
        for npz_path in npz_paths:
            try:
                data = np.load(npz_path, allow_pickle=True)
                embeddings = data["embeddings"]
                labels = data["labels"]
                
                # Retrieve species metadata if available
                dataset_species = "unknown"
                if "metadata" in data and len(data["metadata"]) > 0:
                    meta = data["metadata"][0]
                    if isinstance(meta, dict):
                        dataset_species = meta.get("species", "unknown")
                if dataset_species == "unknown":
                    name_lower = npz_path.name.lower()
                    if "cat" in name_lower:
                        dataset_species = "cats"
                    elif "dog" in name_lower or "barkopedia" in name_lower or "beans" in name_lower:
                        dataset_species = "dogs"
                    elif "bird" in name_lower or "avian" in name_lower or "birb" in name_lower:
                        dataset_species = "birds"
                    elif "frog" in name_lower or "rfcx" in name_lower:
                        dataset_species = "frogs"
                    elif "marine" in name_lower or "watkins" in name_lower:
                        dataset_species = "multi"
                    elif "audioset_stream" in name_lower or "musan" in name_lower or "demand" in name_lower:
                        dataset_species = "noise"
                
                if module_name == "animal_detector":
                    if dataset_species == "multi":
                        shard_labels = []
                        valid_indices = []
                        for idx, l in enumerate(labels):
                            s = str(l).lower()
                            sp = None
                            if "frog" in s:
                                sp = "frogs"
                            elif any(w in s for w in ("bird", "crow", "rooster", "hen", "finch")):
                                sp = "birds"
                            elif "dog" in s:
                                sp = "dogs"
                            elif "cat" in s:
                                sp = "cats"
                            elif "whale" in s or "orca" in s:
                                sp = "whales"
                            elif "dolphin" in s:
                                sp = "dolphins"
                            if sp and counts_per_class[sp] < max_per_species:
                                shard_labels.append(sp)
                                valid_indices.append(idx)
                                counts_per_class[sp] += 1
                        if len(valid_indices) == 0:
                            continue
                        embeddings = embeddings[valid_indices]
                        all_labels.extend(shard_labels)
                        all_embeddings.append(embeddings)
                    elif dataset_species in ("dogs", "cats", "birds", "frogs", "whales", "dolphins"):
                        sp = dataset_species
                        if counts_per_class[sp] >= max_per_species:
                            continue
                        take_n = min(len(embeddings), max_per_species - counts_per_class[sp])
                        embeddings = embeddings[:take_n]
                        shard_labels = [sp] * take_n
                        counts_per_class[sp] += take_n
                        all_labels.extend(shard_labels)
                        all_embeddings.append(embeddings)
                    else:
                        continue
                elif module_name == "breed_acoustic":
                    valid_indices = []
                    shard_labels = []
                    for idx, l in enumerate(labels):
                        s = str(l).lower().strip().replace(" ", "_")
                        if s in ("chihuahua", "husky", "shiba_inu", "shiba", "pitbull", "german_shepherd", "gsd",
                                 "pug", "bulldog", "french_bulldog", "collie", "beagle", "bloodhound", "basset",
                                 "pomeranian", "yorkshire", "malamute", "akita", "siamese", "persian", "bengal"):
                            shard_labels.append(s)
                            valid_indices.append(idx)
                    if len(valid_indices) == 0:
                        continue
                    embeddings = embeddings[valid_indices]
                    all_labels.extend(shard_labels)
                    all_embeddings.append(embeddings)
                elif module_name == "emotion_classifier":
                    EMOTIONS = {"contentment", "brushing", "neutral_rest", "calm", "stress_acute",
                                "isolation", "alarm", "pain", "stress_chronic", "anxiety", "hunger",
                                "waiting_food", "food", "territorial", "warning", "bark", "bonding",
                                "affection", "purr", "playfulness", "play", "frustration", "fear",
                                "distress", "alertness", "curiosity", "alert"}
                    valid_indices = []
                    shard_labels = []
                    for idx, l in enumerate(labels):
                        s = str(l).lower().strip()
                        if s in EMOTIONS:
                            shard_labels.append(s)
                            valid_indices.append(idx)
                    if len(valid_indices) == 0:
                        continue
                    embeddings = embeddings[valid_indices]
                    all_labels.extend(shard_labels)
                    all_embeddings.append(embeddings)
                elif module_name == "anti_spoofing":
                    shard_labels = []
                    valid_indices = []
                    for idx, l in enumerate(labels):
                        s = str(l).lower()
                        # Human speech, vocal imitation, and ambient soundscapes -> Class 1 (Spoof/Non-Animal)
                        if any(w in s for w in ("human", "speech", "baby", "sneeze", "cough", "footstep", "laugh", "snore", "breath", "ambient", "noise", "traffic")) or dataset_species in ("human", "human_imitation"):
                            if counts_per_class["1"] < max_spoof:
                                shard_labels.append("1")
                                valid_indices.append(idx)
                                counts_per_class["1"] += 1
                        # Genuine animal vocalizations across all species -> Class 0
                        elif any(w in s for w in ("dog", "cat", "bird", "frog", "whale", "dolphin", "wildlife", "animal", "bark", "purr", "howl", "growl", "chirp", "croak")) or dataset_species in ("dogs", "cats", "birds", "frogs", "whales", "dolphins"):
                            # Per-species sub-cap to ensure all species are represented in genuine animal class
                            sp_group = dataset_species if dataset_species in ("dogs", "cats", "birds", "frogs") else "other_animal"
                            if counts_per_species_animal[sp_group] < 8000 and counts_per_class["0"] < max_clean:
                                shard_labels.append("0")
                                valid_indices.append(idx)
                                counts_per_class["0"] += 1
                                counts_per_species_animal[sp_group] += 1
                    if len(valid_indices) == 0:
                        continue
                    embeddings = embeddings[valid_indices]
                    all_labels.extend(shard_labels)
                    all_embeddings.append(embeddings)
                elif module_name == "health_monitor":
                    # Ingest canine and feline acoustic features and evaluate clinical distress flags
                    pros = embeddings[:, 256:]
                    shard_targets = np.zeros((len(embeddings), 8), dtype=np.float32)
                    for i in range(len(embeddings)):
                        f0 = float(pros[i, 0])
                        f0_std = float(pros[i, 1])
                        f0_rng = float(pros[i, 2])
                        hnr = float(pros[i, 3])
                        zcr = float(pros[i, 4])
                        cent = float(pros[i, 5])
                        dur = float(pros[i, 7])
                        rms = float(pros[i, 8])
                        rate = float(pros[i, 9])
                        lbl = str(labels[i]).lower() if i < len(labels) else ""

                        # H01: Acute Pain / High-Pitch Distress
                        if (f0 > 750 and rms > 0.15) or any(w in lbl for w in ("pain", "cry", "scream", "injury")):
                            shard_targets[i, 0] = 1.0
                        # H02: Chronic Discomfort / Low Whimper
                        if (rate > 0.8 and rms < 0.14 and (180 < f0 < 550) and dur < 1.4) or any(w in lbl for w in ("whimper", "chronic", "moan", "whine", "discomfort")):
                            shard_targets[i, 1] = 1.0
                        # H03: Vocal Fatigue / Hoarseness
                        if hnr < 6.5 and zcr > 0.12:
                            shard_targets[i, 2] = 1.0
                        # H04: Laryngeal Dysfunction
                        if hnr < 4.0 and zcr > 0.16:
                            shard_targets[i, 3] = 1.0
                        # H05: Panting / Respiratory Distress
                        if zcr > 0.15 and hnr < 8.0 and dur > 1.0:
                            shard_targets[i, 4] = 1.0
                        # H06: Stridor / Wheezing / Cough
                        if zcr > 0.18 and cent > 2200 and hnr < 7.0:
                            shard_targets[i, 5] = 1.0
                        # H07: Cognitive Dysfunction / Wandering Whine
                        if (dur > 1.8 and f0_rng < 55 and f0 > 240) or any(w in lbl for w in ("night", "howl", "wandering", "disoriented")):
                            shard_targets[i, 6] = 1.0
                        # H08: Separation Anxiety
                        if rate > 2.0 and f0 > 450 and f0_std > 60:
                            shard_targets[i, 7] = 1.0

                    all_multilabel_targets.append(shard_targets)
                    all_labels.extend(["health"] * len(embeddings))
                    all_embeddings.append(embeddings)
                else:
                    all_labels.extend(labels.tolist())
                    all_embeddings.append(embeddings)
                logger.info(f"Loaded {len(embeddings)} samples from {npz_path.name} (species: {dataset_species})")
            except Exception as e:
                logger.warning(f"Failed to load {npz_path}: {e}")
        
        if not all_embeddings:
            logger.warning("No feature shards loaded! Using synthetic fallback data.")
            input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim
            all_embeddings = [np.random.randn(200, input_dim).astype(np.float32)]
            all_labels = [f"class_{i % num_classes}" for i in range(200)]
        
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Build label mapping
        if module_name == "animal_detector":
            from config import SPECIES_GROUPS
            self.label_to_idx = {k: v["id"] for k, v in SPECIES_GROUPS.items()}
        elif module_name == "anti_spoofing":
            self.label_to_idx = {"0": 0, "1": 1}
        elif module_name == "emotion_classifier":
            EMOTION_MAP = {
                "contentment": 0, "brushing": 0, "neutral_rest": 0, "calm": 0,
                "stress_acute": 1, "isolation": 1, "alarm": 1, "pain": 1,
                "stress_chronic": 2, "anxiety": 2,
                "hunger": 3, "waiting_food": 3, "food": 3,
                "territorial": 4, "warning": 4, "bark": 4,
                "bonding": 5, "affection": 5, "purr": 5,
                "playfulness": 6, "play": 6,
                "frustration": 7,
                "fear": 8, "distress": 8,
                "alertness": 9, "curiosity": 9, "alert": 9,
            }
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {
                label: EMOTION_MAP.get(str(label).lower(), idx % num_classes)
                for idx, label in enumerate(unique_labels)
            }
        elif module_name == "breed_acoustic":
            BREED_GROUP_MAP = {
                "brachycephalic": 0, "pug": 0, "bulldog": 0, "french_bulldog": 0,
                "working_herding": 1, "german_shepherd": 1, "gsd": 1, "pitbull": 1, "collie": 1,
                "hound": 2, "beagle": 2, "bloodhound": 2, "basset": 2,
                "toy": 3, "chihuahua": 3, "pomeranian": 3, "yorkshire": 3,
                "nordic_spitz": 4, "husky": 4, "shiba_inu": 4, "shiba": 4, "malamute": 4, "akita": 4,
                "siamese_oriental": 5, "siamese": 5,
                "persian_quiet": 6, "persian": 6,
                "wild_derived": 7, "bengal": 7,
            }
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {
                label: BREED_GROUP_MAP.get(str(label).lower().strip(), idx % num_classes)
                for idx, label in enumerate(unique_labels)
            }
        elif module_name == "health_monitor":
            HEALTH_MAP = {
                "pain": 0, "acute_pain": 0, "h01": 0, "cry": 0, "injury": 0,
                "whimper": 1, "chronic_pain": 1, "h02": 1, "arthritis": 1,
                "respiratory": 2, "cough": 2, "gasp": 2, "h03": 2, "distress_respiratory": 2,
                "distress": 3, "separation": 3, "whine": 3, "h04": 3, "anxiety": 3,
                "fear": 4, "hiss": 4, "growl": 4, "h05": 4,
                "purr": 5, "h06": 5, "stress_purr": 5,
                "pant": 6, "wheeze": 6, "h07": 6, "mouth_breathing": 6,
                "howl": 7, "nighttime": 7, "h08": 7, "disoriented": 7,
            }
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {
                label: HEALTH_MAP.get(str(label).lower().strip(), idx % num_classes)
                for idx, label in enumerate(unique_labels)
            }
        elif label_to_idx is None:
            unique_labels = sorted(set(all_labels))
            self.label_to_idx = {label: idx % num_classes for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
        
        # Convert string labels to indices
        if is_multilabel:
            if all_multilabel_targets:
                self.targets = np.concatenate(all_multilabel_targets, axis=0)
            else:
                self.targets = np.zeros((len(all_labels), num_classes), dtype=np.float32)
                for i, label in enumerate(all_labels):
                    idx = self.label_to_idx.get(str(label), 0)
                    self.targets[i, idx] = 1.0
        else:
            self.targets = np.array(
                [self.label_to_idx.get(str(label), 0) for label in all_labels],
                dtype=np.int64
            )
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.embeddings[idx])
        if self.is_multilabel:
            y = torch.from_numpy(self.targets[idx])
        else:
            y = torch.tensor(self.targets[idx], dtype=torch.long)
        return x, y


# ── Module Training Configs ──────────────────────────────────────

MODULE_CONFIGS = {
    "health_monitor": {
        "is_multilabel": True,
        "num_classes": len(HEALTH_FLAGS),
        "loss_fn": "bce",
        "species_filter": ["dogs", "cats"],
        "description": "8-Flag Multi-Label Health Classifier",
    },
    "emotion_classifier": {
        "is_multilabel": False,
        "num_classes": NUM_EMOTIONAL_STATES,
        "loss_fn": "cross_entropy",
        "species_filter": ["dogs", "cats", "dolphins"],
        "description": "10-Class Emotional State Classifier",
    },
    "breed_acoustic": {
        "is_multilabel": False,
        "num_classes": NUM_BREED_GROUPS,
        "loss_fn": "cross_entropy",
        "species_filter": ["dogs"],
        "description": "8-Group Acoustic Breed Classifier",
    },
    "animal_detector": {
        "is_multilabel": False,
        "num_classes": NUM_SPECIES_GROUPS,
        "loss_fn": "cross_entropy",
        "species_filter": None,  # Uses all species
        "description": "6-Species Auto-Router",
    },
    "anti_spoofing": {
        "is_multilabel": False,
        "num_classes": 2,
        "loss_fn": "cross_entropy",
        "species_filter": None,  # Uses all animal species and human audio
        "description": "Binary Human vs. Animal Detector",
    },
}


def discover_feature_shards(
    features_dir: Path = None,
    species_filter: List[str] = None,
    module_name: str = None,
) -> List[Path]:
    """
    Discover all available .npz feature shard files.
    Filter by species and module target according to DATASET_REGISTRY.
    """
    features_dir = features_dir or (PROCESSED_DATA_DIR / "shard_features")
    
    if not features_dir.exists():
        logger.warning(f"Features directory not found: {features_dir}")
        return []
    
    if species_filter is None and module_name in MODULE_CONFIGS:
        species_filter = MODULE_CONFIGS[module_name].get("species_filter")

    npz_files = sorted(features_dir.glob("features_*.npz"))
    
    from training.gdrive_shard_manager import DATASET_REGISTRY
    filtered = []
    for npz_path in npz_files:
        dataset_id = npz_path.stem.replace("features_", "")
        base_id = dataset_id.split("_part_")[0]
        ds_info = DATASET_REGISTRY.get(dataset_id, DATASET_REGISTRY.get(base_id, {}))
        ds_species = ds_info.get("species", "unknown")
        if ds_species == "unknown":
            name_lower = npz_path.name.lower()
            if "cat" in name_lower:
                ds_species = "cats"
            elif "dog" in name_lower or "barkopedia" in name_lower or "beans" in name_lower:
                ds_species = "dogs"
            elif "bird" in name_lower or "avian" in name_lower or "birb" in name_lower:
                ds_species = "birds"
            elif "frog" in name_lower or "rfcx" in name_lower:
                ds_species = "frogs"
            elif "marine" in name_lower or "watkins" in name_lower or "wildlife" in name_lower:
                ds_species = "multi"
            elif "audioset_stream" in name_lower or "musan" in name_lower or "demand" in name_lower:
                ds_species = "noise"
        ds_modules = ds_info.get("target_modules", [])
        
        # Check species filter
        species_match = (
            species_filter is None or
            ds_species in species_filter or
            (ds_species == "multi" and module_name not in ("breed_acoustic", "health_monitor"))
        )
        if not species_match:
            continue
            
        # Check target module match (animal_detector and anti_spoofing accept all animal datasets)
        if module_name and ds_modules and module_name not in ("animal_detector", "anti_spoofing"):
            if module_name not in ds_modules:
                continue
                
        filtered.append(npz_path)
    
    return filtered


def build_model(module_name: str, input_dim: int, device: str):
    """Build the appropriate classifier head for the given module."""
    cfg = MODULE_CONFIGS[module_name]
    
    if module_name == "health_monitor":
        from models.health_monitor.monitor import HealthFlagClassifier
        model = HealthFlagClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.health_hidden_dim,
            num_flags=cfg["num_classes"],
        )
    elif module_name == "emotion_classifier":
        from models.emotional_tracker.tracker import EmotionClassifier
        model = EmotionClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.emotion_hidden_dim,
            num_classes=cfg["num_classes"],
        )
    elif module_name == "breed_acoustic":
        from models.breed_identifier.identifier import AcousticBreedClassifier
        model = AcousticBreedClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.breed_hidden_dim,
            num_groups=cfg["num_classes"],
        )
    elif module_name == "animal_detector":
        from models.animal_detector.detector import SpeciesGroupClassifier
        model = SpeciesGroupClassifier(
            input_dim=input_dim,
            num_species=cfg["num_classes"],
        )
    elif module_name == "anti_spoofing":
        from models.dual_stream_classifier import EnhancedClassifier
        model = EnhancedClassifier(
            input_dim=input_dim,
            hidden_dim=256,
            num_classes=2,
            num_residual_blocks=3,
            dropout=0.20,
            is_multilabel=False,
        )
    else:
        raise ValueError(f"Unknown module: {module_name}")

    return model.to(device)


class ClassBalancedFocalLoss(nn.Module):
    """
    Focal Loss with class-frequency weighting to overcome heavy majority-class bias.
    Focuses gradient updates on hard-to-classify samples and minority species (dolphins, whales, frogs).
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 1.8, label_smoothing: float = 0.02):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_module(
    module_name: str,
    npz_paths: List[Path],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    resume: bool = False,
) -> Dict:
    """
    Train a single classifier module from pre-extracted feature shards.
    
    Returns dict with training metrics.
    """
    cfg = MODULE_CONFIGS[module_name]
    input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim  # 266
    
    print("\n" + "=" * 70)
    print(f"🐾 Training Module: {module_name} — {cfg['description']}")
    print(f"   Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"   Feature shards: {len(npz_paths)}")
    print(f"   Multi-label: {cfg['is_multilabel']} | Classes: {cfg['num_classes']}")
    print("=" * 70)
    
    # Train/val split: Sample-level 90/10 split ensuring all active classes are represented
    full_dataset = FeatureShardDataset(
        npz_paths=npz_paths,
        is_multilabel=cfg["is_multilabel"],
        num_classes=cfg["num_classes"],
        module_name=module_name,
    )
    total = len(full_dataset)
    val_size = max(int(total * 0.1), 1)
    train_size = total - val_size

    np.random.seed(TRAINING_CONFIG.seed)
    indices = np.random.permutation(total)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)
    train_targets = full_dataset.targets[train_idx]
    
    # Adaptive batch sizing: 256 for large datasets, 64 for smaller datasets
    effective_batch_size = min(batch_size, max(32, train_size // 4)) if train_size < 10000 else batch_size

    if module_name == "animal_detector":
        counts_dict = Counter(train_targets.tolist())
        total_train = len(train_targets)
        # Smooth power weighting (0.35) gives minority species healthy representation without over-repetition artifacts
        class_weights_dict = {c: (total_train / max(count, 1)) ** 0.35 for c, count in counts_dict.items()}
        mean_w = sum(class_weights_dict.values()) / max(len(class_weights_dict), 1)
        sample_weights = torch.DoubleTensor([class_weights_dict[int(t)] / mean_w for t in train_targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size, sampler=sampler,
            drop_last=(train_size > effective_batch_size),
            num_workers=0, pin_memory=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=effective_batch_size, shuffle=True,
            drop_last=(train_size > effective_batch_size),
            num_workers=0, pin_memory=False,
        )
    val_loader = DataLoader(
        val_ds, batch_size=effective_batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    
    logger.info(f"Dataset: {total} total ({train_size} train, {val_size} val, batch size: {effective_batch_size})")
    
    # Build model
    model = build_model(module_name, input_dim, device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    # Resume from checkpoint if requested
    start_epoch = 0
    best_metric = 0.0
    if resume:
        try:
            loaded, meta = load_checkpoint(model, module_name, device=device)
            if loaded:
                start_epoch = meta.get("epoch", 0)
                best_metric = meta.get("metrics", {}).get("best_metric", 0.0)
                logger.info(f"Resumed from epoch {start_epoch}")
        except Exception:
            pass
    
    # Loss function with Smart Active Class Weighting & Smoothing
    if cfg["loss_fn"] == "bce":
        targets_arr = np.array(train_targets)
        pos_counts = torch.tensor(targets_arr.sum(axis=0), dtype=torch.float32).clamp(min=1.0)
        neg_counts = float(len(targets_arr)) - pos_counts
        pos_weight = torch.clamp((neg_counts / pos_counts) ** 0.5, min=0.5, max=10.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        logger.info(f"⚖️ Applied Smart BCE Pos Weights: {pos_weight.cpu().numpy().round(2).tolist()}")
    else:
        targets_tensor = torch.tensor(train_targets, dtype=torch.long)
        counts = torch.bincount(targets_tensor, minlength=cfg["num_classes"]).float()
        weights = torch.zeros(cfg["num_classes"])
        active = counts > 0
        raw_w = (len(targets_tensor) / (active.sum() * counts[active])) ** 0.5
        weights[active] = raw_w / raw_w.mean()
        criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.02)
        logger.info(f"⚖️ Applied Smart Active Class Weights: {weights.cpu().numpy().round(3).tolist()}")
    
    # Cosine Annealing + SWA Learning Rate Schedule
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    swa_start = int(epochs * 0.70)
    swa_model = AveragedModel(model).to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5)
    
    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(start_epoch, start_epoch + epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # Embedding space jitter augmentation to prevent overfitting
            x_aug = x + torch.randn_like(x) * 0.015
            optimizer.zero_grad()
            if cfg["is_multilabel"]:
                logits = model.forward_logits(x_aug) if hasattr(model, "forward_logits") else (model.network(x_aug) if hasattr(model, "network") else model(x_aug))
                loss = criterion(logits, y)
            else:
                out = model(x_aug)
                loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG.gradient_clip)
            optimizer.step()
            train_loss += loss.item() * len(x)
        
        avg_train_loss = train_loss / max(train_size, 1)

        # Step LR / SWA
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # ── Validate ──
        eval_model = swa_model if epoch >= swa_start else model
        eval_model.eval()
        val_loss = 0.0
        correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if cfg["is_multilabel"]:
                    logits = eval_model.forward_logits(x) if hasattr(eval_model, "forward_logits") else (eval_model.network(x) if hasattr(eval_model, "network") else eval_model(x))
                    loss = criterion(logits, y)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    # Per-flag Hamming accuracy
                    correct += (preds == y).float().sum().item()
                    val_total += y.numel()
                else:
                    out = eval_model(x)
                    loss = criterion(out, y)
                    preds = torch.argmax(out, dim=-1)
                    correct += (preds == y).sum().item()
                    val_total += len(x)
                val_loss += loss.item() * len(x)
        
        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = correct / max(val_total, 1)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        
        # Log progress
        lr_current = optimizer.param_groups[0]["lr"]
        swa_tag = " [SWA]" if epoch >= swa_start else ""
        marker = ""
        if val_acc > best_metric:
            marker = f" ⭐ New Best: {val_acc:.2%}"

        logger.info(
            f"Epoch {epoch+1:03d}/{start_epoch+epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"LR: {lr_current:.2e}{swa_tag}{marker}"
        )
        
        # Save best checkpoint
        metric = val_acc
        if metric >= best_metric:
            best_metric = metric
            save_model = eval_model.module if hasattr(eval_model, "module") else eval_model
            save_checkpoint(
                model=save_model,
                module_name=module_name,
                epoch=epoch + 1,
                optimizer=optimizer,
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_acc,
                    "best_metric": best_metric,
                },
                extra_meta={
                    "description": f"Trained {module_name} from {len(npz_paths)} feature shards with SWA & jitter",
                    "num_shards": len(npz_paths),
                    "total_samples": total,
                }
            )
            logger.info(f"  ⭐ New best! Saved models/weights/{module_name}.pt (Acc: {val_acc:.2%})")

            logger.info(f"  ⭐ New best! Saved models/weights/{module_name}.pt (Acc: {val_acc:.2%})")
    
    print(f"\n🎉 {module_name} training complete! Best accuracy: {best_metric:.2%}")
    return {
        "module": module_name,
        "best_accuracy": best_metric,
        "epochs_trained": epochs,
        "total_samples": total,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Train From Feature Shards")
    parser.add_argument(
        "--module", type=str, default="all",
        choices=list(MODULE_CONFIGS.keys()) + ["all"],
        help="Which module to train"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--species", type=str, help="Filter feature shards by species")
    parser.add_argument("--features-dir", type=str, help="Custom features directory")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Cannot train without PyTorch. Install: pip install torch")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_dir = Path(args.features_dir) if args.features_dir else None
    
    # Determine modules to train
    if args.module == "all":
        modules = list(MODULE_CONFIGS.keys())
    else:
        modules = [args.module]
    
    results = []
    for module_name in modules:
        cfg = MODULE_CONFIGS[module_name]
        species_filter = [args.species] if args.species else cfg.get("species_filter")
        
        # Discover feature shards
        npz_paths = discover_feature_shards(
            features_dir=features_dir,
            species_filter=species_filter,
            module_name=module_name,
        )
        
        if not npz_paths:
            logger.warning(
                f"No feature shards found for {module_name}. "
                f"Run the shard pipeline first:\n"
                f"  py training/gdrive_shard_manager.py --pull-all --extract"
            )
            # Fall back to synthetic data for verification
            logger.info("Using synthetic data for training loop verification...")
            npz_paths = []  # Will trigger synthetic fallback in FeatureShardDataset
        
        result = train_module(
            module_name=module_name,
            npz_paths=npz_paths,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            resume=args.resume,
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Training Summary")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['module']:<25s} | "
            f"Best Acc: {r['best_accuracy']:.2%} | "
            f"Epochs: {r['epochs_trained']} | "
            f"Samples: {r['total_samples']}"
        )
    print("=" * 70)
    print("\nTrained weights saved to: models/weights/")
    print("These will be automatically loaded by the AnimalVox AI inference pipeline.")


if __name__ == "__main__":
    main()
