"""
AnimalVox AI — Multi-Task Intelligence Classifiers Training Pipeline
=====================================================================
Trains and fine-tunes the modular classification heads:
1. health_monitor: 8-Flag multi-label veterinary health classifier
2. emotion_classifier: 10-Class discrete emotional state classifier
3. breed_acoustic: 8-Group acoustic breed classifier
4. animal_detector: 6-Species auto-routing classifier

Usage:
    python training/train_classifiers.py --module health_monitor --epochs 25
    python training/train_classifiers.py --module emotion_classifier --epochs 30
    python training/train_classifiers.py --module breed_acoustic --epochs 20
    python training/train_classifiers.py --module all
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch or numpy not found. Please install: pip install torch numpy")

from config import (
    MODEL_CONFIG, HEALTH_FLAGS, NUM_EMOTIONAL_STATES,
    NUM_BREED_GROUPS, NUM_SPECIES_GROUPS
)
from models.weight_manager import save_checkpoint, load_checkpoint


class SyntheticEmbeddingDataset(Dataset):
    """
    Dataset for classifier head training.
    Generates or loads (256-dim fused embedding + 10-dim prosodic features = 266-dim input).
    """
    def __init__(self, num_samples: int = 500, input_dim: int = 266, num_classes: int = 10, is_multilabel: bool = False):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.is_multilabel = is_multilabel
        
        # Synthetic feature distribution for testing
        self.data = torch.randn(num_samples, input_dim)
        if is_multilabel:
            self.labels = (torch.rand(num_samples, num_classes) > 0.75).float()
        else:
            self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_single_module(module_name: str, epochs: int = 20, batch_size: int = 16, lr: float = 1e-3, device: str = "cpu"):
    """Train a specific intelligence classifier head."""
    print("\n" + "=" * 64)
    print(f"🐾 Training Intelligence Module: {module_name} 🐾")
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 64)

    input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim  # 266

    if module_name == "health_monitor":
        from models.health_monitor.monitor import HealthFlagClassifier
        model = HealthFlagClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.health_hidden_dim,
            num_flags=len(HEALTH_FLAGS)
        ).to(device)
        criterion = nn.BCELoss()
        dataset = SyntheticEmbeddingDataset(num_samples=600, input_dim=input_dim, num_classes=len(HEALTH_FLAGS), is_multilabel=True)
        is_multilabel = True

    elif module_name == "emotion_classifier":
        from models.emotional_tracker.tracker import EmotionClassifier
        model = EmotionClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.emotion_hidden_dim,
            num_classes=NUM_EMOTIONAL_STATES
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        dataset = SyntheticEmbeddingDataset(num_samples=600, input_dim=input_dim, num_classes=NUM_EMOTIONAL_STATES, is_multilabel=False)
        is_multilabel = False

    elif module_name == "breed_acoustic":
        from models.breed_identifier.identifier import AcousticBreedClassifier
        model = AcousticBreedClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.breed_hidden_dim,
            num_groups=NUM_BREED_GROUPS
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        dataset = SyntheticEmbeddingDataset(num_samples=500, input_dim=input_dim, num_classes=NUM_BREED_GROUPS, is_multilabel=False)
        is_multilabel = False

    else:
        logger.error(f"Unknown module '{module_name}'. Choose from: health_monitor, emotion_classifier, breed_acoustic")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x)

        avg_loss = epoch_loss / len(dataset)
        logger.info(f"[{module_name}] Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model=model,
                module_name=module_name,
                epoch=epoch + 1,
                optimizer=optimizer,
                metrics={"loss": avg_loss, "best_loss": best_loss},
                extra_meta={"description": f"Trained {module_name} classifier checkpoint"}
            )

    print(f"⭐ Saved trained checkpoint: models/weights/{module_name}.pt")


def main():
    parser = argparse.ArgumentParser(description="Train AnimalVox AI Classifier Modules")
    parser.add_argument(
        "--module", type=str, default="all",
        choices=["health_monitor", "emotion_classifier", "breed_acoustic", "all"],
        help="Module to train"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Cannot run training without PyTorch. Install via: pip install torch")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.module == "all":
        for mod in ["health_monitor", "emotion_classifier", "breed_acoustic"]:
            train_single_module(mod, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    else:
        train_single_module(args.module, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)


if __name__ == "__main__":
    main()
