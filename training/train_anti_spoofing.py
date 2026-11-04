"""
AnimalVox AI — Anti-Spoofing Model Training Pipeline
=====================================================
Trains the ECAPA-TDNN neural detector to classify:
- Class 0: Human vocal tract imitation (spoof attempt)
- Class 1: Genuine animal vocalization

Dataset Structure:
    data/raw/
        human_speech/   (e.g., LibriSpeech, VoxCeleb, or recorded human animal sounds)
        animal_audio/   (e.g., ESC-50 animal classes, AudioSet dogs/cats/birds)

Usage:
    python training/train_anti_spoofing.py --epochs 30 --batch-size 16 --lr 0.0002
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
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
    import librosa
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch, librosa, or numpy not found. Please install requirements: pip install torch librosa numpy")

from config import ANTI_SPOOFING_CONFIG, AUDIO_CONFIG
from models.anti_spoofing.detector import HumanImitationDetector
from models.weight_manager import save_checkpoint, load_checkpoint, get_checkpoint_path


class AudioSpoofDataset(Dataset):
    """
    Dataset loader for binary anti-spoofing training.
    Extracts 128-band Mel-spectrograms at 16kHz.
    """
    def __init__(self, file_paths: List[Path], labels: List[int], sr: int = 16000, duration: float = 3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.max_len = int(sr * duration)
        self.n_mels = AUDIO_CONFIG.n_mels
        self.n_fft = AUDIO_CONFIG.n_fft
        self.hop_length = AUDIO_CONFIG.hop_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            audio, _ = librosa.load(str(file_path), sr=self.sr, mono=True)
            if len(audio) < self.max_len:
                audio = np.pad(audio, (0, self.max_len - len(audio)))
            else:
                audio = audio[:self.max_len]
        except Exception:
            # Fallback for silent/corrupt files
            audio = np.zeros(self.max_len, dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        tensor = torch.from_numpy(mel_db).float()  # (n_mels, time)
        return tensor, torch.tensor(label, dtype=torch.long)


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: str) -> Tuple[float, float]:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for mels, labels in dataloader:
        mels = mels.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(mels)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train Anti-Spoofing ECAPA-TDNN Detector")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint in weights/")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("Cannot run training without PyTorch. Install via: pip install torch librosa")
        return

    print("=" * 64)
    print("🐾 AnimalVox AI — Training Anti-Spoofing Neural Detector 🐾")
    print(f"Device: {args.device} | Epochs: {args.epochs} | LR: {args.lr}")
    print("=" * 64)

    # Initialize model
    model = HumanImitationDetector(
        n_mels=AUDIO_CONFIG.n_mels,
        embed_dim=ANTI_SPOOFING_CONFIG.ecapa_embed_dim
    ).to(args.device)

    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        loaded, meta = load_checkpoint(model, "anti_spoofing_ecapa", device=args.device)
        if loaded:
            start_epoch = meta.get("epoch", 0) + 1
            best_acc = meta.get("metrics", {}).get("accuracy", 0.0)
            logger.info(f"Resumed from epoch {start_epoch} (Previous best accuracy: {best_acc:.2%})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Scan dataset directories
    human_dir = PROJECT_ROOT / "data" / "raw" / "human_speech"
    animal_dir = PROJECT_ROOT / "data" / "raw" / "animal_audio"

    human_files = list(human_dir.glob("*.wav")) if human_dir.exists() else []
    animal_files = list(animal_dir.glob("*.wav")) if animal_dir.exists() else []

    all_files = human_files + animal_files
    # Label 0: Human, Label 1: Animal
    all_labels = [0] * len(human_files) + [1] * len(animal_files)

    if len(all_files) < 10:
        logger.warning(
            f"Found only {len(all_files)} audio samples ({len(human_files)} human, {len(animal_files)} animal).\n"
            f"Please place audio files in:\n"
            f"  • {human_dir}\n"
            f"  • {animal_dir}\n"
            f"Generating synthetic demonstration data so the training loop can be verified..."
        )
        # Create small synthetic test batch for demonstration
        synth_dir = PROJECT_ROOT / "data" / "raw" / "synthetic_demo"
        synth_dir.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sr = 16000
        for i in range(10):
            # Human speech proxy (500Hz harmonic)
            t = np.linspace(0, 1.0, sr)
            wav_h = 0.3 * np.sin(2 * np.pi * 500 * t)
            p_h = synth_dir / f"human_demo_{i}.wav"
            sf.write(str(p_h), wav_h, sr)
            human_files.append(p_h)

            # Animal bark proxy (150Hz modulated)
            wav_a = 0.4 * np.sin(2 * np.pi * 180 * t) * np.exp(-3 * t)
            p_a = synth_dir / f"animal_demo_{i}.wav"
            sf.write(str(p_a), wav_a, sr)
            animal_files.append(p_a)

        all_files = human_files + animal_files
        all_labels = [0] * len(human_files) + [1] * len(animal_files)

    dataset = AudioSpoofDataset(all_files, all_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    logger.info(f"Starting training on {len(all_files)} samples...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        loss, acc = train_epoch(model, dataloader, optimizer, criterion, args.device)
        scheduler.step()

        logger.info(f"Epoch {epoch+1:02d}/{start_epoch + args.epochs} | Loss: {loss:.4f} | Acc: {acc:.2%}")

        # Checkpoint saving via weight_manager
        if acc >= best_acc:
            best_acc = acc
            save_checkpoint(
                model=model,
                module_name="anti_spoofing_ecapa",
                epoch=epoch + 1,
                optimizer=optimizer,
                metrics={"loss": loss, "accuracy": acc, "best_acc": best_acc},
                extra_meta={"description": "Trained ECAPA-TDNN anti-spoofing detector"}
            )
            logger.info(f"⭐ Saved new best checkpoint to models/weights/anti_spoofing_ecapa.pt (Acc: {acc:.2%})")

    print("\n" + "=" * 64)
    print("🎉 Training Complete!")
    print(f"Best Accuracy: {best_acc:.2%}")
    print("Checkpoint saved to models/weights/anti_spoofing_ecapa.pt")
    print("AnimalVox AI will now automatically use this TRAINED neural model!")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
