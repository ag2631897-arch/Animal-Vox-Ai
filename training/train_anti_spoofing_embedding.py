"""
AnimalVox AI — Train Embedding-Based Anti-Spoofing Classifier
================================================================
Trains a lightweight MLP classifier on frozen BEATs embeddings to
distinguish real animal sounds from human imitations, speech, and noise.

5 classes:
    0: real_animal       — Genuine animal vocalization
    1: human_imitation   — Human imitating an animal sound
    2: human_speech      — Normal conversational speech
    3: other_human       — Cough, laugh, clap, whistle, etc.
    4: noise             — Music, background noise, silence

Data layout (create this directory structure):
    data/anti_spoofing/
        real_animal/          # Real animal recordings (phone-recorded pets, dataset clips)
            dog_bark_001.wav
            cat_meow_phone_002.wav
            ...
        human_imitation/      # Humans imitating animals (>=10 different people, on phones)
            person01_bark.wav
            person02_meow.wav
            ...
        human_speech/         # Normal human talking
            speech_001.wav
            conversation_002.wav
            ...
        other_human/          # Coughs, laughs, claps, throat clearing
            cough_001.wav
            laugh_002.wav
            ...
        noise/                # Music, ambient noise, silence
            traffic_noise_001.wav
            silence_002.wav
            ...

Requirements:
    - A few hundred clips per class (200+ recommended)
    - Record real animals ON PHONES (not just dataset audio) to avoid channel shortcuts
    - Collect imitations from >=10 people on different phones
    - Split train/test by SPEAKER and RECORDING SESSION, never by clip

Usage:
    # Step 1: Collect data into data/anti_spoofing/{class_name}/
    # Step 2: Train
    python training/train_anti_spoofing_embedding.py

    # Step 3: Evaluate on held-out data
    python training/train_anti_spoofing_embedding.py --evaluate

    # Step 4: Run 30-minute diagnostic
    python training/train_anti_spoofing_embedding.py --diagnose --audio-dir path/to/test_clips/
"""

import sys
import os
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AntiSpoofTrainer")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    logger.error("PyTorch required. pip install torch")
    sys.exit(1)

try:
    import librosa
except ImportError:
    logger.error("librosa required. pip install librosa")
    sys.exit(1)

from config import AUDIO_CONFIG, PROJECT_ROOT as PROJ_ROOT
from models.anti_spoofing.embedding_classifier import (
    SegmentEmbeddingClassifier, CLASS_NAMES, NUM_CLASSES
)
from training.feature_extractor import extract_beats_embedding


# ── Constants ──────────────────────────────────────────────────────
DATA_DIR = PROJ_ROOT / "data" / "anti_spoofing"
SEGMENT_DURATION = 1.0  # seconds
SEGMENT_HOP = 0.5       # seconds
MIN_RMS = 0.012

DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 32
DEFAULT_WEIGHT_DECAY = 1e-4
TARGET_ANIMAL_FPR = 0.02  # Target: <2% of real animals falsely flagged


# ── Dataset ────────────────────────────────────────────────────────
class AntiSpoofEmbeddingDataset(Dataset):
    """
    Dataset of pre-extracted BEATs embeddings for anti-spoofing training.
    Each sample is a 256-dim embedding + class label.
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ── Data Extraction ────────────────────────────────────────────────
def extract_dataset(
    data_dir: Path,
    sr: int = 16000,
    segment_duration: float = SEGMENT_DURATION,
    segment_hop: float = SEGMENT_HOP,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Walk data_dir/{class_name}/ directories, extract BEATs embeddings
    for each 1-second segment, and return (embeddings, labels, filenames).
    """
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    seg_samples = int(segment_duration * sr)
    hop_samples = int(segment_hop * sr)

    all_embeddings = []
    all_labels = []
    all_files = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        files = sorted([
            f for f in class_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in audio_exts
        ])
        logger.info(f"  {class_name}: {len(files)} files")

        for fpath in files:
            try:
                audio, _ = librosa.load(str(fpath), sr=sr, mono=True, duration=10.0)
                if len(audio) < int(0.2 * sr):
                    continue

                # Normalize
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak

                # Segment into 1-second windows
                start = 0
                while start + seg_samples <= len(audio):
                    seg = audio[start: start + seg_samples]
                    rms = float(np.sqrt(np.mean(seg ** 2)))
                    if rms >= MIN_RMS:
                        emb = extract_beats_embedding(seg, sr=sr, embed_dim=256)
                        all_embeddings.append(emb)
                        all_labels.append(class_id)
                        all_files.append(f"{class_name}/{fpath.name}@{start/sr:.1f}s")
                    start += hop_samples

                # Handle short files (< 1 second)
                if len(audio) < seg_samples:
                    padded = np.zeros(seg_samples, dtype=audio.dtype)
                    padded[:len(audio)] = audio
                    rms = float(np.sqrt(np.mean(audio ** 2)))
                    if rms >= MIN_RMS:
                        emb = extract_beats_embedding(padded, sr=sr, embed_dim=256)
                        all_embeddings.append(emb)
                        all_labels.append(class_id)
                        all_files.append(f"{class_name}/{fpath.name}@full")

            except Exception as e:
                logger.debug(f"Failed on {fpath.name}: {e}")

    if not all_embeddings:
        return np.array([]), np.array([]), []

    return (
        np.array(all_embeddings, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
        all_files,
    )


# ── Speaker-Aware Train/Val Split ──────────────────────────────────
def split_by_speaker(
    filenames: List[str], labels: np.ndarray, val_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split by source file (proxy for speaker/session) so the same recording
    never appears in both train and val. This prevents optimistic accuracy.
    """
    # Group indices by source file (before @segment marker)
    file_to_indices: Dict[str, List[int]] = {}
    for idx, fname in enumerate(filenames):
        base = fname.rsplit("@", 1)[0]  # e.g. "human_imitation/person01_bark.wav"
        if base not in file_to_indices:
            file_to_indices[base] = []
        file_to_indices[base].append(idx)

    # Sort files by name for reproducibility, then split
    sorted_files = sorted(file_to_indices.keys())
    np.random.seed(42)
    np.random.shuffle(sorted_files)

    n_val = max(1, int(len(sorted_files) * val_ratio))
    val_files = set(sorted_files[:n_val])

    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)

    for fname, indices in file_to_indices.items():
        if fname in val_files:
            val_mask[indices] = True
        else:
            train_mask[indices] = True

    return train_mask, val_mask


# ── Training ───────────────────────────────────────────────────────
def train(
    data_dir: Path = DATA_DIR,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    device: str = "cpu",
):
    """Train the embedding classifier and save weights."""
    logger.info("=" * 64)
    logger.info("AnimalVox AI — Anti-Spoofing Embedding Classifier Training")
    logger.info("=" * 64)

    # Step 1: Check data availability
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Create the following directory structure:")
        for cls in CLASS_NAMES:
            logger.info(f"  {data_dir / cls}/")
        logger.info("Then add audio files (WAV/MP3/FLAC) to each class folder.")
        logger.info("")
        logger.info("Recommended data collection:")
        logger.info("  real_animal/:     200+ clips of real animals recorded on phones")
        logger.info("  human_imitation/: 200+ clips of humans imitating animals (10+ people)")
        logger.info("  human_speech/:    200+ clips of normal conversation")
        logger.info("  other_human/:     100+ clips of coughs, laughs, claps")
        logger.info("  noise/:           100+ clips of music, ambient noise, silence")
        return

    # Step 2: Extract embeddings
    logger.info(f"Extracting BEATs embeddings from {data_dir}...")
    t0 = time.time()
    embeddings, labels, filenames = extract_dataset(data_dir)
    elapsed = time.time() - t0

    if len(embeddings) == 0:
        logger.error("No samples extracted. Check that audio files exist in class subdirectories.")
        return

    logger.info(f"Extracted {len(embeddings)} segments in {elapsed:.1f}s")
    for i, cls in enumerate(CLASS_NAMES):
        count = int(np.sum(labels == i))
        logger.info(f"  {cls}: {count} segments")

    # Step 3: Speaker-aware split
    train_mask, val_mask = split_by_speaker(filenames, labels)
    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val, y_val = embeddings[val_mask], labels[val_mask]

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    if len(X_val) == 0:
        logger.warning("Validation set is empty — using 20% of training data (not speaker-split)")
        n_val = max(1, len(X_train) // 5)
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train, y_train = X_train[n_val:], y_train[n_val:]

    # Step 4: Create datasets
    train_ds = AntiSpoofEmbeddingDataset(X_train, y_train)
    val_ds = AntiSpoofEmbeddingDataset(X_val, y_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Step 5: Model + optimizer
    model = SegmentEmbeddingClassifier(embed_dim=256, num_classes=NUM_CLASSES, hidden_dim=128)
    model.to(device)

    # Class-balanced loss weights
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_CLASSES
    class_weights = torch.from_numpy(weights).float().to(device)
    logger.info(f"Class weights: {dict(zip(CLASS_NAMES, weights.round(2).tolist()))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Step 6: Training loop
    best_val_acc = 0.0
    best_state = None
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_dl:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_dl:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(y_batch)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(y_batch.cpu().numpy())

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        # Compute animal false positive rate
        # (real_animal samples predicted as human_imitation or other human class)
        all_val_preds = np.array(all_val_preds)
        all_val_labels = np.array(all_val_labels)

        animal_mask = all_val_labels == 0  # class 0 = real_animal
        if animal_mask.sum() > 0:
            animal_flagged = np.isin(all_val_preds[animal_mask], [1, 2, 3]).sum()
            animal_fpr = float(animal_flagged) / float(animal_mask.sum())
        else:
            animal_fpr = 0.0

        # Imitation catch rate
        imitation_mask = all_val_labels == 1  # class 1 = human_imitation
        if imitation_mask.sum() > 0:
            imitation_caught = (all_val_preds[imitation_mask] == 1).sum()
            imitation_catch_rate = float(imitation_caught) / float(imitation_mask.sum())
        else:
            imitation_catch_rate = 0.0

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.3f} | "
                f"Animal FPR: {animal_fpr:.3f} | Imitation Catch: {imitation_catch_rate:.3f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "val_accuracy": float(val_acc),
                "train_accuracy": float(train_acc),
                "animal_false_positive_rate": float(animal_fpr),
                "imitation_catch_rate": float(imitation_catch_rate),
                "val_loss": float(avg_val_loss),
                "epoch": epoch,
            }

    # Step 7: Save best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

        try:
            from models.weight_manager import save_checkpoint
            save_path = save_checkpoint(
                model, "anti_spoofing_embedding",
                epoch=best_metrics.get("epoch", epochs),
                metrics=best_metrics,
            )
            logger.info(f"✅ Saved best checkpoint to {save_path}")
        except Exception as e:
            # Fallback: save directly
            save_dir = PROJ_ROOT / "models" / "weights"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "anti_spoofing_embedding.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "metrics": best_metrics,
                "class_names": CLASS_NAMES,
            }, save_path)
            logger.info(f"✅ Saved best checkpoint to {save_path}")

        logger.info("=" * 64)
        logger.info(f"Best Val Accuracy:           {best_metrics['val_accuracy']:.1%}")
        logger.info(f"Animal False Positive Rate:  {best_metrics['animal_false_positive_rate']:.1%}")
        logger.info(f"Imitation Catch Rate:        {best_metrics['imitation_catch_rate']:.1%}")
        logger.info(f"Target Animal FPR:           <{TARGET_ANIMAL_FPR:.0%}")

        if best_metrics["animal_false_positive_rate"] > TARGET_ANIMAL_FPR:
            logger.warning(
                f"⚠ Animal FPR ({best_metrics['animal_false_positive_rate']:.1%}) exceeds "
                f"target ({TARGET_ANIMAL_FPR:.0%}). Collect more real animal clips "
                f"recorded on phones to reduce channel shortcut."
            )
        logger.info("=" * 64)
    else:
        logger.error("No improvement during training. Check your data.")


# ── Diagnostic Mode ────────────────────────────────────────────────
def run_diagnostic(audio_dir: Path, sr: int = 16000):
    """
    30-minute diagnostic protocol.
    Logs per-clip detector scores across 4 buckets for comparison.
    """
    from models.anti_spoofing.embedding_classifier import SegmentWiseAntiSpoof

    logger.info("=" * 64)
    logger.info("AnimalVox AI — Anti-Spoofing Diagnostic")
    logger.info("=" * 64)

    analyzer = SegmentWiseAntiSpoof(device="cpu")

    if not analyzer.has_trained_weights:
        logger.warning("Embedding classifier not trained yet. Train first with:")
        logger.warning("  python training/train_anti_spoofing_embedding.py")
        logger.info("Diagnostic will still run but all predictions will be 'uncertain'.")

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}
    results_by_bucket = {}

    for bucket_dir in sorted(audio_dir.iterdir()):
        if not bucket_dir.is_dir():
            continue
        bucket_name = bucket_dir.name
        results_by_bucket[bucket_name] = []

        files = sorted([f for f in bucket_dir.iterdir() if f.suffix.lower() in audio_exts])
        logger.info(f"\nBucket: {bucket_name} ({len(files)} clips)")

        for fpath in files:
            try:
                audio, _ = librosa.load(str(fpath), sr=sr, mono=True, duration=10.0)
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak

                diag = analyzer.diagnose(audio, sr=sr)
                results_by_bucket[bucket_name].append({
                    "file": fpath.name,
                    **diag,
                })

                logger.info(
                    f"  {fpath.name:40s} → {diag['outcome']:10s} "
                    f"(animal={diag['animal_ratio']:.2f} imit={diag['imitation_ratio']:.2f} "
                    f"speech={diag.get('human_ratio', 0):.2f})"
                )
            except Exception as e:
                logger.warning(f"  {fpath.name}: ERROR — {e}")

    # Summary
    logger.info("\n" + "=" * 64)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 64)
    for bucket, results in results_by_bucket.items():
        if not results:
            continue
        outcomes = [r["outcome"] for r in results]
        animal_ct = outcomes.count("animal")
        human_ct = outcomes.count("human")
        uncertain_ct = outcomes.count("uncertain")
        logger.info(
            f"  {bucket:30s}: {len(results)} clips → "
            f"animal={animal_ct} human={human_ct} uncertain={uncertain_ct}"
        )

    # Save diagnostic JSON
    diag_path = audio_dir / "diagnostic_results.json"
    with open(diag_path, "w") as f:
        json.dump(results_by_bucket, f, indent=2)
    logger.info(f"\nDetailed results saved to {diag_path}")
    logger.info(
        "\nInterpretation:\n"
        "  - If phone-animal bucket has high 'human' count → channel shortcut\n"
        "  - If background-speech bucket has high 'human' count → clip-level logic issue\n"
        "  - If imitation bucket has high 'animal' count → features too weak, need more data"
    )


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or evaluate the embedding-based anti-spoofing classifier"
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory with class subdirectories")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate only (no training)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run 30-minute diagnostic protocol")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Audio directory for diagnostic mode")

    args = parser.parse_args()

    if args.diagnose:
        diag_dir = Path(args.audio_dir) if args.audio_dir else Path(args.data_dir)
        run_diagnostic(diag_dir)
    else:
        train(
            data_dir=Path(args.data_dir),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
