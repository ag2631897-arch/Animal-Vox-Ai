"""
AnimalVox AI — Bootstrap Anti-Spoofing v3 (Binary, Clean Data Only)
====================================================================
Binary classifier: non_human (0) vs human (1).
Uses ONLY clean data sources — excludes noisy AudioSet animal clips.
ROC-calibrated threshold for <2% animal FPR.

Clean sources (0-6% human overlap):
    DogSpeak (husky, chihuahua), cats_isolated, cats_stephen, catmeows,
    xenocanto birds/frogs, rfcx frogs, watkins marine, birdclef,
    ESC-50 non-human, MUSAN noise, AudioSet ambient_anti_spoof

EXCLUDED (70-84% human overlap due to YouTube background speech):
    AudioSet stream animal labels (dogs, cats, birds, frogs, wildlife)
"""

import sys, os, glob, time, logging
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BootstrapV3")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.anti_spoofing.embedding_classifier import (
    SegmentEmbeddingClassifier, CLASS_NAMES, NUM_CLASSES, EMBED_DIM
)

SHARD_DIR = PROJECT_ROOT / "data" / "processed" / "shard_features"

# Clean animal/noise file prefixes — ALL samples from these are non_human
CLEAN_PREFIXES = [
    "features_dogspeak_",
    "features_cats_isolated_",
    "features_cats_stephen_",
    "features_catmeows",
    "features_xenocanto_",
    "features_rfcx_",
    "features_watkins_",
    "features_birdclef_",
    "features_earth_species_",
]

# Files that need label-based filtering
AUDIOSET_STREAM_PREFIX = "features_audioset_stream_"
ESC50_FILE = "features_esc50.npz"
MUSAN_PREFIX = "features_musan_"
# EXCLUDED: features_audioset_animals.npz, features_wildlife.npz (noisy)

EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 512
WEIGHT_DECAY = 5e-4
VAL_RATIO = 0.15
TARGET_FPR = 0.02


def load_clean_data():
    """Load only clean data sources."""
    all_npz = sorted(glob.glob(str(SHARD_DIR / "*.npz")))
    logger.info(f"Found {len(all_npz)} total NPZ files")

    nonhuman_embs, nonhuman_srcs = [], []
    human_embs, human_srcs = [], []

    stats = {"clean_animal": 0, "noise": 0, "human": 0, "skipped": 0}

    for fpath in all_npz:
        fname = os.path.basename(fpath)
        try:
            d = np.load(fpath, allow_pickle=True)
            embeddings = d["embeddings"]
            labels = d["labels"]

            is_clean = any(fname.startswith(p) for p in CLEAN_PREFIXES)
            is_audioset_stream = fname.startswith(AUDIOSET_STREAM_PREFIX)
            is_esc50 = fname == ESC50_FILE
            is_musan = fname.startswith(MUSAN_PREFIX)

            for i, lab in enumerate(labels):
                lab = str(lab)
                emb = embeddings[i].astype(np.float32)

                # Ensure 266 dims
                if len(emb) < EMBED_DIM:
                    padded = np.zeros(EMBED_DIM, dtype=np.float32)
                    padded[:len(emb)] = emb
                    emb = padded
                elif len(emb) > EMBED_DIM:
                    emb = emb[:EMBED_DIM]

                # L2-normalize
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm

                if is_clean:
                    # All samples from clean sources → non_human
                    nonhuman_embs.append(emb)
                    nonhuman_srcs.append(fname)
                    stats["clean_animal"] += 1

                elif is_audioset_stream:
                    if lab == "human_anti_spoof":
                        human_embs.append(emb)
                        human_srcs.append(fname)
                        stats["human"] += 1
                    elif lab == "ambient_anti_spoof":
                        nonhuman_embs.append(emb)
                        nonhuman_srcs.append(fname)
                        stats["noise"] += 1
                    else:
                        stats["skipped"] += 1  # Skip noisy animal labels

                elif is_esc50:
                    if lab == "human_anti_spoof":
                        human_embs.append(emb)
                        human_srcs.append(fname)
                        stats["human"] += 1
                    else:
                        # ESC-50 is curated — all non-human labels are clean
                        nonhuman_embs.append(emb)
                        nonhuman_srcs.append(fname)
                        stats["clean_animal"] += 1

                elif is_musan:
                    nonhuman_embs.append(emb)
                    nonhuman_srcs.append(fname)
                    stats["noise"] += 1

                else:
                    # Skip unknown/noisy sources (audioset_animals, wildlife)
                    stats["skipped"] += 1

        except Exception as e:
            logger.debug(f"Failed {fname}: {e}")

    logger.info(f"Data loaded:")
    logger.info(f"  Clean animal: {stats['clean_animal']}")
    logger.info(f"  Noise:        {stats['noise']}")
    logger.info(f"  Human:        {stats['human']}")
    logger.info(f"  Skipped:      {stats['skipped']} (noisy AudioSet animals)")
    logger.info(f"  Non-human total: {len(nonhuman_embs)}")
    logger.info(f"  Human total:     {len(human_embs)}")

    return nonhuman_embs, nonhuman_srcs, human_embs, human_srcs


def prepare_data(nh_embs, nh_srcs, h_embs, h_srcs):
    """Concatenate, balance, split by source file."""
    # Cap non_human to 3x human to prevent extreme imbalance
    max_nh = min(len(nh_embs), len(h_embs) * 3)
    if len(nh_embs) > max_nh:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(nh_embs), max_nh, replace=False)
        nh_embs = [nh_embs[i] for i in idx]
        nh_srcs = [nh_srcs[i] for i in idx]
    logger.info(f"After capping: non_human={len(nh_embs)}, human={len(h_embs)}")

    X = np.array(nh_embs + h_embs, dtype=np.float32)
    y = np.array([0]*len(nh_embs) + [1]*len(h_embs), dtype=np.int64)
    srcs = nh_srcs + h_srcs

    # Split by source file
    unique = sorted(set(srcs))
    rng = np.random.RandomState(42)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * VAL_RATIO))
    val_set = set(unique[:n_val])

    train_mask = np.array([s not in val_set for s in srcs])
    val_mask = ~train_mask

    rng2 = np.random.RandomState(999)
    t_perm = rng2.permutation(train_mask.sum())
    v_perm = rng2.permutation(val_mask.sum())

    return (X[train_mask][t_perm], y[train_mask][t_perm],
            X[val_mask][v_perm], y[val_mask][v_perm])


class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def calibrate_threshold(model, X_val, y_val, device, target_fpr=TARGET_FPR):
    """Find threshold on P(human) that gives target FPR on animal (non_human) samples."""
    model.eval()
    dl = DataLoader(DS(X_val, y_val), batch_size=1024, shuffle=False)

    all_probs = []
    all_true = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs[:, 1])  # P(human)
            all_true.append(yb.numpy())

    p_human = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    # Animal (non_human) samples
    animal_mask = y_true == 0
    animal_scores = p_human[animal_mask]
    human_scores = p_human[y_true == 1]

    # Find threshold where FPR (animal classified as human) = target_fpr
    # FPR = fraction of animal samples with P(human) > threshold
    thresholds = np.linspace(0.01, 0.99, 1000)
    best_thresh = 0.50
    best_catch = 0.0

    for t in thresholds:
        fpr = (animal_scores > t).mean()
        catch = (human_scores > t).mean()
        if fpr <= target_fpr and catch > best_catch:
            best_catch = catch
            best_thresh = float(t)

    # If no threshold achieves target FPR, use the one with lowest FPR
    if best_catch == 0.0:
        for t in thresholds[::-1]:
            fpr = (animal_scores > t).mean()
            if fpr <= target_fpr * 2:  # Relax to 2x target
                best_thresh = float(t)
                best_catch = (human_scores > t).mean()
                break

    final_fpr = (animal_scores > best_thresh).mean()
    final_catch = (human_scores > best_thresh).mean()

    logger.info(f"\nROC Calibration:")
    logger.info(f"  Threshold:  {best_thresh:.3f}")
    logger.info(f"  Animal FPR: {final_fpr:.4f} (target <{target_fpr:.0%})")
    logger.info(f"  Human Catch: {final_catch:.1%}")

    # Also report stats at various thresholds
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        fpr = (animal_scores > t).mean()
        catch = (human_scores > t).mean()
        marker = " <<<" if abs(t - best_thresh) < 0.05 else ""
        logger.info(f"  t={t:.1f}: FPR={fpr:.4f}, Catch={catch:.3f}{marker}")

    return best_thresh, float(final_fpr), float(final_catch)


def train(X_tr, y_tr, X_va, y_va, device="cpu"):
    train_dl = DataLoader(DS(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(DS(X_va, y_va), batch_size=BATCH_SIZE, shuffle=False)

    model = SegmentEmbeddingClassifier(
        embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, hidden_dim=256
    ).to(device)

    # Class-balanced loss
    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.sum() * NUM_CLASSES
    cw = torch.from_numpy(w).float().to(device)
    logger.info(f"Class weights: non_human={w[0]:.3f}, human={w[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tc = tt = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tc += (logits.argmax(-1) == yb).sum().item()
            tt += len(yb)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb).argmax(-1)
                vc += (p == yb).sum().item()
                vt += len(yb)

        ta, va = tc / max(tt, 1), vc / max(vt, 1)

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            logger.info(f"Ep {epoch:3d}/{EPOCHS} | Train: {ta:.3f} | Val: {va:.3f}")

        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, best_acc


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Bootstrap Anti-Spoofing v3 (Binary, Clean Data, ROC Calibrated)")
    logger.info("=" * 70)

    nh_e, nh_s, h_e, h_s = load_clean_data()
    if not nh_e or not h_e:
        logger.error("Need both non-human and human data!")
        return

    X_tr, y_tr, X_va, y_va = prepare_data(nh_e, nh_s, h_e, h_s)
    logger.info(f"\nTrain: {len(X_tr)} ({(y_tr==0).sum()} nh, {(y_tr==1).sum()} h)")
    logger.info(f"Val:   {len(X_va)} ({(y_va==0).sum()} nh, {(y_va==1).sum()} h)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model, best_acc = train(X_tr, y_tr, X_va, y_va, device)
    logger.info(f"\nBest val accuracy: {best_acc:.1%}")

    # Calibrate threshold
    threshold, fpr, catch = calibrate_threshold(model, X_va, y_va, device)

    # Save
    metrics = {
        "val_accuracy": float(best_acc),
        "animal_false_positive_rate": fpr,
        "human_catch_rate": catch,
        "calibrated_threshold": threshold,
        "num_classes": NUM_CLASSES,
        "embed_dim": EMBED_DIM,
        "class_names": CLASS_NAMES,
        "clean_data_only": True,
        "binary": True,
    }

    try:
        from models.weight_manager import save_checkpoint
        path = save_checkpoint(model, "anti_spoofing_embedding", epoch=EPOCHS, metrics=metrics)
    except Exception:
        path = PROJECT_ROOT / "models" / "weights" / "anti_spoofing_embedding.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "metrics": metrics}, path)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*70}")
    logger.info(f"SAVED: {path}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Val Accuracy:    {best_acc:.1%}")
    logger.info(f"Threshold:       {threshold:.3f}")
    logger.info(f"Animal FPR:      {fpr:.4f} ({'PASS' if fpr <= TARGET_FPR else 'FAIL'} <{TARGET_FPR:.0%})")
    logger.info(f"Human Catch:     {catch:.1%}")
    logger.info(f"{'='*70}")
    logger.info("\nRestart API server to load new classifier.")


if __name__ == "__main__":
    main()
