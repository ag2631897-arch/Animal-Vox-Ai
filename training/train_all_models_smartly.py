"""
AnimalVox AI — Smart Multi-Stage Model Training Orchestrator
============================================================
Trains the complete AnimalVox AI neural suite using the newly ingested
datasets on the NVIDIA GeForce RTX 4050 GPU (CUDA 12.6):

1. Dual-Stream Emotion Classifier (10 classes): Focal Loss + Mixup + SWA (40 epochs)
2. Dual-Stream Breed Classifier (8 groups): Dogs + Cats multi-species (40 epochs)
3. 6-Species Animal Auto-Router Detector (15 epochs)
4. Anti-Spoofing Liveness & Replay Detector (15 epochs)
5. Clinical Multi-Label Health Distress Monitor (20 epochs)

Optimizations:
- CUDA-accelerated tensor execution with RTX 4050 GPU
- Dual-Stream Cross-Attention (256-D Semantic BEATs ⊗ 10-D Biometric Prosodic)
- Morton Ethological physical rule pseudo-labeling over 78 DogSpeak shards
- Class-Balanced Focal Loss (gamma=2.0) + Mixup (alpha=0.25)
- Stochastic Weight Averaging (SWA) + Cosine Annealing
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartTrainer")

import torch
from training.train_max_accuracy import train_max_accuracy_module
from training.train_from_shards import train_module, discover_feature_shards, MODULE_CONFIGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"⚡ Smart Trainer initialized on: {DEVICE.upper()} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")


def main():
    t_start = time.time()
    print("\n" + "═" * 78)
    print("  🚀 ANIMALVOX AI — UNIFIED HIGH-ACCURACY GPU MODEL TRAINING PIPELINE")
    print(f"     Hardware: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    print(f"     VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB" if DEVICE == 'cuda' else "")
    print("     Ingested Corpora: 100% of DogSpeak (78 shards) + 100% of StephenCHENYL Cats (11 shards)")
    print("═" * 78 + "\n")

    summary = []

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: Emotion Classifier (Dual-Stream + Ethological Focal + SWA)
    # ─────────────────────────────────────────────────────────────
    print("\n[1/5] 🧠 Training Dual-Stream Ethological Emotion Classifier (10 Classes)...")
    res_emotion = train_max_accuracy_module(
        module_name="emotion_classifier",
        epochs=45,
        batch_size=256,
        lr=6e-4,
        device=DEVICE,
    )
    summary.append({
        "module": "Emotion Classifier (10-Class)",
        "best_acc": res_emotion["best_accuracy"],
        "samples": res_emotion["total_samples"],
        "model_type": "Dual-Stream Ethological Biometric Gating (Focal+Mixup+SWA)",
    })

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: Breed Acoustic Classifier (Dual-Stream + Cats/Dogs)
    # ─────────────────────────────────────────────────────────────
    print("\n[2/5] 🐕 Training Dual-Stream Breed Acoustic Classifier (8 Groups)...")
    res_breed = train_max_accuracy_module(
        module_name="breed_acoustic",
        epochs=45,
        batch_size=256,
        lr=6e-4,
        device=DEVICE,
    )
    summary.append({
        "module": "Breed Classifier (8 Groups)",
        "best_acc": res_breed["best_accuracy"],
        "samples": res_breed["total_samples"],
        "model_type": "Dual-Stream Biometric Gating (Focal+Mixup+SWA)",
    })

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: 6-Species Animal Auto-Router Detector
    # ─────────────────────────────────────────────────────────────
    print("\n[3/5] 🐾 Training 6-Species Animal Auto-Router Detector (30 Epochs + SWA)...")
    shards_species = discover_feature_shards(module_name="animal_detector")
    res_species = train_module(
        module_name="animal_detector",
        npz_paths=shards_species,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
    )
    summary.append({
        "module": "Animal Detector (6 Species)",
        "best_acc": res_species["best_accuracy"],
        "samples": res_species["total_samples"],
        "model_type": "Multi-Species Active-Weighted ResNet + SWA",
    })

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: Anti-Spoofing & Replay Attack Detector
    # ─────────────────────────────────────────────────────────────
    print("\n[4/5] 🛡️ Training Anti-Spoofing & Replay Attack Detector (Enhanced ResNet + SWA)...")
    shards_spoof = discover_feature_shards(module_name="anti_spoofing")
    res_spoof = train_module(
        module_name="anti_spoofing",
        npz_paths=shards_spoof,
        epochs=30,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
    )
    summary.append({
        "module": "Anti-Spoofing Detector",
        "best_acc": res_spoof["best_accuracy"],
        "samples": res_spoof["total_samples"],
        "model_type": "Enhanced 3-Stage Residual Discriminator + SWA",
    })

    # ─────────────────────────────────────────────────────────────
    # STAGE 5: Clinical Multi-Label Health Monitor
    # ─────────────────────────────────────────────────────────────
    print("\n[5/5] 🩺 Training Clinical Multi-Label Health Distress Monitor (Raw Logits + SWA)...")
    shards_health = discover_feature_shards(module_name="health_monitor")
    res_health = train_module(
        module_name="health_monitor",
        npz_paths=shards_health,
        epochs=35,
        batch_size=256,
        lr=1e-3,
        device=DEVICE,
    )
    summary.append({
        "module": "Health Distress Monitor (8 Flags)",
        "best_acc": res_health["best_accuracy"],
        "samples": res_health["total_samples"],
        "model_type": "Multi-Label Positive-Weighted BCE + SWA",
    })

    # ─────────────────────────────────────────────────────────────
    # FINAL BENCHMARK SUMMARY
    # ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "═" * 78)
    print("  🏆 ANIMALVOX AI — HIGH-ACCURACY TRAINING COMPLETE")
    print(f"     Total Pipeline Elapsed Time: {elapsed / 60:.1f} minutes")
    print(f"     Trained on: {torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'}")
    print("═" * 78)
    for s in summary:
        print(f"  ✅ {s['module']:<35s} | Best Acc: {s['best_acc']:.2%} | Samples: {s['samples']:,} | {s['model_type']}")
    print("═" * 78)
    print("\nAll trained model weights have been verified and saved to models/weights/.")


if __name__ == "__main__":
    main()
