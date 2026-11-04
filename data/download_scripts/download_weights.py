"""
AnimalVox AI — Pretrained Weights & Checkpoint Downloader CLI
=============================================================
Usage:
    python data/download_scripts/download_weights.py --status
    python data/download_scripts/download_weights.py --init-seeds
    python data/download_scripts/download_weights.py --download-all
    python data/download_scripts/download_weights.py --module anti_spoofing_ecapa

Features:
- Inspects which modules have trained weights vs. running in heuristic fallback.
- Initializes calibrated test seed checkpoints for all classification heads.
- Downloads external pretrained foundation backbones (e.g. SpeechBrain ECAPA-TDNN, EfficientNet).
- Validates checkpoint compatibility with current PyTorch models.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.weight_manager import (
    CHECKPOINT_REGISTRY,
    get_weights_dir,
    get_weights_status,
    download_pretrained_backbone,
    initialize_seed_weights,
    TORCH_AVAILABLE,
)


def print_banner():
    print("=" * 72)
    print(" 🐾  AnimalVox AI — Neural Weights & Heuristic Fallback Manager  🐾")
    print("=" * 72)


def display_status():
    """Print clean formatted table of all model checkpoints."""
    print_banner()
    weights_dir = get_weights_dir()
    print(f"\n📂 Weights Directory: {weights_dir}")
    print(f"🔥 PyTorch Status:    {'Available' if TORCH_AVAILABLE else 'Not Installed (Fallback Only)'}\n")

    status = get_weights_status()
    print(f"{'Module Name':<22} | {'Mode':<18} | {'Size':<10} | {'Fallback Strategy'}")
    print("-" * 72)

    for key, info in status.items():
        mode_str = "✅ TRAINED" if info["has_trained_weights"] else "⚡ FALLBACK"
        size_str = info["file_size_human"]
        fallback_str = info["fallback_mechanism"]
        if len(fallback_str) > 26:
            fallback_str = fallback_str[:23] + "..."
        print(f"{key:<22} | {mode_str:<18} | {size_str:<10} | {fallback_str}")

    print("-" * 72)
    trained_count = sum(1 for s in status.values() if s["has_trained_weights"])
    total_count = len(status)
    print(f"\nSummary: {trained_count}/{total_count} models have trained checkpoints.")
    if trained_count < total_count:
        print("💡 Modules without checkpoints run 100% on Acoustic Physics & Heuristics.")
        print("   They will automatically upgrade when you run training or download weights.")
    print("=" * 72 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AnimalVox AI Weights Downloader & Fallback Inspector"
    )
    parser.add_argument(
        "--status", action="store_true", default=False,
        help="Display the status of all model weights and fallback modes."
    )
    parser.add_argument(
        "--init-seeds", action="store_true", default=False,
        help="Initialize calibrated PyTorch seed checkpoints for all classification heads."
    )
    parser.add_argument(
        "--download-all", action="store_true", default=False,
        help="Download and initialize all available model weights and backbones."
    )
    parser.add_argument(
        "--module", type=str, default=None,
        help="Target a specific module (e.g. anti_spoofing_ecapa, breed_visual, health_monitor)."
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Force overwrite existing checkpoints."
    )

    args = parser.parse_args()

    # If no flags passed, display status by default
    if not (args.status or args.init_seeds or args.download_all or args.module):
        display_status()
        return

    if args.status:
        display_status()

    if args.init_seeds:
        print_banner()
        print("\n⚙️  Initializing calibrated test seed checkpoints...")
        results = initialize_seed_weights(force=args.force)
        for name, ok in results.items():
            status_icon = "✅" if ok else "❌"
            print(f"  {status_icon} {name}")
        print("\n🎉 Seed initialization complete! Running status check:")
        display_status()

    if args.module:
        print_banner()
        print(f"\n⚙️  Processing module: {args.module}...")
        ok = download_pretrained_backbone(args.module, force=args.force)
        print(f"  Result: {'✅ Success' if ok else '❌ Failed'}")
        display_status()

    if args.download_all:
        print_banner()
        print("\n⚙️  Downloading and initializing all checkpoints...")
        # 1. Initialize seed weights for task-specific heads
        initialize_seed_weights(force=args.force)
        # 2. Download external foundation backbones
        for key, info in CHECKPOINT_REGISTRY.items():
            if info.get("url"):
                print(f"Fetching {key}...")
                download_pretrained_backbone(key, force=args.force)
        print("\n🎉 All weights processed successfully!")
        display_status()


if __name__ == "__main__":
    main()
