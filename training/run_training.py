"""
AnimalVox AI — Master Training Orchestrator
=============================================
Single entry point that runs the full pipeline:
  1. Verify environment (Python, PyTorch, rclone, disk space)
  2. Setup rclone Google Drive remote (interactive, one-time)
  3. Pull dataset shards from Google Drive in priority order
  4. Extract 266-dim feature embeddings from raw audio
  5. Auto-delete raw audio to protect local disk
  6. Train all classifier heads from extracted features
  7. Save trained .pt checkpoints to models/weights/

Usage:
    # Full auto pipeline:
    py training/run_training.py --full

    # Setup rclone only:
    py training/run_training.py --setup

    # Pull + Extract only (no training):
    py training/run_training.py --extract-only --max-shards 3

    # Train only (from existing features):
    py training/run_training.py --train-only --epochs 50

    # Pull a specific dataset and train its target modules:
    py training/run_training.py --dataset dogspeak --epochs 40
"""

import os
import sys
import shutil
import logging
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")


def print_banner():
    """Print the AnimalVox AI training banner."""
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║         🐾 AnimalVox AI — Training Orchestrator 🐾          ║
    ║                                                              ║
    ║   5TB Google Drive ──► Shard ──► Extract ──► Train ──► .pt   ║
    ║          Cloud            Local (40GB safe)     Weights       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def check_environment() -> dict:
    """Verify all required components are available."""
    print("\n🔍 Checking Environment...")
    env = {}

    # Python
    env["python"] = sys.version.split()[0]
    print(f"  ✅ Python {env['python']}")

    # PyTorch
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
        device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if env["cuda"] else "CPU"
        print(f"  ✅ PyTorch {env['torch']} ({device_str})")
    except ImportError:
        env["torch"] = None
        print("  ❌ PyTorch not installed. Run: pip install torch")

    # Audio libraries
    for lib in ["librosa", "soundfile", "numpy", "scipy"]:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "?")
            env[lib] = ver
            print(f"  ✅ {lib} {ver}")
        except ImportError:
            env[lib] = None
            print(f"  ❌ {lib} not installed")

    # HuggingFace datasets
    try:
        import datasets
        env["datasets"] = datasets.__version__
        print(f"  ✅ datasets {env['datasets']}")
    except ImportError:
        env["datasets"] = None
        print("  ⚠️  datasets not installed (needed for HuggingFace). Run: pip install datasets")

    # rclone
    try:
        result = subprocess.run(["rclone", "version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ver = result.stdout.strip().split("\n")[0]
            env["rclone"] = ver
            print(f"  ✅ {ver}")
        else:
            env["rclone"] = None
            print("  ⚠️  rclone found but returned error")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        env["rclone"] = None
        print("  ⚠️  rclone not found. Install: winget install Rclone.Rclone")

    # Disk space
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    env["disk_free_gb"] = free / (1024**3)
    env["disk_total_gb"] = total / (1024**3)
    status = "✅" if env["disk_free_gb"] > 15 else "⚠️"
    print(f"  {status} Disk: {env['disk_free_gb']:.1f} GB free / {env['disk_total_gb']:.1f} GB total")

    return env


def setup_rclone():
    """Guide user through rclone Google Drive setup."""
    print("\n" + "=" * 60)
    print("📁 Setting Up rclone for Google Drive")
    print("=" * 60)
    print("""
    This will configure rclone to access your Google Drive.
    
    Steps:
    1. rclone will open a browser for Google OAuth login
    2. Grant permission to access your Google Drive
    3. The remote will be saved as 'gdrive'
    
    After setup, you can mount Google Drive as a virtual drive:
      rclone mount gdrive: X: --vfs-cache-mode full --vfs-cache-max-size 15G
    
    Or use this pipeline to copy shards on demand.
    """)

    try:
        # Check if gdrive remote already exists
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, timeout=10)
        remotes = [r.strip().rstrip(":") for r in result.stdout.strip().split("\n") if r.strip()]
        if "gdrive" in remotes:
            print("  ✅ Remote 'gdrive' already configured!")
            return True
    except Exception:
        pass

    print("  Starting rclone config... Follow the prompts in your terminal.\n")
    try:
        subprocess.run(["rclone", "config"], timeout=300)
        return True
    except Exception as e:
        logger.error(f"rclone config failed: {e}")
        return False


def setup_gdrive_folder_structure(remote_name: str = "gdrive"):
    """Create the recommended folder structure on Google Drive."""
    print("\n📂 Creating Google Drive folder structure...")

    folders = [
        "AnimalVoxDatasets/shards/week1/dogspeak",
        "AnimalVoxDatasets/shards/week1/barkopedia_emotion",
        "AnimalVoxDatasets/shards/week1/catmeows",
        "AnimalVoxDatasets/shards/week1/animal_sounds",
        "AnimalVoxDatasets/shards/week2/birdset",
        "AnimalVoxDatasets/shards/week2/openwhistle",
        "AnimalVoxDatasets/shards/week2/dclde_2022",
        "AnimalVoxDatasets/shards/week2/rfcx",
        "AnimalVoxDatasets/shards/week3/vocal_imitation",
        "AnimalVoxDatasets/shards/week3/demand",
        "AnimalVoxDatasets/shards/week3/esc50",
        "AnimalVoxDatasets/shards/week3/audioset_dogs",
        "AnimalVoxDatasets/features",
        "AnimalVoxDatasets/checkpoints",
    ]

    for folder in folders:
        try:
            subprocess.run(
                ["rclone", "mkdir", f"{remote_name}:{folder}"],
                capture_output=True, text=True, timeout=30
            )
            print(f"  ✅ {folder}")
        except Exception:
            print(f"  ⚠️  {folder} (skipped)")


def run_extraction_pipeline(
    max_shards: int = 5,
    species: str = None,
    module: str = None,
    min_free_gb: float = 15.0,
):
    """Run the shard pull → extract → purge pipeline."""
    from training.gdrive_shard_manager import GDriveShardManager

    manager = GDriveShardManager(min_free_gb=min_free_gb)

    print("\n📦 Dataset Registry:")
    manager.list_registry(species_filter=species, module_filter=module)

    print(f"\n🚀 Starting extraction pipeline (max {max_shards} shards)...")
    manager.run_priority_pipeline(
        max_shards=max_shards,
        species_filter=species,
        module_filter=module,
    )


def run_training_pipeline(
    module: str = "all",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    species: str = None,
    resume: bool = False,
):
    """Run training from pre-extracted feature shards."""
    from training.train_from_shards import (
        MODULE_CONFIGS, discover_feature_shards, train_module
    )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if module == "all":
        modules = list(MODULE_CONFIGS.keys())
    else:
        modules = [module]

    results = []
    for mod_name in modules:
        cfg = MODULE_CONFIGS[mod_name]
        species_filter = [species] if species else cfg.get("species_filter")

        npz_paths = discover_feature_shards(species_filter=species_filter)

        if not npz_paths:
            logger.warning(
                f"No feature shards for {mod_name}. "
                f"Using synthetic data for training verification."
            )

        result = train_module(
            module_name=mod_name,
            npz_paths=npz_paths,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            resume=resume,
        )
        results.append(result)

    # Final summary
    print("\n" + "=" * 70)
    print("🏆 Final Training Results")
    print("=" * 70)
    for r in results:
        print(
            f"  🐾 {r['module']:<25s} | "
            f"Best Acc: {r['best_accuracy']:.2%} | "
            f"Epochs: {r['epochs_trained']} | "
            f"Samples: {r['total_samples']}"
        )
    print("=" * 70)
    print("\n✅ All trained weights saved to: models/weights/")
    print("   AnimalVox AI will automatically use these trained models!\n")


def main():
    parser = argparse.ArgumentParser(
        description="AnimalVox AI — Master Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py training/run_training.py --setup                    # Configure rclone
  py training/run_training.py --full                     # Full pipeline
  py training/run_training.py --extract-only             # Pull + extract only
  py training/run_training.py --train-only --epochs 50   # Train from features
  py training/run_training.py --dataset dogspeak         # Specific dataset
        """
    )
    parser.add_argument("--full", action="store_true", help="Run full pipeline (setup → extract → train)")
    parser.add_argument("--setup", action="store_true", help="Setup rclone and Google Drive structure")
    parser.add_argument("--extract-only", action="store_true", help="Pull and extract only (no training)")
    parser.add_argument("--train-only", action="store_true", help="Train from existing features only")
    parser.add_argument("--dataset", type=str, help="Process a specific dataset ID")
    parser.add_argument("--module", type=str, default="all", help="Train specific module")
    parser.add_argument("--species", type=str, help="Filter by species")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-shards", type=int, default=5, help="Max shards to process")
    parser.add_argument("--min-free-gb", type=float, default=15.0, help="Disk safety watermark (GB)")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--check", action="store_true", help="Check environment only")
    args = parser.parse_args()

    print_banner()
    env = check_environment()

    if args.check:
        return

    if args.setup or args.full:
        if env.get("rclone"):
            setup_rclone()
            setup_gdrive_folder_structure()
        else:
            print("\n⚠️  rclone not installed. Install first:")
            print("    winget install Rclone.Rclone")
            print("  Then re-run: py training/run_training.py --setup")
            if not args.full:
                return

    if args.dataset:
        # Process a specific dataset
        from training.gdrive_shard_manager import GDriveShardManager, DATASET_REGISTRY
        manager = GDriveShardManager(min_free_gb=args.min_free_gb)
        result = manager.pull_and_extract(args.dataset)
        if result.success:
            # Train the target modules for this dataset
            ds_info = DATASET_REGISTRY.get(args.dataset, {})
            for mod in ds_info.get("target_modules", []):
                if mod in ("augmentation",):
                    continue
                run_training_pipeline(
                    module=mod, epochs=args.epochs,
                    batch_size=args.batch_size, lr=args.lr,
                    resume=args.resume,
                )
        return

    if args.extract_only or args.full:
        run_extraction_pipeline(
            max_shards=args.max_shards,
            species=args.species,
            module=args.module if args.module != "all" else None,
            min_free_gb=args.min_free_gb,
        )

    if args.train_only or args.full:
        run_training_pipeline(
            module=args.module,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            species=args.species,
            resume=args.resume,
        )

    if not any([args.full, args.setup, args.extract_only, args.train_only, args.dataset, args.check]):
        parser.print_help()


if __name__ == "__main__":
    main()
