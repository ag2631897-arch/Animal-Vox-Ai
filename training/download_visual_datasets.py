"""
AnimalVox AI — Visual Breed Datasets Downloader & Ingestion Engine
==================================================================
Downloads and prepares visual datasets for multi-modal late fusion (Module 02):
1. Oxford-IIIT Pet Dataset (37 breeds: 25 dog breeds, 12 cat breeds, 7,349 images)
2. Stanford Dogs Dataset (120 dog breeds, 20,580 images)
3. EfficientNet-B4 Pretrained Visual Weights (breed_visual.pt, 74.5 MB)

Usage:
    python training/download_visual_datasets.py --dataset all
    python training/download_visual_datasets.py --dataset oxford_pets
    python training/download_visual_datasets.py --dataset stanford_dogs
    python training/download_visual_datasets.py --dataset backbone
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import requests
import io
import logging
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VisualDatasetDownloader")

VISUAL_DIR = DATA_DIR / "raw" / "visual"
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"


def check_disk_space(required_gb: float = 3.0) -> bool:
    """Ensure host drive has at least 15 GB free space."""
    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = free / (1024 ** 3)
    logger.info(f"💾 Host Free Disk Space: {free_gb:.2f} GB (Required: {required_gb:.1f} GB, Safety Watermark: >15 GB)")
    if free_gb - required_gb < 15.0:
        logger.error(f"❌ Disk space safety guard triggered! Free: {free_gb:.2f} GB. Cannot download safely.")
        return False
    return True


def download_visual_backbone(force: bool = False) -> bool:
    """Download pretrained EfficientNet-B4 visual weights to models/weights/breed_visual.pt."""
    target_path = WEIGHTS_DIR / "breed_visual.pt"
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024 and not force:
        logger.info(f"✅ Pretrained visual backbone already exists at {target_path} ({target_path.stat().st_size / 1024**2:.1f} MB)")
        return True

    url = "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"
    logger.info(f"📥 Downloading EfficientNet-B4 visual backbone from PyTorch ({url})...")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total_len = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_len > 0:
                        pct = (downloaded / total_len) * 100.0
                        sys.stdout.write(f"\r  Progress: {pct:.1f}% ({downloaded / 1024**2:.1f} / {total_len / 1024**2:.1f} MB)")
                        sys.stdout.flush()
        sys.stdout.write("\n")
        logger.info(f"✅ Saved pretrained visual weights to {target_path} ({target_path.stat().st_size / 1024**2:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download visual backbone: {e}")
        return False


def download_oxford_pets(max_per_breed: Optional[int] = None) -> Dict:
    """
    Download and extract Oxford-IIIT Pet Dataset (37 dog & cat breeds).
    Saves organized images into data/raw/visual/oxford_pets/<breed>/<image_id>.jpg.
    """
    out_dir = VISUAL_DIR / "oxford_pets"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📦 Ingesting Oxford-IIIT Pet Dataset (37 Breeds: 25 Dog + 12 Cat)...")

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    repo_id = "timm/oxford-iiit-pet"
    splits = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"]

    breed_counts: Dict[str, int] = {}
    species_map: Dict[str, str] = {}
    total_extracted = 0

    for split_rel in splits:
        logger.info(f"📥 Fetching {split_rel} from {repo_id}...")
        parquet_path = hf_hub_download(repo_id, split_rel, repo_type="dataset")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        logger.info(f"  Loaded {len(df)} images from {Path(split_rel).name}")

        for idx, row in df.iterrows():
            img_id = str(row["image_id"])
            is_dog = bool(row["label_cat_dog"] == 1)
            species = "dog" if is_dog else "cat"

            # Derive clean breed name from image_id (e.g. "american_bulldog_80" -> "american_bulldog")
            parts = img_id.rsplit("_", 1)
            breed_raw = parts[0].lower().replace(" ", "_")

            if max_per_breed and breed_counts.get(breed_raw, 0) >= max_per_breed:
                continue

            breed_dir = images_dir / breed_raw
            breed_dir.mkdir(parents=True, exist_ok=True)
            img_file = breed_dir / f"{img_id}.jpg"

            if not img_file.exists():
                try:
                    img_data = row["image"]["bytes"]
                    with open(img_file, "wb") as f:
                        f.write(img_data)
                except Exception as e:
                    logger.debug(f"Failed to write image {img_id}: {e}")
                    continue

            breed_counts[breed_raw] = breed_counts.get(breed_raw, 0) + 1
            species_map[breed_raw] = species
            total_extracted += 1

            if total_extracted % 500 == 0:
                logger.info(f"  Extracted {total_extracted} images across {len(breed_counts)} breeds...")

    # Save summary metadata
    meta = {
        "dataset_name": "Oxford-IIIT Pet Dataset",
        "num_breeds": len(breed_counts),
        "total_images": total_extracted,
        "breeds": {
            breed: {"count": count, "species": species_map.get(breed, "unknown")}
            for breed, count in sorted(breed_counts.items())
        },
    }
    with open(out_dir / "breeds.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"🎉 Oxford-IIIT Pet Ingestion Complete: {total_extracted} images across {len(breed_counts)} breeds!")
    return meta


def download_stanford_dogs(max_per_breed: Optional[int] = None) -> Dict:
    """
    Download and extract Stanford Dogs Dataset (120 dog breeds, 20,580 images).
    Saves organized images into data/raw/visual/stanford_dogs/images/<breed>/<image_id>.jpg.
    """
    out_dir = VISUAL_DIR / "stanford_dogs"
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        logger.info("📥 Fetching Stanford Dogs archive (705 MB) via resilient Hugging Face Hub...")
        archive_path = Path(hf_hub_download(repo_id="Alanox/stanford-dogs", filename="images.tar.gz", repo_type="dataset"))
        logger.info(f"✅ Stanford Dogs archive verified at {archive_path} ({archive_path.stat().st_size / 1024**2:.1f} MB).")
    except Exception as e:
        logger.error(f"❌ Stanford Dogs download failed: {e}")
        return {}

    # Extract images by breed using metadata.csv mapping
    import pandas as pd
    logger.info("📋 Loading Stanford Dogs breed mapping from metadata.csv...")
    meta_csv_path = hf_hub_download(repo_id="Alanox/stanford-dogs", filename="metadata.csv", repo_type="dataset")
    meta_df = pd.read_csv(meta_csv_path)
    name_to_breed = {str(r["name"]): str(r["target"]).lower().replace(" ", "_") for _, r in meta_df.iterrows()}
    logger.info(f"  Loaded {len(name_to_breed)} image-to-breed mappings.")

    logger.info("📦 Unpacking Stanford Dogs archive into breed directories...")
    breed_counts: Dict[str, int] = {}
    total_extracted = 0

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            logger.info(f"  Archive contains {len(members)} entries. Extracting...")
            for m in members:
                if not m.isfile():
                    continue
                file_name = Path(m.name).name
                breed_name = name_to_breed.get(file_name, "unknown_dog")

                if max_per_breed and breed_counts.get(breed_name, 0) >= max_per_breed:
                    continue

                breed_dir = images_dir / breed_name
                breed_dir.mkdir(parents=True, exist_ok=True)
                dest_file = breed_dir / file_name

                if not dest_file.exists():
                    f_obj = tar.extractfile(m)
                    if f_obj:
                        with open(dest_file, "wb") as out_f:
                            out_f.write(f_obj.read())

                breed_counts[breed_name] = breed_counts.get(breed_name, 0) + 1
                total_extracted += 1

                if total_extracted % 2000 == 0:
                    logger.info(f"  Extracted {total_extracted} images across {len(breed_counts)} dog breeds...")

        # Remove archive only if located in output directory
        if archive_path.parent == out_dir and archive_path.exists():
            archive_path.unlink()
            logger.info("🗑️  Cleaned up raw images.tar.gz archive to preserve disk space.")

        # Save metadata
        meta = {
            "dataset_name": "Stanford Dogs Dataset",
            "num_breeds": len(breed_counts),
            "total_images": total_extracted,
            "breeds": {breed: count for breed, count in sorted(breed_counts.items())},
        }
        with open(out_dir / "breeds.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"🎉 Stanford Dogs Ingestion Complete: {total_extracted} images across {len(breed_counts)} dog breeds!")
        return meta
    except Exception as e:
        logger.error(f"❌ Error unpacking Stanford Dogs archive: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Visual Breed Dataset Downloader")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "oxford_pets", "stanford_dogs", "backbone"],
        help="Which visual dataset/model to download",
    )
    parser.add_argument("--max-per-breed", type=int, default=None, help="Optional cap of images per breed")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    print("=" * 70)
    print(" 🐾  AnimalVox AI — Visual Breed Dataset Ingestion Engine  🐾")
    print("=" * 70)

    if not check_disk_space(required_gb=3.0):
        sys.exit(1)

    # 1. Visual Backbone
    if args.dataset in ("all", "backbone"):
        logger.info("\n--- [STEP 1] Pretrained EfficientNet-B4 Visual Backbone ---")
        download_visual_backbone(force=args.force)

    # 2. Oxford-IIIT Pet Dataset
    if args.dataset in ("all", "oxford_pets"):
        logger.info("\n--- [STEP 2] Oxford-IIIT Pet Dataset (37 Breeds) ---")
        download_oxford_pets(max_per_breed=args.max_per_breed)

    # 3. Stanford Dogs Dataset
    if args.dataset in ("all", "stanford_dogs"):
        logger.info("\n--- [STEP 3] Stanford Dogs Dataset (120 Dog Breeds) ---")
        download_stanford_dogs(max_per_breed=args.max_per_breed)

    print("\n" + "=" * 70)
    print(" 🎉 All Requested Visual Datasets & Models Successfully Downloaded! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
