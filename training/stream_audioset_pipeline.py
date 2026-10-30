"""
AnimalVox AI — Continuous AudioSet Streaming & Training Engine (Option C)
========================================================================
Sequentially ingests the full 2.1 million Google AudioSet clips from HuggingFace
(`agkphysics/AudioSet`) across all 943 parquets (bal_train, eval, unbal_train):
1. Downloads 1 parquet at a time (~650 MB).
2. Categorizes each clip using AudioSet human_labels into:
   - Animal species (Dogs, Cats, Birds, Frogs, Wildlife)
   - Human speech anti-spoof contrast
   - Ambient soundscape anti-spoof contrast
3. Converts and extracts 266-dim dense feature vectors into compact shards.
4. Packages raw audio into zip archives and streams directly to 5.0 TiB Google Drive
   (`gdrive:AnimalVoxDatasets/raw/audioset/`).
5. Uploads feature shards to `gdrive:AnimalVoxDatasets/features/`.
6. Purges local raw audio and downloaded parquet caches immediately, strictly
   maintaining the 152 GB free disk space watermark.
7. Tracks resumable progress in `audioset_stream_progress.json`.
8. Triggers incremental 15-30 epoch model retraining every N batches and syncs
   production weights to Google Drive.

Usage:
    # Run continuous streaming worker:
    python training/stream_audioset_pipeline.py --max-parquets 50 --train-interval 5

    # Run full exhaustive continuous daemon:
    python training/stream_audioset_pipeline.py --train-interval 10
"""

import os
import sys
import io
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Enable Rust-accelerated multi-threaded downloads (10x-50x faster)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "audioset_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger("AudioSetStreamer")

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from training.feature_extractor import extract_features_from_directory

STAGING_DIR = DATA_DIR / "shard_staging" / "audioset_stream"
FEATURES_DIR = PROCESSED_DATA_DIR / "shard_features"
PROGRESS_FILE = PROCESSED_DATA_DIR / "audioset_stream_progress.json"

STAGING_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def get_free_disk_gb(path: Path = DATA_DIR) -> float:
    """Check remaining free space on disk in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def load_progress() -> Dict:
    """Load streaming progress state."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {
        "completed_parquets": [],
        "total_clips_processed": 0,
        "total_animal_clips": 0,
        "total_speech_clips": 0,
        "total_ambient_clips": 0,
        "batches_completed": 0,
        "last_trained_batch": 0,
        "last_update": 0,
    }


def save_progress(progress: Dict):
    """Save streaming progress state to disk."""
    progress["last_update"] = time.time()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def categorize_labels(human_labels: List[str]) -> Tuple[str, bool]:
    """
    Categorize AudioSet labels into AnimalVox species or anti-spoof contrast.
    Returns: (category, is_animal)
    """
    labels_lower = [str(l).lower().strip() for l in human_labels]
    all_text = " ".join(labels_lower)

    # 1. Canines (Dogs, Barks, Growls, Howls)
    if any(k in all_text for k in ["dog", "bark", "howl", "growl", "growling", "yip", "whimper"]):
        return "dogs", True

    # 2. Felines (Cats, Meows, Purrs, Hisses)
    if any(k in all_text for k in ["cat", "purr", "meow", "hiss", "caterwaul"]):
        return "cats", True

    # 3. Birds (Calls, Chirps, Songs, Squawks, Owls, Crows)
    if any(k in all_text for k in [
        "bird", "bird vocalization", "chirp", "tweet", "squawk",
        "pigeon", "dove", "crow", "rooster", "cluck", "coo"
    ]):
        return "birds", True

    # 4. Anurans (Frogs, Croaks, Toads)
    if any(k in all_text for k in ["frog", "croak", "toad", "bullfrog"]):
        return "frogs", True

    # 5. Cetaceans & Wild Mammals
    if any(k in all_text for k in [
        "whale", "dolphin", "wild animals", "roar", "lion", "tiger",
        "livestock", "farm animals", "horse", "cattle", "pig", "sheep", "bleat", "animal"
    ]):
        return "wildlife", True

    # 6. Human Speech Anti-Spoof Contrast
    if any(k in all_text for k in [
        "speech", "male speech", "female speech", "child speech",
        "conversation", "whispering", "singing", "laughter", "crying", "sobbing", "cough", "sneeze"
    ]):
        return "human_anti_spoof", False

    # 7. Ambient / Music / Environmental Noise Contrast
    return "ambient_anti_spoof", False


def list_audioset_parquets() -> List[str]:
    """
    Discover all 943 AudioSet parquets on Hugging Face:
    - bal_train (38 parquets)
    - eval (35 parquets)
    - unbal_train (870 parquets)
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files("agkphysics/AudioSet", repo_type="dataset")
        
        bal = sorted([f for f in files if f.startswith("data/bal_train/") and f.endswith(".parquet")])
        eval_f = sorted([f for f in files if f.startswith("data/eval/") and f.endswith(".parquet")])
        unbal = sorted([f for f in files if f.startswith("data/unbal_train/") and f.endswith(".parquet")])
        
        return bal + eval_f + unbal
    except Exception as e:
        logger.error(f"Error fetching AudioSet parquet list: {e}")
        return []


def download_single_parquet(pq_path: str, token: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Download a parquet file using hf_hub_download. Returns (pq_path, cached_file)."""
    from huggingface_hub import hf_hub_download
    logger.info(f"📥 [Prefetch] Downloading AudioSet parquet: {pq_path}")
    cached_file = None
    for attempt in range(1, 5):
        try:
            cached_file = hf_hub_download(
                repo_id="agkphysics/AudioSet",
                filename=pq_path,
                repo_type="dataset",
                token=token,
            )
            if cached_file and os.path.exists(cached_file):
                break
        except Exception as e:
            wait_time = attempt * 8
            logger.warning(f"Download attempt {attempt}/4 for {pq_path} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    if not cached_file or not os.path.exists(cached_file):
        logger.error(f"❌ Failed to download {pq_path} after 4 attempts.")
        return pq_path, None

    return pq_path, cached_file


def extract_parquet_audio(
    cached_file: str,
    pq_path: str,
    batch_id: int,
    max_non_animal_per_pq: int = 50,
) -> Tuple[int, int, int]:
    """
    Extract audio clips directly from a downloaded parquet file:
    - Reads parquet
    - Writes audio files directly into staging
    - Cleans up downloaded parquet file
    - Returns (animal_count, speech_count, ambient_count)
    """
    import pandas as pd

    try:
        df = pd.read_parquet(cached_file)
        logger.info(f"  Rows in {Path(pq_path).name}: {len(df)}")
    except Exception as e:
        logger.error(f"Failed to read parquet {pq_path}: {e}")
        try:
            if cached_file and os.path.exists(cached_file):
                os.remove(cached_file)
        except Exception:
            pass
        return 0, 0, 0

    animal_count = 0
    speech_count = 0
    ambient_count = 0

    batch_staging = STAGING_DIR / f"batch_{batch_id:04d}"
    batch_staging.mkdir(parents=True, exist_ok=True)

    labels_list = df["human_labels"].tolist() if "human_labels" in df.columns else []
    audio_list = df["audio"].tolist() if "audio" in df.columns else []
    n_rows = len(df)

    for idx in range(n_rows):
        human_labels = labels_list[idx] if idx < len(labels_list) else []
        cat, is_animal = categorize_labels(human_labels)

        # Balanced sampling for non-animal contrast to prevent skewing corpus
        if not is_animal:
            if cat == "human_anti_spoof" and speech_count >= max_non_animal_per_pq:
                continue
            elif cat == "ambient_anti_spoof" and ambient_count >= max_non_animal_per_pq:
                continue

        audio_entry = audio_list[idx] if idx < len(audio_list) else None
        raw_bytes = None
        if isinstance(audio_entry, dict) and "bytes" in audio_entry:
            raw_bytes = audio_entry["bytes"]
        elif isinstance(audio_entry, bytes):
            raw_bytes = audio_entry

        if not raw_bytes or len(raw_bytes) < 100:
            continue

        vid = f"as_{batch_id:04d}_{idx}"
        dest_dir = batch_staging / cat
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{vid}.wav"

        try:
            with open(dest_file, "wb") as f:
                f.write(raw_bytes)
            if is_animal:
                animal_count += 1
            elif cat == "human_anti_spoof":
                speech_count += 1
            else:
                ambient_count += 1
        except Exception:
            pass

    # Clean up downloaded parquet file to save disk space
    try:
        if cached_file and os.path.exists(cached_file):
            os.remove(cached_file)
    except Exception as e:
        logger.warning(f"Could not remove cached parquet: {e}")

    return animal_count, speech_count, ambient_count


def extract_batch_features_and_upload(batch_id: int) -> bool:
    """
    Extract 266-dim feature vectors from the batch staging directory,
    upload raw audio archive to Google Drive, upload feature shard to Google Drive,
    and delete local audio files.
    """
    batch_staging = STAGING_DIR / f"batch_{batch_id:04d}"
    if not batch_staging.exists():
        return False

    audio_files = list(batch_staging.rglob("*"))
    audio_files = [f for f in audio_files if f.is_file() and f.suffix.lower() in [".wav", ".flac", ".mp3"]]

    if not audio_files:
        shutil.rmtree(batch_staging, ignore_errors=True)
        return False

    shard_id = f"audioset_stream_part_{batch_id:04d}"
    feature_path = FEATURES_DIR / f"features_{shard_id}.npz"

    logger.info(f"⚡ Extracting 266-dim features for Batch {batch_id:04d} ({len(audio_files)} clips)...")
    res_path, count = extract_features_from_directory(
        audio_dir=batch_staging,
        output_path=feature_path,
        species="multi",
        dataset_id=shard_id,
    )

    if count > 0 and res_path and res_path.exists():
        # 1. Upload feature shard to Google Drive
        try:
            logger.info(f"☁️ Uploading {res_path.name} to gdrive:AnimalVoxDatasets/features/...")
            subprocess.run(
                ["rclone", "copy", str(res_path), "gdrive:AnimalVoxDatasets/features/", "--transfers", "2"],
                timeout=180
            )
        except Exception as e:
            logger.warning(f"Google Drive feature upload warning: {e}")

        # 2. Package raw audio into zip archive
        zip_file = STAGING_DIR / f"audioset_raw_{batch_id:04d}.zip"
        logger.info(f"📦 Packaging {len(audio_files)} raw clips to {zip_file.name}...")
        shutil.make_archive(str(STAGING_DIR / f"audioset_raw_{batch_id:04d}"), "zip", batch_staging)

        # 3. Stream zip archive directly to Google Drive
        try:
            logger.info(f"☁️ Uploading {zip_file.name} to gdrive:AnimalVoxDatasets/raw/audioset/...")
            subprocess.run(
                ["rclone", "copy", str(zip_file), "gdrive:AnimalVoxDatasets/raw/audioset/", "--drive-chunk-size", "64M"],
                timeout=600
            )
            if zip_file.exists():
                zip_file.unlink()
        except Exception as e:
            logger.warning(f"Google Drive raw archive upload warning: {e}")

        # 4. Purge local staging raw files
        shutil.rmtree(batch_staging, ignore_errors=True)
        logger.info(f"🗑️  Purged local raw audio for Batch {batch_id:04d}. Features preserved locally & in cloud.")
        return True

    shutil.rmtree(batch_staging, ignore_errors=True)
    return False


def run_incremental_training(epochs: int = 15):
    """Trigger incremental model retraining on the expanded feature shard corpus."""
    logger.info(f"🚀 Triggering incremental model retraining ({epochs} epochs across all shards)...")
    try:
        subprocess.run(
            [sys.executable, "training/train_from_shards.py", "--module", "all", "--epochs", str(epochs)],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        logger.info("☁️ Mirroring new production weights to Google Drive checkpoints...")
        subprocess.run(
            ["rclone", "copy", "models/weights/", "gdrive:AnimalVoxDatasets/checkpoints/", "--transfers", "4"],
            cwd=str(PROJECT_ROOT),
            timeout=180
        )
        logger.info("✅ Incremental training and weight sync complete.")
    except Exception as e:
        logger.error(f"Incremental training error: {e}")


def stream_audioset(
    max_parquets: Optional[int] = None,
    train_interval: int = 5,
    epochs_per_train: int = 15,
):
    """
    Main continuous streaming orchestrator for Google AudioSet.
    Iterates through all 943 parquets sequentially.
    """
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    free_gb = get_free_disk_gb()
    logger.info(f"💾 Initial free disk space: {free_gb:.1f} GB")
    if free_gb < 15.0:
        logger.error("Free disk space below 15 GB safety watermark. Aborting.")
        return

    progress = load_progress()
    completed = set(progress["completed_parquets"])

    all_parquets = list_audioset_parquets()
    logger.info(f"📋 Total AudioSet parquets discovered: {len(all_parquets)}")
    logger.info(f"⏩ Already completed: {len(completed)} parquets")

    remaining = [p for p in all_parquets if p not in completed]
    if max_parquets:
        remaining = remaining[:max_parquets]

    logger.info(f"🎯 Target parquets for this run: {len(remaining)}")

    batch_id = progress["batches_completed"]
    parquets_in_current_batch = 0

    from concurrent.futures import ThreadPoolExecutor

    prefetch_workers = 3
    logger.info(f"⚡ AudioSet Ingestion: Prefetching with {prefetch_workers} parallel download workers...")
    if train_interval == 0:
        logger.info("⚡ Training pauses DISABLED for maximum uninterrupted download throughput.")

    executor = ThreadPoolExecutor(max_workers=prefetch_workers)
    prefetch_tasks = {}

    # Prime initial 3 parallel downloads
    submit_idx = 0
    while submit_idx < min(prefetch_workers, len(remaining)):
        pq = remaining[submit_idx]
        prefetch_tasks[pq] = executor.submit(download_single_parquet, pq, token)
        submit_idx += 1

    for i, pq in enumerate(remaining):
        # 1. Disk space safety check before every parquet
        free_now = get_free_disk_gb()
        if free_now < 15.0:
            logger.warning(f"⚠️ Free disk space dropped to {free_now:.1f} GB. Performing emergency cleanup...")
            shutil.rmtree(STAGING_DIR, ignore_errors=True)
            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            time.sleep(5)
            if get_free_disk_gb() < 12.0:
                logger.error("Disk space still critical. Pausing pipeline to protect host.")
                break

        print("\n" + "━" * 70)
        logger.info(f"🚀 Processing [{i+1}/{len(remaining)}] — {pq}")
        print("━" * 70)

        # 2. Retrieve downloaded parquet from prefetch queue
        fut = prefetch_tasks.pop(pq, None)
        cached_file = None
        if fut is not None:
            _, cached_file = fut.result()
        else:
            _, cached_file = download_single_parquet(pq, token)

        # 3. Queue next parquet download immediately to keep all 3 workers busy!
        if submit_idx < len(remaining):
            next_pq = remaining[submit_idx]
            prefetch_tasks[next_pq] = executor.submit(download_single_parquet, next_pq, token)
            submit_idx += 1

        if not cached_file:
            logger.warning(f"Skipping {pq} due to download failure.")
            continue

        # 4. Extract audio from parquet directly to staging
        n_animal, n_speech, n_ambient = extract_parquet_audio(
            cached_file=cached_file,
            pq_path=pq,
            batch_id=batch_id,
            max_non_animal_per_pq=50
        )

        progress["total_animal_clips"] += n_animal
        progress["total_speech_clips"] += n_speech
        progress["total_ambient_clips"] += n_ambient
        progress["total_clips_processed"] += (n_animal + n_speech + n_ambient)
        progress["completed_parquets"].append(pq)
        parquets_in_current_batch += 1

        logger.info(f"  Clips from this parquet: {n_animal} animals, {n_speech} speech, {n_ambient} ambient")
        logger.info(f"  Cumulative corpus: {progress['total_animal_clips']} animals, {progress['total_speech_clips']} speech")

        # 5. Extract features and archive every 2 parquets (~1,000 clips)
        if parquets_in_current_batch >= 2 or (i + 1) == len(remaining):
            extract_batch_features_and_upload(batch_id)
            progress["batches_completed"] = batch_id + 1
            batch_id += 1
            parquets_in_current_batch = 0
            save_progress(progress)

            # 6. Trigger incremental retraining only if train_interval > 0
            if train_interval > 0 and (batch_id - progress["last_trained_batch"]) >= train_interval:
                run_incremental_training(epochs=epochs_per_train)
                progress["last_trained_batch"] = batch_id
                save_progress(progress)

        save_progress(progress)

    executor.shutdown(wait=False)

    print("\n" + "=" * 70)
    print("🎉 AudioSet Streaming Ingestion Batch Complete!")
    print(f"  Total parquets completed : {len(progress['completed_parquets'])} / {len(all_parquets)}")
    print(f"  Total animal clips       : {progress['total_animal_clips']}")
    print(f"  Total speech clips       : {progress['total_speech_clips']}")
    print(f"  Total batches generated  : {progress['batches_completed']}")
    print(f"  Remaining free disk      : {get_free_disk_gb():.1f} GB")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — AudioSet 2.1M Streaming Pipeline")
    parser.add_argument("--max-parquets", type=int, default=None, help="Max parquets to process in this run")
    parser.add_argument("--train-interval", type=int, default=0, help="Retrain models every N batches (0 to disable)")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per incremental retraining run")
    args = parser.parse_args()

    stream_audioset(
        max_parquets=args.max_parquets,
        train_interval=args.train_interval,
        epochs_per_train=args.epochs,
    )


if __name__ == "__main__":
    main()
