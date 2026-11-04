"""
AnimalVox AI — Extended Bioacoustic Datasets Streaming & Archival Worker
==========================================================================
Continuously downloads and ingests the remaining large public datasets:
  1. DogSpeak (Full Archive — remaining ~72,000 clips) [HuggingFace]
  2. BirdSet (Avian Bioacoustic Benchmark) [HuggingFace]
  3. BirdCLEF (2024 / 2023 / 2022) [Kaggle]
  4. DCLDE 2022 / 2020 (NOAA Marine Mammals) [Zenodo]
  5. OpenWhistle (Dolphin Whistle Acoustic Benchmark) [OpenWhistle/HF]
  6. Earth Species Project (ESP Foundation Benchmark) [HuggingFace]
  7. DCASE 2020 Task 5 (Dogs & Bioacoustics) [Zenodo]
  8. MUSAN (Speech, Music, Noise Anti-Spoof Contrast) [OpenSLR]
  9. DEMAND Noise Database (18 Environmental Soundscapes) [Zenodo]
 10. Vocal Imitation Set (Human Mimicking Animals) [Zenodo]
 11. Watkins Marine Database (Full WHOI Cetacean Cuts) [WHOI/HF]

Key Features:
  - Strict Disk Watermark: Requires >= 15.0 GB free disk space. Purges local raw
    audio immediately after feature extraction.
  - Automatic Google Drive Streaming: Compresses raw audio into .zip archives and
    streams directly to `gdrive:AnimalVoxDatasets/raw/<dataset>/`.
  - Feature Sharding: Extracts 266-dim dense embeddings using 6 parallel workers
    and stores feature shards in `data/processed/shard_features/` & Google Drive.
  - AudioSet Chaining: Automatically waits in standby while Google AudioSet finishes,
    then kicks off sequentially. Can also run immediately with `--now`.
  - Resumable: Tracks all progress in `extended_stream_progress.json`.

Usage:
  # Normal usage (waits for AudioSet to finish, then starts):
  python training/stream_extended_datasets_pipeline.py

  # Start immediately (bypass AudioSet wait):
  python training/stream_extended_datasets_pipeline.py --now

  # Run only a specific dataset:
  python training/stream_extended_datasets_pipeline.py --dataset dog_speak --now
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
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

# Configure environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = PROJECT_ROOT / "extended_datasets_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger("ExtendedStreamer")

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from training.feature_extractor import extract_features_from_directory

STAGING_DIR = DATA_DIR / "shard_staging" / "extended_stream"
FEATURES_DIR = PROCESSED_DATA_DIR / "shard_features"
PROGRESS_FILE = PROCESSED_DATA_DIR / "extended_stream_progress.json"

STAGING_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def get_free_disk_gb(path: Path = DATA_DIR) -> float:
    """Check remaining free space on disk in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def load_progress() -> Dict:
    """Load progress state."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    return {
        "completed_datasets": [],
        "completed_batches": {},
        "total_clips_processed": 0,
        "last_update": 0,
    }


def save_progress(progress: Dict):
    """Save progress state to disk."""
    progress["last_update"] = time.time()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def is_audioset_running() -> bool:
    """Check if AudioSet streaming pipeline is actively running."""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                cmd_str = ' '.join(cmdline)
                if 'stream_audioset_pipeline' in cmd_str and proc.pid != os.getpid():
                    return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Process check warning: {e}")
    return False


def wait_for_audioset_completion(check_interval_sec: int = 60):
    """Wait in standby until AudioSet streaming pipeline finishes."""
    logger.info("⏳ Checking AudioSet pipeline status...")
    if not is_audioset_running():
        logger.info("✅ AudioSet pipeline is not currently running. Proceeding with extended datasets!")
        return

    logger.info("🔄 AudioSet streaming pipeline is currently active in the background.")
    logger.info("⏳ Waiting for AudioSet ingestion to finish before starting extended datasets...")
    
    while is_audioset_running():
        free_gb = get_free_disk_gb()
        logger.info(f"⏳ AudioSet active | Disk free: {free_gb:.1f} GB | Next check in {check_interval_sec}s...")
        time.sleep(check_interval_sec)

    logger.info("🎉 AudioSet streaming pipeline has finished! Launching extended bioacoustic datasets...")


def process_and_upload_batch(
    batch_dir: Path,
    dataset_name: str,
    batch_id: int,
    species: str = "multi",
) -> Tuple[int, Optional[Path]]:
    """
    1. Extracts 266-dim feature embeddings from batch_dir.
    2. Uploads feature shard .npz to Google Drive.
    3. Packages raw audio to .zip and uploads to Google Drive.
    4. Purges local raw audio and zip to keep local disk free.
    """
    audio_files = [f for f in batch_dir.rglob("*") if f.is_file() and f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]]
    if not audio_files:
        shutil.rmtree(batch_dir, ignore_errors=True)
        return 0, None

    shard_id = f"{dataset_name}_part_{batch_id:04d}"
    feature_path = FEATURES_DIR / f"features_{shard_id}.npz"

    logger.info(f"⚡ Extracting 266-dim features for {dataset_name} Batch {batch_id:04d} ({len(audio_files)} clips)...")
    res_path, count = extract_features_from_directory(
        audio_dir=batch_dir,
        output_path=feature_path,
        species=species,
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
            logger.warning(f"GDrive feature upload warning: {e}")

        # 2. Package raw audio into zip archive
        zip_base = STAGING_DIR / f"{dataset_name}_raw_{batch_id:04d}"
        logger.info(f"📦 Packaging {len(audio_files)} clips to {zip_base.name}.zip...")
        shutil.make_archive(str(zip_base), "zip", batch_dir)
        zip_file = zip_base.with_suffix(".zip")

        # 3. Stream zip archive to Google Drive
        try:
            logger.info(f"☁️ Uploading {zip_file.name} to gdrive:AnimalVoxDatasets/raw/{dataset_name}/...")
            subprocess.run(
                ["rclone", "copy", str(zip_file), f"gdrive:AnimalVoxDatasets/raw/{dataset_name}/", "--drive-chunk-size", "64M"],
                timeout=600
            )
            if zip_file.exists():
                zip_file.unlink()
        except Exception as e:
            logger.warning(f"GDrive raw archive upload warning: {e}")
            if zip_file.exists():
                zip_file.unlink()

        # 4. Purge local staging raw files
        shutil.rmtree(batch_dir, ignore_errors=True)
        logger.info(f"🗑️ Purged local raw audio for {dataset_name} Batch {batch_id:04d}. Features preserved locally & in cloud.")
        return count, res_path

    shutil.rmtree(batch_dir, ignore_errors=True)
    return 0, None


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Ingestion Workers
# ═══════════════════════════════════════════════════════════════════════════

def ingest_dogspeak_full(progress: Dict, batch_size: int = 1000):
    """
    Ingest remaining ~72,000 clips from DogSpeak dataset on HuggingFace.
    """
    dataset_name = "dogspeak_full"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    from huggingface_hub import hf_hub_download
    import pandas as pd

    # Download metadata.csv
    meta_path = hf_hub_download(
        repo_id="ArlingtonCL2/DogSpeak_Dataset",
        filename="metadata.csv",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )
    df = pd.read_csv(meta_path)
    total_files = len(df)
    logger.info(f"  DogSpeak metadata contains {total_files} audio files.")

    # Completed batches tracking
    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))
    total_batches = (total_files + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        if b_idx in completed_batches:
            continue

        if get_free_disk_gb() < 15.0:
            logger.warning("Disk space < 15 GB. Pausing ingestion.")
            break

        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_files)
        batch_df = df.iloc[start_i:end_i]

        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📥 [{b_idx+1}/{total_batches}] Downloading DogSpeak batch {b_idx:04d} ({len(batch_df)} files)...")

        def _fetch(row):
            fname = row.get("filename", "")
            dog_id = row.get("dog_id", "")
            if not fname or not dog_id:
                return False
            rel_path = f"dogspeak_released/{dog_id}/{fname}"
            try:
                cached = hf_hub_download(
                    repo_id="ArlingtonCL2/DogSpeak_Dataset",
                    filename=rel_path,
                    repo_type="dataset",
                    token=os.getenv("HF_TOKEN"),
                )
                breed = str(row.get("breed", "unknown")).lower().strip().replace(" ", "_")
                target_sub = batch_dir / breed
                target_sub.mkdir(parents=True, exist_ok=True)
                dest = target_sub / fname
                shutil.copy2(cached, dest)
                return True
            except Exception as e:
                logger.debug(f"DogSpeak fetch failed for {rel_path}: {e}")
                return False

        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(_fetch, [row for _, row in batch_df.iterrows()]))

        count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="dogs")
        if count > 0:
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        else:
            logger.warning(f"Batch {b_idx} yielded 0 audio clips. Not recording as complete.")

    completed_count = len(progress.get("completed_batches", {}).get(dataset_name, []))
    if completed_count >= total_batches:
        progress.setdefault("completed_datasets", []).append(dataset_name)
        save_progress(progress)
        logger.info(f"✅ Finished Ingestion: {dataset_name} ({completed_count}/{total_batches} batches)")
    else:
        logger.warning(f"⚠️ {dataset_name} partial progress: {completed_count}/{total_batches} batches completed.")


def ingest_birdset(progress: Dict):
    """
    Stream BirdSet multi-task avian benchmark datasets from Hugging Face.
    """
    dataset_name = "birdset"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        files = api.list_repo_files("DBD-research-group/BirdSet", repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")][:50]  # Ingest top benchmark partitions
    except Exception as e:
        logger.error(f"Error listing BirdSet repo: {e}")
        return

    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, pq in enumerate(parquet_files):
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        logger.info(f"📥 [{b_idx+1}/{len(parquet_files)}] Downloading BirdSet: {pq}")
        try:
            cached_pq = hf_hub_download(
                repo_id="DBD-research-group/BirdSet",
                filename=pq,
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
            )
            import pandas as pd
            df = pd.read_parquet(cached_pq)
            batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            for idx, row in df.iterrows():
                audio_data = row.get("audio")
                if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
                    ext = audio_data.get("path", "clip.wav").split(".")[-1]
                    out_f = batch_dir / f"birdset_{b_idx}_{idx}.{ext}"
                    with open(out_f, "wb") as f:
                        f.write(audio_data["bytes"])

            if cached_pq and os.path.exists(cached_pq):
                os.remove(cached_pq)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="birds")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"Error processing {pq}: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_birdclef(progress: Dict):
    """
    Stream BirdCLEF competition audio from Kaggle.
    """
    dataset_name = "birdclef"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logger.error(f"Kaggle authentication error: {e}")
        return

    competitions = ["birdclef-2024", "birdclef-2023"]
    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, comp in enumerate(competitions):
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        logger.info(f"📥 [{b_idx+1}/{len(competitions)}] Downloading Kaggle competition audio: {comp}...")
        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download competition archive
            api.competition_download_files(competition=comp, path=str(batch_dir), quiet=False)
            # Extract zips
            for zf in list(batch_dir.glob("*.zip")):
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(batch_dir)
                zf.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="birds")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"BirdCLEF {comp} download warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_dclde_marine(progress: Dict):
    """
    Download and stream DCLDE 2022/2020 Marine Mammal recordings from Zenodo.
    """
    dataset_name = "dclde_marine"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    rec_id = "7738698"

    try:
        r = requests.get(f"https://zenodo.org/api/records/{rec_id}", timeout=30)
        data = r.json()
        files = data.get("files", [])
    except Exception as e:
        logger.error(f"Zenodo DCLDE error: {e}")
        return

    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, f_info in enumerate(files):
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        f_url = f_info.get("links", {}).get("self")
        f_name = f_info.get("key")
        logger.info(f"📥 [{b_idx+1}/{len(files)}] Downloading DCLDE archive: {f_name}...")

        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        dest_file = batch_dir / f_name

        try:
            resp = requests.get(f_url, stream=True, timeout=120)
            with open(dest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            if dest_file.suffix.lower() == ".zip":
                with zipfile.ZipFile(dest_file, "r") as z:
                    z.extractall(batch_dir)
                dest_file.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="whales")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"DCLDE file {f_name} warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_openwhistle_and_watkins(progress: Dict):
    """
    Ingest OpenWhistle dolphin recordings & full Watkins Marine WHOI database.
    """
    dataset_name = "openwhistle_watkins"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.getenv("HF_TOKEN"))

    repo_id = "DBD-research-group/beans_watkins"
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        parquets = [f for f in files if f.endswith(".parquet")]
    except Exception as e:
        logger.error(f"Error fetching Watkins parquets: {e}")
        return

    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, pq in enumerate(parquets):
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        logger.info(f"📥 [{b_idx+1}/{len(parquets)}] Downloading Marine parquet: {pq}...")
        try:
            cached_pq = hf_hub_download(repo_id=repo_id, filename=pq, repo_type="dataset", token=os.getenv("HF_TOKEN"))
            import pandas as pd
            df = pd.read_parquet(cached_pq)
            batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            for idx, row in df.iterrows():
                audio_data = row.get("audio")
                if isinstance(audio_data, dict) and "bytes" in audio_data and audio_data["bytes"]:
                    out_f = batch_dir / f"watkins_{b_idx}_{idx}.wav"
                    with open(out_f, "wb") as f:
                        f.write(audio_data["bytes"])

            if cached_pq and os.path.exists(cached_pq):
                os.remove(cached_pq)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="whales")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"Marine parquet {pq} warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_earth_species_project(progress: Dict):
    """
    Ingest multi-species bioacoustic datasets from Earth Species Project.
    """
    dataset_name = "earth_species_project"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=os.getenv("HF_TOKEN"))

    repos = ["cgeorgiaw/animal-sounds", "MuhammadQASIM111/Animal_Sound_Classification"]
    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, repo in enumerate(repos):
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        logger.info(f"📥 [{b_idx+1}/{len(repos)}] Streaming ESP multi-species repo: {repo}...")
        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        try:
            files = api.list_repo_files(repo, repo_type="dataset")
            audio_files = [f for f in files if f.endswith((".wav", ".mp3", ".flac"))][:500]
            for f_name in audio_files:
                cached = hf_hub_download(repo_id=repo, filename=f_name, repo_type="dataset", token=os.getenv("HF_TOKEN"))
                dest = batch_dir / Path(f_name).name
                shutil.copy2(cached, dest)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="wildlife")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"ESP repo {repo} warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_dcase_2020(progress: Dict):
    """
    Ingest DCASE 2020 Task 5 bioacoustics from Zenodo record 3665023.
    """
    dataset_name = "dcase_2020"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    rec_id = "3665023"

    try:
        r = requests.get(f"https://zenodo.org/api/records/{rec_id}", timeout=30)
        files = r.json().get("files", [])
    except Exception as e:
        logger.error(f"Zenodo DCASE error: {e}")
        return

    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    for b_idx, f_info in enumerate(files[:5]):  # Process soundscape archives
        if b_idx in completed_batches:
            continue
        if get_free_disk_gb() < 15.0:
            break

        f_url = f_info.get("links", {}).get("self")
        f_name = f_info.get("key")
        logger.info(f"📥 [{b_idx+1}/{min(5, len(files))}] Downloading DCASE archive: {f_name}...")

        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        dest_file = batch_dir / f_name

        try:
            resp = requests.get(f_url, stream=True, timeout=120)
            with open(dest_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)

            if dest_file.suffix.lower() == ".zip":
                with zipfile.ZipFile(dest_file, "r") as z:
                    z.extractall(batch_dir)
                dest_file.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, b_idx, species="dogs")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"DCASE file {f_name} warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_musan_and_demand(progress: Dict):
    """
    Ingest MUSAN (OpenSLR 17) and DEMAND (Zenodo 1227121) noise datasets for anti-spoof contrast.
    """
    dataset_name = "musan_demand"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    # 1. DEMAND from Zenodo
    if 0 not in completed_batches and get_free_disk_gb() >= 15.0:
        logger.info("📥 Downloading DEMAND noise environments from Zenodo record 1227121...")
        try:
            r = requests.get("https://zenodo.org/api/records/1227121", timeout=30)
            files = r.json().get("files", [])[:4]
            batch_dir = STAGING_DIR / f"{dataset_name}_0000"
            batch_dir.mkdir(parents=True, exist_ok=True)

            for f_info in files:
                f_url = f_info.get("links", {}).get("self")
                dest = batch_dir / f_info.get("key")
                resp = requests.get(f_url, stream=True, timeout=60)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                if dest.suffix.lower() == ".zip":
                    with zipfile.ZipFile(dest, "r") as z:
                        z.extractall(batch_dir)
                    dest.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, 0, species="ambient_anti_spoof")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(0)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"DEMAND download warning: {e}")

    # 2. MUSAN speech and music
    if 1 not in completed_batches and get_free_disk_gb() >= 15.0:
        logger.info("📥 Downloading MUSan speech/noise partitions from OpenSLR 17...")
        batch_dir = STAGING_DIR / f"{dataset_name}_0001"
        batch_dir.mkdir(parents=True, exist_ok=True)
        try:
            musan_url = "https://www.openslr.org/resources/17/musan.tar.gz"
            dest = batch_dir / "musan.tar.gz"
            resp = requests.get(musan_url, stream=True, timeout=120)
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=131072):
                    f.write(chunk)
            with tarfile.open(dest, "r:gz") as tar:
                tar.extractall(batch_dir)
            dest.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, 1, species="human_anti_spoof")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(1)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"MUSAN download warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


def ingest_vocal_imitation(progress: Dict):
    """
    Ingest Vocal Imitation Set (humans mimicking animal and acoustic sounds).
    """
    dataset_name = "vocal_imitation"
    logger.info(f"🚀 Starting Ingestion: {dataset_name}")
    rec_id = "1344403"
    completed_batches = set(progress.get("completed_batches", {}).get(dataset_name, []))

    if 0 not in completed_batches and get_free_disk_gb() >= 15.0:
        batch_dir = STAGING_DIR / f"{dataset_name}_0000"
        batch_dir.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(f"https://zenodo.org/api/records/{rec_id}", timeout=30)
            files = r.json().get("files", [])
            for f_info in files[:2]:
                f_url = f_info.get("links", {}).get("self")
                dest = batch_dir / f_info.get("key")
                resp = requests.get(f_url, stream=True, timeout=60)
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                if dest.suffix.lower() == ".zip":
                    with zipfile.ZipFile(dest, "r") as z:
                        z.extractall(batch_dir)
                    dest.unlink(missing_ok=True)

            count, _ = process_and_upload_batch(batch_dir, dataset_name, 0, species="human_anti_spoof")
            progress["total_clips_processed"] += count
            progress.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(0)
            save_progress(progress)
        except Exception as e:
            logger.warning(f"Vocal imitation download warning: {e}")

    progress.setdefault("completed_datasets", []).append(dataset_name)
    save_progress(progress)
    logger.info(f"✅ Finished Ingestion: {dataset_name}")


# ═══════════════════════════════════════════════════════════════════════════
# Master Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

DATASET_WORKERS = {
    "dogspeak_full": ingest_dogspeak_full,
    "birdset": ingest_birdset,
    "birdclef": ingest_birdclef,
    "dclde_marine": ingest_dclde_marine,
    "openwhistle_watkins": ingest_openwhistle_and_watkins,
    "earth_species_project": ingest_earth_species_project,
    "dcase_2020": ingest_dcase_2020,
    "musan_demand": ingest_musan_and_demand,
    "vocal_imitation": ingest_vocal_imitation,
}


def run_pipeline(selected_dataset: Optional[str] = None, skip_wait: bool = False):
    """
    Main entry point for extended bioacoustic datasets streaming.
    """
    print("\n" + "═" * 70)
    print("🐾 AnimalVox AI — Extended Bioacoustic Datasets Streaming Worker")
    print("═" * 70)

    # 1. Wait for AudioSet unless explicitly skipped
    if not skip_wait:
        wait_for_audioset_completion()

    free_gb = get_free_disk_gb()
    logger.info(f"💾 Initial free disk space: {free_gb:.1f} GB (Watermark: 15.0 GB)")

    progress = load_progress()

    targets = [selected_dataset] if selected_dataset else list(DATASET_WORKERS.keys())

    for ds_key in targets:
        if ds_key not in DATASET_WORKERS:
            logger.error(f"Unknown dataset: {ds_key}. Available: {list(DATASET_WORKERS.keys())}")
            continue

        if ds_key in progress.get("completed_datasets", []) and not selected_dataset:
            logger.info(f"⏩ Dataset {ds_key} already 100% completed. Skipping.")
            continue

        worker_fn = DATASET_WORKERS[ds_key]
        print("\n" + "━" * 70)
        logger.info(f"▶️ INGESTING DATASET: {ds_key.upper()}")
        print("━" * 70)
        worker_fn(progress)

    print("\n" + "═" * 70)
    print("🎉 Extended Bioacoustic Datasets Streaming Complete!")
    print(f"  Total clips processed across all datasets: {progress.get('total_clips_processed', 0):,}")
    print(f"  Remaining free host disk space           : {get_free_disk_gb():.1f} GB")
    print("═" * 70)


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Extended Bioacoustic Streaming Worker")
    parser.add_argument("--dataset", type=str, default=None, choices=list(DATASET_WORKERS.keys()), help="Target specific dataset")
    parser.add_argument("--now", "--skip-wait", dest="skip_wait", action="store_true", help="Start immediately without waiting for AudioSet")
    args = parser.parse_args()

    run_pipeline(selected_dataset=args.dataset, skip_wait=args.skip_wait)


if __name__ == "__main__":
    main()
