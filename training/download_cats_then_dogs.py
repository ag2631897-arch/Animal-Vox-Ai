"""
AnimalVox AI — Sequential Ingestion Runner: Cats Then DogSpeak
==============================================================
Orchestrates the two requested dataset completions in strict order:
1. StephenCHENYL Cat Audio: remaining 8 batches (Batches 3..10, from video_audio partition, 3,964 clips)
2. DogSpeak Full Archive: remaining 11 batches (Batches 3..13, 11,000 clips with repo path resolution)

Features:
- Batched SIMD extraction with BEATs acoustic backbone
- Non-blocking asynchronous cloud upload to Google Drive
- Immediate staging cache purging after feature extraction
- Persistent progress tracking in universal_stream_progress.json & extended_stream_progress.json
"""

import os
import io
import sys
import json
import time
import shutil
import zipfile
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

LOG_FILE = PROJECT_ROOT / "all_species_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("CatsThenDogsRunner")

import numpy as np
import soundfile as sf
import torch
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from training.feature_extractor import extract_batch_waveforms

# Configure PyTorch CPU threads
if hasattr(torch, "set_num_threads"):
    torch.set_num_threads(os.cpu_count() or 16)

# Directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "shard_features"
STAGING_DIR = DATA_DIR / "shard_staging" / "cats_then_dogs"
UNIVERSAL_PROGRESS_FILE = PROCESSED_DIR / "universal_stream_progress.json"
EXTENDED_PROGRESS_FILE = PROCESSED_DIR / "extended_stream_progress.json"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

GDRIVE_FEATURES = "gdrive:AnimalVoxDatasets/features/"
GDRIVE_RAW_CATS = "gdrive:AnimalVoxDatasets/raw/cats/"
GDRIVE_RAW_DOGS = "gdrive:AnimalVoxDatasets/raw/dogspeak_full/"

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN", "")

# Background upload executor
upload_executor = ThreadPoolExecutor(max_workers=2)


def get_free_disk_gb() -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    return free / (1024 ** 3)


def upload_to_gdrive(local_path: Path, remote_dest: str):
    cmd = ["rclone", "copy", str(local_path), remote_dest, "--transfers", "4", "--stats", "0"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if res.returncode == 0:
            logger.info(f"☁️ Uploaded {local_path.name} to {remote_dest}")
        else:
            logger.warning(f"Rclone warning for {local_path.name}: {res.stderr.strip()[:100]}")
    except Exception as e:
        logger.warning(f"Rclone error for {local_path.name}: {e}")
    finally:
        # If it was a temporary zip file in staging, delete it after upload
        if local_path.suffix == ".zip" and STAGING_DIR in local_path.parents:
            local_path.unlink(missing_ok=True)


def load_audio_file(filepath: Path, sr: int = 16000) -> np.ndarray:
    try:
        data, file_sr = sf.read(str(filepath), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if file_sr != sr:
            import librosa
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        return data
    except Exception as e:
        logger.debug(f"Audio read error for {filepath.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# PHASE 1: StephenCHENYL Cat Audio (Remaining 8 Batches: 3..10)
# ══════════════════════════════════════════════════════════════════

def run_phase_1_cats():
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("🐱 PHASE 1: StephenCHENYL Cat Audio Remaining Batches (3..10)")
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info(f"Current Free Disk Space: {get_free_disk_gb():.1f} GB")

    api = HfApi(token=HF_TOKEN)
    info = api.dataset_info("StephenCHENYL/cat_audio_dataset")
    
    # The first 3 batches (0, 1, 2) covered pure_audio (1,241 clips).
    # The remaining 8 batches (3..10) are in video_audio (3,964 clips).
    video_wavs = sorted([
        s.rfilename for s in info.siblings
        if s.rfilename.startswith("video_audio/audio/") and s.rfilename.endswith(".wav")
    ])
    logger.info(f"Total video_audio cat clips available: {len(video_wavs)}")

    meta_path = hf_hub_download(
        repo_id="StephenCHENYL/cat_audio_dataset",
        filename="video_audio/metadata/metadata.json",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    meta_json = json.load(open(meta_path, encoding="utf-8"))
    meta_map = {}
    for item in meta_json:
        aid = item.get("id") or Path(item.get("audio_path", "")).stem
        meta_map[aid] = item.get("intention", "contentment")

    BATCH_SIZE = 500
    # Batches 3 through 10 (8 batches total)
    total_video_batches = (len(video_wavs) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Partitioned video_audio clips into {total_video_batches} batches (Batches 3..{2 + total_video_batches})")

    for local_idx in range(total_video_batches):
        b_idx = 3 + local_idx
        shard_name = f"features_cats_stephen_part_{b_idx:04d}.npz"
        out_npz = FEATURES_DIR / shard_name

        if out_npz.exists() and out_npz.stat().st_size > 1000:
            logger.info(f"⏩ StephenCHENYL batch {b_idx} already exists ({shard_name}). Skipping.")
            continue

        if get_free_disk_gb() < 15.0:
            raise RuntimeError(f"Disk space low ({get_free_disk_gb():.1f} GB). Pausing.")

        b_files = video_wavs[local_idx * BATCH_SIZE : (local_idx + 1) * BATCH_SIZE]
        logger.info(f"📥 [StephenCHENYL Batch {b_idx}/10] Downloading {len(b_files)} clips...")

        batch_dir = STAGING_DIR / f"stephen_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        def _dl(rfile):
            dest = batch_dir / Path(rfile).name
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            try:
                p = hf_hub_download(
                    repo_id="StephenCHENYL/cat_audio_dataset",
                    filename=rfile,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                shutil.copyfile(p, dest)
                return dest
            except Exception as e:
                logger.debug(f"Cat download error: {e}")
                return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_dl, rf) for rf in b_files]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    downloaded.append(res)

        logger.info(f"Downloaded {len(downloaded)} clips for batch {b_idx}. Extracting 266-dim features...")

        raw_waveforms, labels, files = [], [], []
        for dp in downloaded:
            raw_intent = str(meta_map.get(dp.stem, "contentment")).lower()
            if any(k in raw_intent for k in ["rest", "lying", "inactive", "sleep"]):
                emo = "calm"
            elif any(k in raw_intent for k in ["food", "eat", "feed", "hungry"]):
                emo = "food"
            elif any(k in raw_intent for k in ["stress", "pain", "alarm", "fear", "anger"]):
                emo = "stress_acute"
            elif any(k in raw_intent for k in ["greet", "happy", "rub", "purr", "bonding", "affection"]):
                emo = "bonding"
            else:
                emo = "contentment"

            w = load_audio_file(dp, sr=16000)
            if w is not None:
                raw_waveforms.append(w)
                labels.append(emo)
                files.append(dp.name)

        if raw_waveforms:
            features = []
            valid_labels = []
            valid_files = []
            CHUNK = 32
            for i in range(0, len(raw_waveforms), CHUNK):
                chunk_w = raw_waveforms[i : i + CHUNK]
                chunk_feats = extract_batch_waveforms(chunk_w, sr=16000)
                for f_idx, feat in enumerate(chunk_feats):
                    if feat is not None:
                        features.append(feat)
                        valid_labels.append(labels[i + f_idx])
                        valid_files.append(files[i + f_idx])

            if features:
                features_np = np.stack(features).astype(np.float32)
                np.savez_compressed(
                    out_npz,
                    embeddings=features_np,
                    labels=np.array(valid_labels),
                    files=np.array(valid_files),
                    metadata=[{"species": "cats", "dataset": "stephen_cat_audio"}],
                )
                logger.info(f"✅ Saved {shard_name} ({len(features)} cat samples)")
                upload_executor.submit(upload_to_gdrive, out_npz, GDRIVE_FEATURES)

                # Package raw audio to zip and upload
                zip_path = STAGING_DIR / f"cats_stephen_raw_{b_idx:04d}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    for dp in downloaded:
                        z.write(dp, arcname=dp.name)
                upload_executor.submit(upload_to_gdrive, zip_path, GDRIVE_RAW_CATS)

        # Purge local staging raw files
        shutil.rmtree(batch_dir, ignore_errors=True)

        # Update universal progress
        if UNIVERSAL_PROGRESS_FILE.exists():
            try:
                with open(UNIVERSAL_PROGRESS_FILE, "r", encoding="utf-8") as f:
                    u_prog = json.load(f)
                u_prog["total_clips_processed"] = u_prog.get("total_clips_processed", 0) + len(features)
                u_prog["cats_batches"] = max(u_prog.get("cats_batches", 0), b_idx + 1)
                with open(UNIVERSAL_PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(u_prog, f, indent=2)
            except Exception:
                pass

    logger.info("🎉 PHASE 1 COMPLETE: All 11 batches of StephenCHENYL Cat Audio (5,205 clips) are fully ingested!")


# ══════════════════════════════════════════════════════════════════
# PHASE 2: DogSpeak Full Archive (Remaining 11 Batches: 3..13)
# ══════════════════════════════════════════════════════════════════

def run_phase_2_dogs():
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("🐕 PHASE 2: DogSpeak Full Archive Remaining Batches (3..13)")
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info(f"Current Free Disk Space: {get_free_disk_gb():.1f} GB")

    dataset_name = "dogspeak_full"

    api = HfApi(token=HF_TOKEN)
    logger.info("🔍 Indexing repository files for ArlingtonCL2/DogSpeak_Dataset...")
    repo_files = api.list_repo_files("ArlingtonCL2/DogSpeak_Dataset", repo_type="dataset")
    basename_to_path = {os.path.basename(f): f for f in repo_files if f.endswith(".wav")}
    logger.info(f"Indexed {len(basename_to_path)} DogSpeak audio files in repo.")

    meta_path = hf_hub_download(
        repo_id="ArlingtonCL2/DogSpeak_Dataset",
        filename="metadata.csv",
        repo_type="dataset",
        token=HF_TOKEN,
    )
    df = pd.read_csv(meta_path)
    total_files = len(df)
    batch_size = 1000
    total_batches = (total_files + batch_size - 1) // batch_size
    logger.info(f"DogSpeak metadata: {total_files} audio files across {total_batches} batches.")

    # Load extended stream progress
    ext_prog = {}
    if EXTENDED_PROGRESS_FILE.exists():
        try:
            with open(EXTENDED_PROGRESS_FILE, "r", encoding="utf-8") as f:
                ext_prog = json.load(f)
        except Exception:
            pass

    completed_batches = set(ext_prog.get("completed_batches", {}).get(dataset_name, []))
    missing_batches = [b for b in range(total_batches) if b not in completed_batches]
    logger.info(f"DogSpeak completed batches count: {len(completed_batches)}/{total_batches}")
    logger.info(f"DogSpeak missing batches to process: {missing_batches}")

    for b_idx in missing_batches:
        if get_free_disk_gb() < 15.0:
            raise RuntimeError(f"Disk space low ({get_free_disk_gb():.1f} GB). Pausing.")

        shard_name = f"features_dogspeak_full_part_{b_idx:04d}.npz"
        out_npz = FEATURES_DIR / shard_name

        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_files)
        batch_df = df.iloc[start_i:end_i]

        batch_dir = STAGING_DIR / f"{dataset_name}_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📥 [DogSpeak Batch {b_idx + 1}/{total_batches}] Downloading {len(batch_df)} files...")

        def _fetch(row):
            fname = row.get("filename", "")
            if not fname:
                return None
            rel_path = basename_to_path.get(fname)
            if not rel_path:
                dog_id = row.get("dog_id", "")
                rel_path = f"dogspeak_released/{dog_id}/{fname}"

            dest = batch_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                breed = str(row.get("breed", "unknown")).lower().strip().replace(" ", "_")
                return (dest, breed, fname)
            try:
                cached = hf_hub_download(
                    repo_id="ArlingtonCL2/DogSpeak_Dataset",
                    filename=rel_path,
                    repo_type="dataset",
                    token=HF_TOKEN,
                )
                shutil.copy2(cached, dest)
                breed = str(row.get("breed", "unknown")).lower().strip().replace(" ", "_")
                return (dest, breed, fname)
            except Exception as e:
                logger.debug(f"DogSpeak fetch failed for {rel_path}: {e}")
                return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_fetch, row) for _, row in batch_df.iterrows()]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    downloaded.append(res)

        logger.info(f"Downloaded {len(downloaded)} dog clips for batch {b_idx}. Extracting 266-dim features...")

        raw_waveforms, labels, files = [], [], []
        for dest, breed, fname in downloaded:
            w = load_audio_file(dest, sr=16000)
            if w is not None:
                raw_waveforms.append(w)
                labels.append(breed)
                files.append(fname)

        if raw_waveforms:
            features = []
            valid_labels = []
            valid_files = []
            CHUNK = 32
            for i in range(0, len(raw_waveforms), CHUNK):
                chunk_w = raw_waveforms[i : i + CHUNK]
                chunk_feats = extract_batch_waveforms(chunk_w, sr=16000)
                for f_idx, feat in enumerate(chunk_feats):
                    if feat is not None:
                        features.append(feat)
                        valid_labels.append(labels[i + f_idx])
                        valid_files.append(files[i + f_idx])
                if len(features) % 128 < CHUNK or (i + CHUNK) >= len(raw_waveforms):
                    logger.info(f"  ⚡ [DogSpeak Batch {b_idx}] Extracted {len(features)}/{len(raw_waveforms)} features ({len(features)/len(raw_waveforms)*100:.0f}%)...")
                    for h in logger.handlers:
                        h.flush()

            if features:
                features_np = np.stack(features).astype(np.float32)
                np.savez_compressed(
                    out_npz,
                    embeddings=features_np,
                    labels=np.array(valid_labels),
                    files=np.array(valid_files),
                    metadata=[{"species": "dogs", "dataset": "dogspeak_full"}],
                )
                logger.info(f"✅ Saved {shard_name} ({len(features)} dog barks)")
                upload_executor.submit(upload_to_gdrive, out_npz, GDRIVE_FEATURES)

                # Package raw audio to zip and upload
                zip_path = STAGING_DIR / f"{dataset_name}_raw_{b_idx:04d}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    for dest, _, fname in downloaded:
                        z.write(dest, arcname=fname)
                upload_executor.submit(upload_to_gdrive, zip_path, GDRIVE_RAW_DOGS)

                # Update progress
                ext_prog.setdefault("completed_batches", {}).setdefault(dataset_name, []).append(b_idx)
                ext_prog["total_clips_processed"] = ext_prog.get("total_clips_processed", 0) + len(features)
                ext_prog["last_update"] = time.time()
                with open(EXTENDED_PROGRESS_FILE, "w", encoding="utf-8") as f:
                    json.dump(ext_prog, f, indent=2)

        # Purge local staging raw files
        shutil.rmtree(batch_dir, ignore_errors=True)

    # Mark completed dataset
    completed_now = len(ext_prog.get("completed_batches", {}).get(dataset_name, []))
    if completed_now >= total_batches:
        if dataset_name not in ext_prog.get("completed_datasets", []):
            ext_prog.setdefault("completed_datasets", []).append(dataset_name)
            with open(EXTENDED_PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(ext_prog, f, indent=2)
        logger.info(f"🎉 PHASE 2 COMPLETE: DogSpeak Full Archive 100% Ingested ({completed_now}/{total_batches} batches)!")


# ══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════

def main():
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("🚀 Launching Sequential Ingestion: StephenCHENYL Cats → DogSpeak")
    logger.info("══════════════════════════════════════════════════════════════════")
    t0 = time.time()

    # Step 1: Cats (Remaining 8 batches of StephenCHENYL)
    run_phase_1_cats()

    # Step 2: Dogs (Remaining 11 batches of DogSpeak)
    run_phase_2_dogs()

    # Flush all cloud uploads
    logger.info("⏳ Waiting for all background Google Drive uploads to complete...")
    upload_executor.shutdown(wait=True)

    # Cleanup staging directory
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    elapsed = time.time() - t0
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info(f"🎉 COMPLETE! Both Cats & DogSpeak ingested successfully in {elapsed / 60:.1f} minutes!")
    logger.info(f"Remaining Free Host Disk Space: {get_free_disk_gb():.1f} GB")
    logger.info("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
