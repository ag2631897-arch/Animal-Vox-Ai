"""
AnimalVox AI — High-Speed Expanded Cat Audio Ingestion Engine
============================================================
Downloads and extracts 266-dim embeddings for comprehensive feline vocalizations:
1. StephenCHENYL/cat_audio_dataset (Remaining ~4,000 wav files with emotion/intention labels)
2. liladhii/isolated-cat-meows (4,751 clean isolated cat meow clips)

Features are extracted in memory via PyTorch SIMD + batched BEATs,
saved to compressed .npz shards, backed up to Google Drive, and local raw audio purged.
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "all_species_download.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("CatIngestion")

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from training.feature_extractor import extract_batch_waveforms

# Directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "shard_features"
STAGING_DIR = DATA_DIR / "shard_staging" / "cats_expanded"
PROGRESS_FILE = PROCESSED_DIR / "universal_stream_progress.json"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

GDRIVE_FEATURES = "gdrive:AnimalVoxDatasets/features/"
GDRIVE_RAW = "gdrive:AnimalVoxDatasets/raw/cats/"


def upload_to_gdrive(local_path: Path, remote_dest: str):
    import subprocess
    cmd = ["rclone", "copy", str(local_path), remote_dest, "--transfers", "4", "--stats", "0"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if res.returncode == 0:
            logger.info(f"☁️ Uploaded {local_path.name} to {remote_dest}")
        else:
            logger.warning(f"Rclone warning: {res.stderr.strip()[:100]}")
    except Exception as e:
        logger.warning(f"Rclone error: {e}")


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
# 1. StephenCHENYL/cat_audio_dataset (Remaining Batches 3..10)
# ══════════════════════════════════════════════════════════════════

def ingest_stephen_remaining():
    logger.info("🐱 Ingesting remaining batches from StephenCHENYL/cat_audio_dataset...")
    api = HfApi()
    info = api.dataset_info("StephenCHENYL/cat_audio_dataset")
    wav_siblings = [
        s.rfilename for s in info.siblings
        if s.rfilename.startswith("pure_audio/audio/") and s.rfilename.endswith(".wav")
    ]
    logger.info(f"Total StephenCHENYL wav files in repo: {len(wav_siblings)}")

    meta_path = hf_hub_download(
        repo_id="StephenCHENYL/cat_audio_dataset",
        filename="pure_audio/metadata/metadata.json",
        repo_type="dataset"
    )
    meta_json = json.load(open(meta_path, encoding="utf-8"))
    meta_map = {}
    for item in meta_json:
        aid = item.get("id") or Path(item.get("audio_path", "")).stem
        meta_map[aid] = item.get("intention", "affection")

    BATCH_SIZE = 500
    total_batches = (len(wav_siblings) + BATCH_SIZE - 1) // BATCH_SIZE

    # Find which batches already exist on disk
    existing_parts = set()
    for npz_file in FEATURES_DIR.glob("features_cats_stephen_part_*.npz"):
        try:
            part_num = int(npz_file.stem.split("_")[-1])
            existing_parts.add(part_num)
        except Exception:
            pass

    logger.info(f"Existing StephenCHENYL parts: {sorted(list(existing_parts))}")

    for b_idx in range(total_batches):
        if b_idx in existing_parts:
            logger.info(f"⏩ StephenCHENYL batch {b_idx} already exists. Skipping.")
            continue

        b_files = wav_siblings[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        logger.info(f"📥 [StephenCHENYL Batch {b_idx + 1}/{total_batches}] Downloading {len(b_files)} clips...")

        batch_dir = STAGING_DIR / f"stephen_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        def _dl(rfile):
            dest = batch_dir / Path(rfile).name
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            try:
                p = hf_hub_download(repo_id="StephenCHENYL/cat_audio_dataset", filename=rfile, repo_type="dataset")
                shutil.copyfile(p, dest)
                return dest
            except Exception:
                return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_dl, rf) for rf in b_files]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    downloaded.append(res)

        logger.info(f"Downloaded {len(downloaded)} clips for batch {b_idx}. Extracting features...")

        raw_waveforms, labels, files = [], [], []
        for dp in downloaded:
            raw_intent = meta_map.get(dp.stem, "contentment").lower()
            if "rest" in raw_intent or "lying" in raw_intent:
                emo = "calm"
            elif "food" in raw_intent or "eat" in raw_intent or "feed" in raw_intent:
                emo = "food"
            elif "stress" in raw_intent or "pain" in raw_intent:
                emo = "stress_acute"
            elif "greet" in raw_intent or "happy" in raw_intent:
                emo = "bonding"
            else:
                emo = "contentment"

            w = load_audio_file(dp, sr=16000)
            if w is not None:
                raw_waveforms.append(w)
                labels.append(emo)
                files.append(dp.name)

        if raw_waveforms:
            # Batched extraction in chunks of 32
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
                shard_name = f"features_cats_stephen_part_{b_idx:04d}.npz"
                out_npz = FEATURES_DIR / shard_name
                np.savez_compressed(
                    out_npz,
                    embeddings=features_np,
                    labels=np.array(valid_labels),
                    files=np.array(valid_files),
                    metadata=[{"species": "cats", "dataset": "stephen_cat_audio"}],
                )
                logger.info(f"✅ Saved {shard_name} ({len(features)} cat samples)")
                upload_to_gdrive(out_npz, GDRIVE_FEATURES)

        # Cleanup raw audio
        shutil.rmtree(batch_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# 2. liladhii/isolated-cat-meows (4,751 Clean Isolated Meows)
# ══════════════════════════════════════════════════════════════════

def ingest_isolated_cat_meows():
    logger.info("🐾 Ingesting liladhii/isolated-cat-meows (4,751 clean clips)...")
    api = HfApi()
    info = api.dataset_info("liladhii/isolated-cat-meows")
    wav_files = sorted([
        s.rfilename for s in info.siblings
        if s.rfilename.startswith("data/") and s.rfilename.endswith(".wav")
    ])
    logger.info(f"Found {len(wav_files)} isolated cat meow clips on Hugging Face.")

    BATCH_SIZE = 500
    total_batches = (len(wav_files) + BATCH_SIZE - 1) // BATCH_SIZE

    existing_parts = set()
    for npz_file in FEATURES_DIR.glob("features_cats_isolated_part_*.npz"):
        try:
            part_num = int(npz_file.stem.split("_")[-1])
            existing_parts.add(part_num)
        except Exception:
            pass

    for b_idx in range(total_batches):
        if b_idx in existing_parts:
            logger.info(f"⏩ Isolated meows batch {b_idx} already exists. Skipping.")
            continue

        b_files = wav_files[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        logger.info(f"📥 [Isolated Meow Batch {b_idx + 1}/{total_batches}] Downloading {len(b_files)} clips...")

        batch_dir = STAGING_DIR / f"isolated_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        def _dl(rfile):
            dest = batch_dir / Path(rfile).name
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            try:
                p = hf_hub_download(repo_id="liladhii/isolated-cat-meows", filename=rfile, repo_type="dataset")
                shutil.copyfile(p, dest)
                return dest
            except Exception:
                return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_dl, rf) for rf in b_files]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    downloaded.append(res)

        logger.info(f"Downloaded {len(downloaded)} clips for batch {b_idx}. Extracting features...")

        raw_waveforms, labels, files = [], [], []
        for dp in downloaded:
            w = load_audio_file(dp, sr=16000)
            if w is not None:
                raw_waveforms.append(w)
                labels.append("meow")
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
                shard_name = f"features_cats_isolated_part_{b_idx:04d}.npz"
                out_npz = FEATURES_DIR / shard_name
                np.savez_compressed(
                    out_npz,
                    embeddings=features_np,
                    labels=np.array(valid_labels),
                    files=np.array(valid_files),
                    metadata=[{"species": "cats", "dataset": "isolated_cat_meows"}],
                )
                logger.info(f"✅ Saved {shard_name} ({len(features)} isolated cat meows)")
                upload_to_gdrive(out_npz, GDRIVE_FEATURES)

        # Cleanup raw audio
        shutil.rmtree(batch_dir, ignore_errors=True)


def main():
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("🐱 Launching Expanded Feline (Cat) Audio Ingestion Pipeline")
    logger.info("══════════════════════════════════════════════════════════════════")
    t0 = time.time()
    
    # 1. Finish remaining StephenCHENYL cat clips (~4,000 clips)
    ingest_stephen_remaining()

    # 2. Ingest liladhii/isolated-cat-meows (4,751 clips)
    ingest_isolated_cat_meows()

    # Cleanup staging directory
    shutil.rmtree(STAGING_DIR, ignore_errors=True)

    elapsed = time.time() - t0
    logger.info(f"🎉 Cat Ingestion Complete in {elapsed:.1f}s!")
    logger.info("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
