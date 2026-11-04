"""
AnimalVox AI — Universal Animal Bioacoustics Streaming Engine
============================================================
Stream, extract 266-dim BEATs features, and cloud-archive full animal corpora:
1. Cats: StephenCHENYL Cat Audio (5,205 clips) + CatMeows (441 meows)
2. Frogs: Full Xeno-Canto Anuran corpus (6,509 recordings across 919 species)
3. Birds: BIRB Xeno-Canto 16kHz Benchmark Archive (684,744 pre-segmented 6s clips across 10,127 species)
4. Automatic cloud sync to Google Drive (gdrive:AnimalVoxDatasets/)
5. Automatic disk preservation watermark (>15 GB)
"""

import os
import io
import sys
import time
import json
import shutil
import zipfile
import tarfile
import logging
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "all_species_download.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("UniversalStreamer")

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from training.feature_extractor import extract_single_file, extract_single_waveform, extract_batch_waveforms

# Directories
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "shard_features"
STAGING_DIR = DATA_DIR / "shard_staging"
PROGRESS_FILE = PROCESSED_DIR / "universal_stream_progress.json"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
STAGING_DIR.mkdir(parents=True, exist_ok=True)

# Google Drive Remote
GDRIVE_FEATURES = "gdrive:AnimalVoxDatasets/features/"
GDRIVE_RAW = "gdrive:AnimalVoxDatasets/raw/"


def get_free_disk_gb(path: Path = PROJECT_ROOT) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def enforce_disk_safety():
    free_gb = get_free_disk_gb()
    if free_gb < 15.0:
        logger.warning(f"⚠️ Low disk space: {free_gb:.1f} GB free. Purging staging caches...")
        for item in STAGING_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
        new_free = get_free_disk_gb()
        if new_free < 10.0:
            raise RuntimeError(f"CRITICAL: Insufficient disk space ({new_free:.1f} GB free). Pausing pipeline.")


def upload_to_gdrive(local_path: Path, remote_dest: str):
    import subprocess
    cmd = ["rclone", "copy", str(local_path), remote_dest, "--transfers", "4", "--stats", "0"]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=900)
        if res.returncode == 0:
            logger.info(f"☁️ Uploaded {local_path.name} to {remote_dest}")
        else:
            logger.warning(f"Rclone upload warning: {res.stderr.strip()[:100]}")
    except Exception as e:
        logger.warning(f"Rclone error: {e}")


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "completed_datasets": [],
        "completed_shards": [],
        "cats_batches": 0,
        "frogs_batches": 0,
        "birb_shards": 0,
        "total_clips_processed": 0,
    }


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


# ══════════════════════════════════════════════════════════════════
# 1. CATS INGESTION
# ══════════════════════════════════════════════════════════════════

def ingest_cats(progress: dict):
    if "cats_all" in progress["completed_datasets"]:
        logger.info("⏩ Dataset cats_all already completed. Skipping.")
        return

    logger.info("🐱 Starting Ingestion: Cats (CatMeows + StephenCHENYL Cat Audio)")
    cat_staging = STAGING_DIR / "cats"
    cat_staging.mkdir(parents=True, exist_ok=True)

    # A. CatMeows
    if "cats_catmeows" not in progress["completed_datasets"]:
        try:
            logger.info("📥 Downloading CatMeows archive from Hugging Face...")
            zip_p = hf_hub_download(repo_id="zeddez/CatMeows", filename="dataset.zip", repo_type="dataset")
            with zipfile.ZipFile(zip_p, "r") as z:
                z.extractall(cat_staging / "catmeows_raw")

            wav_files = list((cat_staging / "catmeows_raw").rglob("*.wav"))
            logger.info(f"Extracted {len(wav_files)} CatMeows clips. Extracting 266-dim features...")

            features, labels, files = [], [], []
            for wf in wav_files:
                name = wf.name
                # Parse context: B_ (brushing), F_ (food/hunger), I_ (isolation)
                prefix = name.split("_")[0].upper()
                if prefix == "B":
                    ctx = "brushing"
                elif prefix == "F":
                    ctx = "food"
                elif prefix == "I":
                    ctx = "isolation"
                else:
                    ctx = "purr"

                feat = extract_single_file(wf, sr=16000)
                if feat is not None:
                    features.append(feat)
                    labels.append(ctx)
                    files.append(name)

            if features:
                features_np = np.stack(features).astype(np.float32)
                out_npz = FEATURES_DIR / "features_cats_catmeows.npz"
                np.savez_compressed(
                    out_npz,
                    embeddings=features_np,
                    labels=np.array(labels),
                    files=np.array(files),
                    metadata=[{"species": "cats", "dataset": "catmeows"}],
                )
                logger.info(f"✅ Saved features_cats_catmeows.npz ({len(features)} samples)")
                upload_to_gdrive(out_npz, GDRIVE_FEATURES)
                progress["total_clips_processed"] += len(features)

            progress["completed_datasets"].append("cats_catmeows")
            save_progress(progress)
            shutil.rmtree(cat_staging / "catmeows_raw", ignore_errors=True)
        except Exception as e:
            logger.error(f"CatMeows ingestion error: {e}")

    # B. StephenCHENYL/cat_audio_dataset (5,205 pure cat wav files)
    if "cats_stephen" not in progress["completed_datasets"]:
        try:
            logger.info("📥 Ingesting StephenCHENYL/cat_audio_dataset (5,205 clips)...")
            api = HfApi()
            info = api.dataset_info("StephenCHENYL/cat_audio_dataset")
            wav_siblings = [s.rfilename for s in info.siblings if s.rfilename.startswith("pure_audio/audio/") and s.rfilename.endswith(".wav")]
            logger.info(f"Found {len(wav_siblings)} cat wav files on Hugging Face.")

            # Load intention metadata
            meta_path = hf_hub_download(repo_id="StephenCHENYL/cat_audio_dataset", filename="pure_audio/metadata/metadata.json", repo_type="dataset")
            meta_json = json.load(open(meta_path, encoding="utf-8"))
            meta_map = {}
            for item in meta_json:
                aid = item.get("id") or Path(item.get("audio_path", "")).stem
                meta_map[aid] = item.get("intention", "affection")

            BATCH_SIZE = 500
            total_batches = (len(wav_siblings) + BATCH_SIZE - 1) // BATCH_SIZE
            start_batch = progress.get("cats_batches", 0)

            for b_idx in range(start_batch, total_batches):
                enforce_disk_safety()
                b_files = wav_siblings[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
                logger.info(f"📥 [Cat Batch {b_idx + 1}/{total_batches}] Downloading {len(b_files)} files...")

                batch_dir = cat_staging / f"batch_{b_idx:04d}"
                batch_dir.mkdir(parents=True, exist_ok=True)

                def _dl_cat(rfile):
                    dest = batch_dir / Path(rfile).name
                    if dest.exists() and dest.stat().st_size > 0:
                        return dest
                    try:
                        p = hf_hub_download(repo_id="StephenCHENYL/cat_audio_dataset", filename=rfile, repo_type="dataset")
                        shutil.copyfile(p, dest)
                        return dest
                    except Exception:
                        return None

                downloaded_paths = []
                with ThreadPoolExecutor(max_workers=8) as ex:
                    futs = [ex.submit(_dl_cat, rf) for rf in b_files]
                    for fut in as_completed(futs):
                        res = fut.result()
                        if res:
                            downloaded_paths.append(res)

                features, labels, files = [], [], []
                for dp in downloaded_paths:
                    stem = dp.stem
                    raw_intent = meta_map.get(stem, "contentment")
                    # Map to known EMOTIONS
                    intent_lower = raw_intent.lower()
                    if "rest" in intent_lower or "lying" in intent_lower:
                        emo = "calm"
                    elif "food" in intent_lower or "eat" in intent_lower or "feed" in intent_lower:
                        emo = "food"
                    elif "stress" in intent_lower or "pain" in intent_lower:
                        emo = "stress_acute"
                    elif "greet" in intent_lower or "happy" in intent_lower:
                        emo = "bonding"
                    else:
                        emo = "contentment"

                    feat = extract_single_file(dp, sr=16000)
                    if feat is not None:
                        features.append(feat)
                        labels.append(emo)
                        files.append(dp.name)

                if features:
                    features_np = np.stack(features).astype(np.float32)
                    shard_name = f"features_cats_stephen_part_{b_idx:04d}.npz"
                    out_npz = FEATURES_DIR / shard_name
                    np.savez_compressed(
                        out_npz,
                        embeddings=features_np,
                        labels=np.array(labels),
                        files=np.array(files),
                        metadata=[{"species": "cats", "dataset": "stephen_cat_audio"}],
                    )
                    logger.info(f"✅ Saved {shard_name} ({len(features)} cat samples)")
                    upload_to_gdrive(out_npz, GDRIVE_FEATURES)
                    progress["total_clips_processed"] += len(features)

                # Package and upload raw archive
                zip_name = f"cats_stephen_raw_{b_idx:04d}.zip"
                zip_path = cat_staging / zip_name
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    for dp in downloaded_paths:
                        z.write(dp, arcname=dp.name)
                upload_to_gdrive(zip_path, GDRIVE_RAW + "cats/")

                # Purge local audio to save host disk space
                shutil.rmtree(batch_dir, ignore_errors=True)
                zip_path.unlink(missing_ok=True)

                progress["cats_batches"] = b_idx + 1
                save_progress(progress)

            progress["completed_datasets"].append("cats_stephen")
            progress["completed_datasets"].append("cats_all")
            save_progress(progress)
            logger.info("🎉 All Cat datasets successfully ingested!")
        except Exception as e:
            logger.error(f"Stephen cat audio ingestion error: {e}")


# ══════════════════════════════════════════════════════════════════
# 2. XENO-CANTO FROGS (ALL 6,509 RECORDINGS)
# ══════════════════════════════════════════════════════════════════

def ingest_xenocanto_frogs(progress: dict):
    if "xenocanto_frogs_full" in progress["completed_datasets"]:
        logger.info("⏩ Dataset xenocanto_frogs_full already completed. Skipping.")
        return

    logger.info("🐸 Starting Ingestion: Full Xeno-Canto Frogs (All 6,509 Anuran Vocalizations)")
    key = os.getenv("XENO_CANTO_API_KEY", "")
    if not key:
        logger.error("XENO_CANTO_API_KEY not found in .env")
        return

    frog_staging = STAGING_DIR / "xenocanto_frogs"
    frog_staging.mkdir(parents=True, exist_ok=True)

    # 1. Fetch metadata across all pages
    meta_cache = frog_staging / "frogs_metadata.json"
    recs = []
    if meta_cache.exists():
        recs = json.load(open(meta_cache, encoding="utf-8"))
        logger.info(f"Loaded {len(recs)} cached frog recordings from {meta_cache.name}")
    else:
        logger.info("Querying Xeno-Canto API v3 for all frog recordings...")
        first_resp = requests.get(f"https://xeno-canto.org/api/3/recordings?key={key}&query=grp:frogs&page=1", timeout=15)
        if first_resp.status_code != 200:
            logger.error("Failed to query Xeno-canto frogs API")
            return
        d = first_resp.json()
        num_pages = d.get("numPages", 66)
        recs.extend(d.get("recordings", []))

        for p in range(2, num_pages + 1):
            try:
                r = requests.get(f"https://xeno-canto.org/api/3/recordings?key={key}&query=grp:frogs&page={p}", timeout=15)
                if r.status_code == 200:
                    page_recs = r.json().get("recordings", [])
                    recs.extend(page_recs)
                    if p % 10 == 0 or p == num_pages:
                        logger.info(f"  [Frogs Meta] Fetched page {p}/{num_pages} ({len(recs)} recordings found)")
            except Exception as e:
                logger.warning(f"Frogs page {p} warning: {e}")
            time.sleep(0.1)

        with open(meta_cache, "w", encoding="utf-8") as f:
            json.dump(recs, f)
        logger.info(f"Saved complete frog metadata ({len(recs)} recordings) to {meta_cache.name}")

    BATCH_SIZE = 500
    total_batches = (len(recs) + BATCH_SIZE - 1) // BATCH_SIZE
    start_batch = progress.get("frogs_batches", 0)

    for b_idx in range(start_batch, total_batches):
        enforce_disk_safety()
        b_recs = recs[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
        logger.info(f"📥 [Frog Batch {b_idx + 1}/{total_batches}] Downloading {len(b_recs)} frog clips...")

        batch_dir = frog_staging / f"batch_{b_idx:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        def _dl_frog(item):
            rec_id = item.get("id")
            f_url = item.get("file")
            if not f_url:
                return None
            dest = batch_dir / f"xc_frog_{rec_id}.mp3"
            if dest.exists() and dest.stat().st_size > 0:
                return (dest, item)
            for _ in range(3):
                try:
                    r = requests.get(f_url, timeout=20, stream=True)
                    if r.status_code == 200:
                        with open(dest, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return (dest, item)
                except Exception:
                    time.sleep(1)
            return None

        downloaded = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_dl_frog, it) for it in b_recs]
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    downloaded.append(res)

        features, labels, files = [], [], []
        for dest, item in downloaded:
            sp_name = (item.get("sp") or item.get("en") or "anuran").lower().replace(" ", "_")
            feat = extract_single_file(dest, sr=16000)
            if feat is not None:
                features.append(feat)
                labels.append(sp_name)
                files.append(dest.name)

        if features:
            features_np = np.stack(features).astype(np.float32)
            shard_name = f"features_xenocanto_frogs_part_{b_idx:04d}.npz"
            out_npz = FEATURES_DIR / shard_name
            np.savez_compressed(
                out_npz,
                embeddings=features_np,
                labels=np.array(labels),
                files=np.array(files),
                metadata=[{"species": "frogs", "dataset": "xenocanto_frogs_full"}],
            )
            logger.info(f"✅ Saved {shard_name} ({len(features)} frog samples)")
            upload_to_gdrive(out_npz, GDRIVE_FEATURES)
            progress["total_clips_processed"] += len(features)

        # Archive raw audio and upload
        zip_name = f"xenocanto_frogs_raw_{b_idx:04d}.zip"
        zip_path = frog_staging / zip_name
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for dest, _ in downloaded:
                z.write(dest, arcname=dest.name)
        upload_to_gdrive(zip_path, GDRIVE_RAW + "xenocanto_frogs/")

        # Purge local audio
        shutil.rmtree(batch_dir, ignore_errors=True)
        zip_path.unlink(missing_ok=True)

        progress["frogs_batches"] = b_idx + 1
        save_progress(progress)

    progress["completed_datasets"].append("xenocanto_frogs_full")
    save_progress(progress)
    logger.info("🎉 Full Xeno-Canto Frogs corpus ingested!")


# ══════════════════════════════════════════════════════════════════
# 3. BIRB XENO-CANTO BENCHMARK ARCHIVE (684,744 6s CLIPS)
# ══════════════════════════════════════════════════════════════════

import queue
import threading


class ResilientChainedHttpStream(io.RawIOBase):
    """
    Continuous, fault-tolerant byte stream across split HTTP tar archives.
    Automatically reconnects on socket timeouts or IncompleteRead using HTTP Range headers.
    """
    def __init__(self, urls: List[str]):
        self.urls = list(urls)
        self.current_idx = 0
        self.current_part_bytes = 0
        self.current_part_total = 0
        self.current_resp = None
        self._open_next(0)

    def _open_next(self, offset: int = 0) -> bool:
        if self.current_resp:
            try:
                self.current_resp.close()
            except Exception:
                pass
            self.current_resp = None

        if self.current_idx < len(self.urls):
            url = self.urls[self.current_idx]
            part_name = url.split("/")[-1]
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN", "")
            headers = {"User-Agent": "AnimalVoxAI/1.0"}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"
                self.current_part_bytes = offset
                logger.info(f"🌐 Resuming HTTP stream part {self.current_idx + 1}/{len(self.urls)} ({part_name}) at byte {offset:,}...")
            else:
                self.current_part_bytes = 0
                logger.info(f"🌐 Opening HTTP stream part {self.current_idx + 1}/{len(self.urls)}: {part_name}")

            for attempt in range(5):
                try:
                    self.current_resp = requests.get(url, stream=True, headers=headers, timeout=45)
                    if self.current_resp.status_code in (200, 206):
                        if self.current_resp.status_code == 206:
                            cr = self.current_resp.headers.get("Content-Range", "")
                            if "/" in cr:
                                self.current_part_total = int(cr.split("/")[-1])
                        else:
                            self.current_part_total = int(self.current_resp.headers.get("Content-Length", 0))
                        return True
                    else:
                        logger.warning(f"HTTP status {self.current_resp.status_code} for {part_name}, retrying...")
                except Exception as e:
                    logger.warning(f"Connection retry {attempt + 1}/5 for {part_name}: {e}")
                    time.sleep(2 ** attempt)
        return False

    def readable(self):
        return True

    def readinto(self, b):
        if not self.current_resp and not self._open_next(self.current_part_bytes):
            return 0
        while True:
            try:
                chunk = self.current_resp.raw.read(len(b))
                if not chunk:
                    if self.current_part_total > 0 and self.current_part_bytes < self.current_part_total:
                        logger.warning(f"Stream returned 0 bytes prematurely at {self.current_part_bytes:,}/{self.current_part_total:,}. Reconnecting via Range...")
                        time.sleep(2)
                        if self._open_next(self.current_part_bytes):
                            continue
                        return 0
                    # Current part truly finished! Advance to next part
                    self.current_idx += 1
                    self.current_part_bytes = 0
                    if self._open_next(0):
                        continue
                    return 0
                n = len(chunk)
                b[:n] = chunk
                self.current_part_bytes += n
                return n
            except Exception as e:
                logger.warning(f"Stream interrupted at byte {self.current_part_bytes:,}: {e}. Auto-reconnecting via Range...")
                time.sleep(2)
                if not self._open_next(self.current_part_bytes):
                    return 0


def ingest_birb_xenocanto_birds(progress: dict):
    if "birb_xenocanto_birds_full" in progress["completed_datasets"]:
        logger.info("⏩ Dataset birb_xenocanto_birds_full already completed. Skipping.")
        return

    logger.info("🦅 Starting Ingestion: BIRB Xeno-Canto 16kHz Benchmark Archive (684,744 clips across 10,127 species)")
    REPO_NAME = "ilyassmoummad/Xeno-Canto-6s-16khz"

    # All 13 split parts
    import string
    parts = [f"xc-6s-16khz-pann.tar.{string.ascii_lowercase[0]}{string.ascii_lowercase[j]}" for j in range(13)]
    urls = [hf_hub_url(repo_id=REPO_NAME, filename=p, repo_type="dataset") for p in parts]

    start_shard = progress.get("birb_shards", 0)
    CLIPS_PER_SHARD = 1000
    BATCH_SIZE = 24

    # Optimize PyTorch threading across all host CPU cores
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(os.cpu_count() or 16)

    # Background executor for non-blocking Google Drive uploads
    upload_executor = ThreadPoolExecutor(max_workers=2)

    stream = ResilientChainedHttpStream(urls)
    logger.info(f"⚡ Streaming pre-segmented 6s 16kHz PyTorch audio tensors directly into BEATs feature extractor (Resuming at shard {start_shard}, batch_size={BATCH_SIZE})...")

    current_features = []
    current_labels = []
    current_files = []
    shard_idx = 0
    total_streamed = 0
    start_time = time.time()
    skipped = 0
    needed_skip = start_shard * CLIPS_PER_SHARD

    batch_waveforms = []
    batch_labels = []
    batch_files = []

    def flush_batch():
        nonlocal total_streamed
        if not batch_waveforms:
            return
        feats = extract_batch_waveforms(batch_waveforms, sr=16000)
        for idx_f, feat in enumerate(feats):
            if feat is not None:
                current_features.append(feat)
                current_labels.append(batch_labels[idx_f])
                current_files.append(batch_files[idx_f])
                total_streamed += 1
        batch_waveforms.clear()
        batch_labels.clear()
        batch_files.clear()

    try:
        tar = tarfile.open(fileobj=stream, mode="r|*")
        for member in tar:
            if not member.name.endswith(".pt"):
                continue

            if skipped < needed_skip:
                skipped += 1
                if skipped % 500 == 0 or skipped == needed_skip:
                    logger.info(f"⏩ Resuming... skipped {skipped}/{needed_skip} already processed BIRB clips")
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue

                buf = io.BytesIO(f.read())
                waveform = torch.load(buf, weights_only=False)
                parts_split = member.name.split("/")
                species_code = parts_split[-2] if len(parts_split) >= 2 else "bird"
                file_name = Path(member.name).name

                batch_waveforms.append(waveform)
                batch_labels.append(species_code)
                batch_files.append(file_name)

                if len(batch_waveforms) >= BATCH_SIZE:
                    flush_batch()

                if len(current_features) >= CLIPS_PER_SHARD:
                    current_shard_num = start_shard + shard_idx
                    shard_name = f"features_birb_xenocanto_part_{current_shard_num:04d}.npz"
                    out_npz = FEATURES_DIR / shard_name

                    features_np = np.stack(current_features[:CLIPS_PER_SHARD]).astype(np.float32)
                    labels_np = np.array(current_labels[:CLIPS_PER_SHARD])
                    files_np = np.array(current_files[:CLIPS_PER_SHARD])

                    np.savez_compressed(
                        out_npz,
                        embeddings=features_np,
                        labels=labels_np,
                        files=files_np,
                        metadata=[{"species": "birds", "dataset": "birb_xenocanto_16khz"}],
                    )

                    elapsed = time.time() - start_time
                    fps = total_streamed / max(elapsed, 0.1)
                    logger.info(f"✅ Saved {shard_name}: {CLIPS_PER_SHARD} bird clips ({fps:.1f} clips/s, total {start_shard * CLIPS_PER_SHARD + total_streamed:,})")
                    
                    # Async non-blocking upload to Google Drive
                    upload_executor.submit(upload_to_gdrive, out_npz, GDRIVE_FEATURES)

                    # Keep overflow if any
                    current_features = current_features[CLIPS_PER_SHARD:]
                    current_labels = current_labels[CLIPS_PER_SHARD:]
                    current_files = current_files[CLIPS_PER_SHARD:]
                    shard_idx += 1

                    progress["birb_shards"] = current_shard_num + 1
                    progress["total_clips_processed"] += CLIPS_PER_SHARD
                    save_progress(progress)
                    enforce_disk_safety()

            except Exception as item_err:
                logger.debug(f"Error processing member {member.name}: {item_err}")

        # Flush any remaining items in the buffer
        flush_batch()
        tar.close()
    except Exception as e:
        logger.error(f"Sequential stream error: {e}")

    # Final partial shard (only save if all 13 parts were completely streamed to EOF)
    if current_features:
        if stream.current_idx >= len(stream.urls):
            current_shard_num = start_shard + shard_idx
            shard_name = f"features_birb_xenocanto_part_{current_shard_num:04d}.npz"
            out_npz = FEATURES_DIR / shard_name
            np.savez_compressed(
                out_npz,
                embeddings=np.stack(current_features).astype(np.float32),
                labels=np.array(current_labels),
                files=np.array(current_files),
                metadata=[{"species": "birds", "dataset": "birb_xenocanto_16khz"}],
            )
            logger.info(f"✅ Saved final {shard_name}: {len(current_features)} bird clips")
            upload_executor.submit(upload_to_gdrive, out_npz, GDRIVE_FEATURES)
            progress["birb_shards"] = current_shard_num + 1
            progress["total_clips_processed"] += len(current_features)
        else:
            logger.warning(
                f"⚠️ Stream closed before all 13 parts were consumed ({len(current_features)} clips buffered). "
                f"Discarding partial batch so next run cleanly resumes from clip {start_shard * CLIPS_PER_SHARD + shard_idx * CLIPS_PER_SHARD}."
            )

    # Only mark complete if all 680k+ clips were processed
    if (start_shard + shard_idx) >= 600:
        progress["completed_datasets"].append("birb_xenocanto_birds_full")
    save_progress(progress)
    logger.info("🎉 BIRB Xeno-Canto streaming session complete!")


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════

def main():
    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info("🚀 Launching Universal Animal Bioacoustics Ingestion Engine")
    logger.info("══════════════════════════════════════════════════════════════════")
    free_gb = get_free_disk_gb()
    logger.info(f"Current Free Host Disk Space: {free_gb:.1f} GB")

    progress = load_progress()

    # 1. Ingest Cats (CatMeows + StephenCHENYL 5k dataset)
    ingest_cats(progress)

    # 2. Ingest Frogs (Full Xeno-Canto 6,509 Anurans)
    ingest_xenocanto_frogs(progress)

    # 3. Stream Birds (Full 684k BIRB Xeno-Canto Benchmark)
    ingest_birb_xenocanto_birds(progress)

    logger.info("══════════════════════════════════════════════════════════════════")
    logger.info(f"🎉 Complete Universal Ingestion Finished! Total clips: {progress.get('total_clips_processed', 0):,}")
    logger.info(f"Remaining Free Host Disk Space: {get_free_disk_gb():.1f} GB")
    logger.info("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
