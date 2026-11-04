"""
AnimalVox AI — Google Drive Shard Manager
==========================================
Orchestrates the flow of data between 5TB Google Drive and local disk:
1. Discovers dataset shards on the rclone-mounted Google Drive.
2. Downloads one shard at a time to local staging.
3. Monitors local disk space (enforces safety watermark).
4. After feature extraction, purges the raw audio shard.
5. Supports both rclone mount (virtual drive) and rclone copy (manual pull).

Usage:
    # Setup rclone first (one-time):
    #   rclone config   → create remote named "gdrive"
    #
    # Option A: Mount as virtual drive (recommended):
    #   rclone mount gdrive: X: --vfs-cache-mode full --vfs-cache-max-size 15G
    #
    # Option B: Use this script to copy shards on demand

    python training/gdrive_shard_manager.py --remote gdrive --list
    python training/gdrive_shard_manager.py --remote gdrive --pull shard_001_dogspeak
    python training/gdrive_shard_manager.py --remote gdrive --pull-all --extract
"""

import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GDriveShardManager")

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR


# ── Dataset Registry ────────────────────────────────────────────
# Maps each dataset to its source, expected size, and target module
DATASET_REGISTRY = {
    # ── DOGS ──
    "dogspeak": {
        "display_name": "DogSpeak (77k barks)",
        "source_type": "huggingface",
        "hf_repo": "ArlingtonCL2/DogSpeak_Dataset",
        "gdrive_shard": "shards/week1/dogspeak",
        "target_modules": ["breed_acoustic", "emotion_classifier", "health_monitor", "animal_detector"],
        "species": "dogs",
        "est_size_gb": 12.0,
        "priority": 1,
    },
    "barkopedia_vocal": {
        "display_name": "Barkopedia Dog Vocal Detection",
        "source_type": "huggingface",
        "hf_repo": "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection",
        "gdrive_shard": "shards/week1/barkopedia_vocal",
        "target_modules": ["animal_detector"],
        "species": "dogs",
        "est_size_gb": 2.0,
        "priority": 3,
    },
    "barkopedia_emotion": {
        "display_name": "Barkopedia Emotion Classification",
        "source_type": "huggingface",
        "hf_repo": "ArlingtonCL2/BarkopediaDogEmotionClassification_Data",
        "gdrive_shard": "shards/week1/barkopedia_emotion",
        "target_modules": ["emotion_classifier"],
        "species": "dogs",
        "est_size_gb": 3.0,
        "priority": 2,
    },
    "barkopedia_age": {
        "display_name": "Barkopedia Age Group (4,920 clips)",
        "source_type": "huggingface",
        "hf_repo": "hlx1021/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET",
        "gdrive_shard": "shards/week1/barkopedia_age",
        "target_modules": ["breed_acoustic", "animal_detector", "anti_spoofing"],
        "species": "dogs",
        "est_size_gb": 1.0,
        "priority": 4,
    },
    "audioset_dogs": {
        "display_name": "AudioSet Dog Subset (39k clips)",
        "source_type": "manual",
        "url": "https://research.google.com/audioset",
        "gdrive_shard": "shards/week3/audioset_dogs",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "dogs",
        "est_size_gb": 8.0,
        "priority": 8,
    },
    "audioset_animals": {
        "display_name": "AudioSet Strongly Annotated Animals (Dogs, Cats, Birds, Frogs)",
        "source_type": "huggingface",
        "hf_repo": "enyoukai/AudioSet-Strong-Animals",
        "gdrive_shard": "shards/week3/audioset_animals",
        "target_modules": ["animal_detector", "anti_spoofing", "breed_acoustic", "emotion_classifier"],
        "species": "multi",
        "est_size_gb": 2.5,
        "priority": 2,
    },
    # ── CATS ──
    "catmeows": {
        "display_name": "CatMeows (440 meows, 3 contexts)",
        "source_type": "huggingface",
        "hf_repo": "zeddez/CatMeows",
        "gdrive_shard": "shards/week1/catmeows",
        "target_modules": ["emotion_classifier", "health_monitor", "animal_detector"],
        "species": "cats",
        "est_size_gb": 0.1,
        "priority": 5,
    },
    "animal_sound_classification": {
        "display_name": "Animal Sound Classification (Multi-species)",
        "source_type": "huggingface",
        "hf_repo": "MuhammadQASIM111/Animal_Sound_Classification",
        "gdrive_shard": "shards/week1/animal_sounds",
        "target_modules": ["animal_detector"],
        "species": "multi",
        "est_size_gb": 0.5,
        "priority": 6,
    },
    # ── BIRDS ──
    "birdset": {
        "display_name": "BirdSet (Multi-task benchmark)",
        "source_type": "huggingface",
        "hf_repo": "DBD-research-group/BirdSet",
        "gdrive_shard": "shards/week2/birdset",
        "target_modules": ["animal_detector", "behavioral_classifier"],
        "species": "birds",
        "est_size_gb": 50.0,
        "priority": 3,
    },
    # ── DOLPHINS ──
    "openwhistle": {
        "display_name": "OpenWhistle (180k dolphin whistles)",
        "source_type": "manual",
        "url": "https://arxiv.org/pdf/2609.34839",
        "gdrive_shard": "shards/week2/openwhistle",
        "target_modules": ["behavioral_classifier", "emotion_classifier"],
        "species": "dolphins",
        "est_size_gb": 25.0,
        "priority": 6,
    },
    # ── WHALES ──
    "dclde_2022": {
        "display_name": "DCLDE 2022 (Expert cetacean annotations)",
        "source_type": "zenodo",
        "url": "https://doi.org/10.5281/zenodo.7738698",
        "gdrive_shard": "shards/week2/dclde_2022",
        "target_modules": ["behavioral_classifier", "animal_detector"],
        "species": "whales",
        "est_size_gb": 15.0,
        "priority": 7,
    },
    # ── FROGS ──
    "rfcx": {
        "display_name": "RFCX (Tropical birds + frogs)",
        "source_type": "kaggle",
        "kaggle_id": "rfcx-species-audio-detection",
        "gdrive_shard": "shards/week2/rfcx",
        "target_modules": ["animal_detector"],
        "species": "frogs",
        "est_size_gb": 5.0,
        "priority": 8,
    },
    # ── ANTI-SPOOFING ──
    "vocal_imitation_set": {
        "display_name": "Vocal Imitation Set (Human imitating animals)",
        "source_type": "manual",
        "url": "https://labsites.rochester.edu/air/projects/vroom/datacollection.html",
        "gdrive_shard": "shards/week3/vocal_imitation",
        "target_modules": ["anti_spoofing"],
        "species": "human_imitation",
        "est_size_gb": 2.0,
        "priority": 10,
    },
    # ── NOISE & AUGMENTATION ──
    "demand": {
        "display_name": "DEMAND (18 noise environments)",
        "source_type": "zenodo",
        "url": "https://zenodo.org/record/1227121",
        "gdrive_shard": "shards/week3/demand",
        "target_modules": ["augmentation"],
        "species": "noise",
        "est_size_gb": 4.0,
        "priority": 11,
    },
    "musan": {
        "display_name": "MUSAN (Music, Speech, Noise)",
        "source_type": "manual",
        "url": "https://openslr.org/17",
        "gdrive_shard": "shards/week3/musan",
        "target_modules": ["augmentation"],
        "species": "noise",
        "est_size_gb": 11.0,
        "priority": 11,
    },
    # ── MULTI-SPECIES ──
    "esc50": {
        "display_name": "ESC-50 (10 animal classes)",
        "source_type": "github",
        "url": "https://github.com/karoldvl/ESC-50",
        "gdrive_shard": "shards/week3/esc50",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "multi",
        "est_size_gb": 0.6,
        "priority": 9,
    },
    "wildlife": {
        "display_name": "Wildlife Bioacoustics (1,117 clips — whales, birds, otters, bats)",
        "source_type": "huggingface",
        "hf_repo": "cgeorgiaw/animal-sounds",
        "gdrive_shard": "shards/week3/wildlife",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "multi",
        "est_size_gb": 0.75,
        "priority": 7,
    },
    "xenocanto_birds": {
        "display_name": "Xeno-Canto Avian Bioacoustics (Songs, Calls, Alarms)",
        "source_type": "api",
        "url": "https://xeno-canto.org",
        "gdrive_shard": "shards/week4/xenocanto_birds",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "birds",
        "est_size_gb": 0.5,
        "priority": 6,
    },
    "xenocanto_frogs": {
        "display_name": "Xeno-Canto Neotropical Anurans (Frog Calls)",
        "source_type": "api",
        "url": "https://xeno-canto.org",
        "gdrive_shard": "shards/week4/xenocanto_frogs",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "frogs",
        "est_size_gb": 0.3,
        "priority": 6,
    },
    "barkopedia_detection": {
        "display_name": "Barkopedia Dog Vocal Detection (17,836 barks)",
        "source_type": "huggingface",
        "hf_repo": "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection",
        "gdrive_shard": "shards/week4/barkopedia_detection",
        "target_modules": ["animal_detector", "breed_acoustic"],
        "species": "dogs",
        "est_size_gb": 0.8,
        "priority": 5,
    },
    "watkins_marine": {
        "display_name": "Watkins Marine Mammal Database (Dolphins & Whales)",
        "source_type": "huggingface",
        "hf_repo": "DBD-research-group/beans_watkins",
        "gdrive_shard": "shards/week4/watkins_marine",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "multi",
        "est_size_gb": 0.4,
        "priority": 5,
    },
    "avian_bioacoustics": {
        "display_name": "Avian Bioacoustics (NIPS4Bplus & Zebra Finch)",
        "source_type": "huggingface",
        "hf_repo": "cgeorgiaw/animal-sounds",
        "gdrive_shard": "shards/week4/avian_bioacoustics",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "birds",
        "est_size_gb": 0.5,
        "priority": 4,
    },
    "rfcx_frogs": {
        "display_name": "Rainforest Connection (RFCX Frogs & Soundscapes)",
        "source_type": "huggingface",
        "hf_repo": "DBD-research-group/beans_rfcx",
        "gdrive_shard": "shards/week4/rfcx_frogs",
        "target_modules": ["animal_detector", "anti_spoofing"],
        "species": "frogs",
        "est_size_gb": 0.4,
        "priority": 5,
    },
    "beans_dogs": {
        "display_name": "BEANS Dogs Benchmark Vocalizations",
        "source_type": "huggingface",
        "hf_repo": "DBD-research-group/beans_dogs",
        "gdrive_shard": "shards/week4/beans_dogs",
        "target_modules": ["animal_detector", "breed_acoustic", "health_monitor"],
        "species": "dogs",
        "est_size_gb": 0.3,
        "priority": 5,
    },
}


def get_free_disk_gb(path: Path = DATA_DIR) -> float:
    """Returns free disk space in GB."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def check_rclone_installed() -> bool:
    """Check if rclone is available on PATH."""
    try:
        result = subprocess.run(
            ["rclone", "version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.strip().split("\n")[0]
            logger.info(f"rclone found: {version_line}")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def check_rclone_remote_exists(remote_name: str = "gdrive") -> bool:
    """Check if the named rclone remote is configured."""
    try:
        result = subprocess.run(
            ["rclone", "listremotes"], capture_output=True, text=True, timeout=10
        )
        remotes = [r.strip().rstrip(":") for r in result.stdout.strip().split("\n") if r.strip()]
        return remote_name in remotes
    except Exception:
        return False


def rclone_list_shards(remote_name: str = "gdrive", base_path: str = "AnimalVoxDatasets/shards") -> List[Dict]:
    """List available shards on the Google Drive remote."""
    try:
        result = subprocess.run(
            ["rclone", "lsjson", f"{remote_name}:{base_path}", "--dirs-only"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"Failed to list remote shards: {e}")
    return []


def rclone_pull_shard(
    remote_name: str,
    remote_path: str,
    local_path: Path,
    bandwidth_limit: str = "0",
    transfers: int = 4
) -> bool:
    """
    Download a shard from Google Drive to local staging using rclone copy.

    Args:
        remote_name: rclone remote name (e.g., "gdrive")
        remote_path: Path on the remote (e.g., "AnimalVoxDatasets/shards/week1/dogspeak")
        local_path: Local destination directory
        bandwidth_limit: Bandwidth limit (e.g., "10M" for 10 MB/s, "0" for unlimited)
        transfers: Number of parallel file transfers
    """
    local_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rclone", "copy",
        f"{remote_name}:{remote_path}",
        str(local_path),
        "--transfers", str(transfers),
        "--progress",
        "--stats", "5s",
    ]
    if bandwidth_limit != "0":
        cmd.extend(["--bwlimit", bandwidth_limit])

    logger.info(f"rclone copy {remote_name}:{remote_path} → {local_path}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"  [rclone] {line}")
        process.wait()
        return process.returncode == 0
    except Exception as e:
        logger.error(f"rclone copy failed: {e}")
        return False


def rclone_push_features(
    remote_name: str,
    local_features_dir: Path,
    remote_features_path: str = "AnimalVoxDatasets/features"
) -> bool:
    """Push extracted features back to Google Drive for backup."""
    cmd = [
        "rclone", "copy",
        str(local_features_dir),
        f"{remote_name}:{remote_features_path}",
        "--progress",
    ]
    logger.info(f"Backing up features to {remote_name}:{remote_features_path}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Feature backup failed: {e}")
        return False


@dataclass
class ShardCycleResult:
    """Result of a single shard download-extract-purge cycle."""
    shard_id: str
    success: bool
    num_files_downloaded: int
    num_features_extracted: int
    feature_file: Optional[Path]
    disk_freed_gb: float
    duration_seconds: float
    error: Optional[str] = None


class GDriveShardManager:
    """
    Full lifecycle manager for the Google Drive ↔ Local sharded pipeline.
    
    Workflow:
      1. Check disk safety (must have ≥ min_free_gb available)
      2. Pull one shard from gdrive → local staging
      3. Run feature extraction (raw audio → compact .npz embeddings)
      4. Delete raw audio shard from local disk
      5. Optionally backup .npz features to Google Drive
      6. Repeat for next shard
    """

    def __init__(
        self,
        remote_name: str = "gdrive",
        remote_base: str = "AnimalVoxDatasets",
        staging_dir: Optional[Path] = None,
        features_dir: Optional[Path] = None,
        min_free_gb: float = 15.0,
        max_shard_gb: float = 8.0,
        bandwidth_limit: str = "0",
    ):
        self.remote_name = remote_name
        self.remote_base = remote_base
        self.staging_dir = staging_dir or (DATA_DIR / "shard_staging")
        self.features_dir = features_dir or (PROCESSED_DATA_DIR / "shard_features")
        self.min_free_gb = min_free_gb
        self.max_shard_gb = max_shard_gb
        self.bandwidth_limit = bandwidth_limit
        self.completed_shards: List[ShardCycleResult] = []

        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Load processing history
        self.history_file = self.features_dir / "shard_history.json"
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {"completed": [], "failed": []}

    def _save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2, default=str)

    def check_disk_safe(self) -> Tuple[bool, float]:
        """Returns (is_safe, free_gb)."""
        free_gb = get_free_disk_gb(self.staging_dir)
        safe = free_gb >= self.min_free_gb
        if not safe:
            logger.error(
                f"⛔ DISK SAFETY VIOLATION: {free_gb:.1f} GB free < {self.min_free_gb:.1f} GB required"
            )
        return safe, free_gb

    def list_registry(self, species_filter: str = None, module_filter: str = None):
        """Print the dataset registry with status indicators."""
        print("\n" + "=" * 80)
        print("📦 AnimalVox AI — Dataset Registry & Shard Status")
        print("=" * 80)

        sorted_datasets = sorted(DATASET_REGISTRY.items(), key=lambda x: x[1]["priority"])

        for ds_id, ds in sorted_datasets:
            if species_filter and ds["species"] != species_filter:
                continue
            if module_filter and module_filter not in ds["target_modules"]:
                continue

            status = "✅" if ds_id in self.history.get("completed", []) else "⏳"
            feature_file = self.features_dir / f"features_{ds_id}.npz"
            has_features = "📊" if feature_file.exists() else "  "

            print(
                f"  {status} {has_features} [{ds['priority']:2d}] "
                f"{ds['display_name']:<45s} "
                f"~{ds['est_size_gb']:5.1f} GB  "
                f"[{ds['species']:<8s}]  "
                f"→ {', '.join(ds['target_modules'])}"
            )

        free_gb = get_free_disk_gb()
        print(f"\n  💾 Local disk free: {free_gb:.1f} GB | Safety buffer: {self.min_free_gb} GB")
        print("=" * 80 + "\n")

    def pull_and_extract(
        self,
        dataset_id: str,
        extract: bool = True,
        delete_raw: bool = True,
        backup_features: bool = False
    ) -> ShardCycleResult:
        """
        Execute one full shard cycle for a registered dataset.
        
        Steps:
        1. Check disk space safety.
        2. Download the dataset (HuggingFace streaming / rclone / Kaggle).
        3. Extract 266-dim embeddings into compact .npz file.
        4. Delete the raw audio shard.
        5. Optionally back up .npz to Google Drive.
        """
        start_time = time.time()

        if dataset_id not in DATASET_REGISTRY:
            return ShardCycleResult(
                shard_id=dataset_id, success=False,
                num_files_downloaded=0, num_features_extracted=0,
                feature_file=None, disk_freed_gb=0, duration_seconds=0,
                error=f"Unknown dataset_id '{dataset_id}'. Use --list to see available datasets."
            )

        ds = DATASET_REGISTRY[dataset_id]
        shard_local_dir = self.staging_dir / dataset_id

        print("\n" + "━" * 70)
        logger.info(f"🚀 Starting Shard Cycle: {ds['display_name']}")
        logger.info(f"   Species: {ds['species']} | Est. Size: {ds['est_size_gb']} GB")
        logger.info(f"   Target Modules: {', '.join(ds['target_modules'])}")
        print("━" * 70)

        # Step 1: Disk safety
        safe, free_gb = self.check_disk_safe()
        if not safe:
            return ShardCycleResult(
                shard_id=dataset_id, success=False,
                num_files_downloaded=0, num_features_extracted=0,
                feature_file=None, disk_freed_gb=0,
                duration_seconds=time.time() - start_time,
                error=f"Insufficient disk space: {free_gb:.1f} GB free"
            )

        # Step 2: Download / Fetch
        num_files = 0
        try:
            if ds["source_type"] == "huggingface":
                num_files = self._fetch_huggingface(ds["hf_repo"], shard_local_dir, dataset_id)
            elif ds["source_type"] in ("manual", "zenodo", "github"):
                # Try rclone first (if data was pre-uploaded to Google Drive)
                remote_path = f"{self.remote_base}/{ds['gdrive_shard']}"
                success = rclone_pull_shard(
                    self.remote_name, remote_path, shard_local_dir,
                    bandwidth_limit=self.bandwidth_limit
                )
                if success:
                    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
                    num_files = sum(1 for f in shard_local_dir.rglob("*")
                                   if f.is_file() and f.suffix.lower() in audio_exts)
                else:
                    logger.warning(
                        f"rclone pull failed for {dataset_id}. "
                        f"Please manually download from: {ds.get('url', 'N/A')} "
                        f"and place files in: {shard_local_dir}"
                    )
            elif ds["source_type"] == "kaggle":
                num_files = self._fetch_kaggle(ds.get("kaggle_id", ""), shard_local_dir)
        except Exception as e:
            logger.error(f"Download failed for {dataset_id}: {e}")
            return ShardCycleResult(
                shard_id=dataset_id, success=False,
                num_files_downloaded=0, num_features_extracted=0,
                feature_file=None, disk_freed_gb=0,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )

        logger.info(f"Downloaded {num_files} audio files for '{dataset_id}'")

        # Step 3: Feature extraction
        feature_file = None
        num_features = 0
        if extract and num_files > 0:
            try:
                from training.feature_extractor import extract_features_from_directory
                feature_file, num_features = extract_features_from_directory(
                    audio_dir=shard_local_dir,
                    output_path=self.features_dir / f"features_{dataset_id}.npz",
                    species=ds["species"],
                    dataset_id=dataset_id,
                )
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")

        # Step 4: Clean up raw audio
        disk_freed = 0.0
        if delete_raw and shard_local_dir.exists():
            before = get_free_disk_gb()
            shutil.rmtree(shard_local_dir, ignore_errors=True)
            after = get_free_disk_gb()
            disk_freed = after - before
            logger.info(f"🗑️  Purged raw shard. Reclaimed {disk_freed:.2f} GB")

        # Step 5: Backup features
        if backup_features and feature_file and feature_file.exists():
            rclone_push_features(
                self.remote_name, self.features_dir,
                f"{self.remote_base}/features"
            )

        # Record history
        duration = time.time() - start_time
        self.history["completed"].append(dataset_id)
        self._save_history()

        result = ShardCycleResult(
            shard_id=dataset_id,
            success=True,
            num_files_downloaded=num_files,
            num_features_extracted=num_features,
            feature_file=feature_file,
            disk_freed_gb=disk_freed,
            duration_seconds=duration,
        )
        self.completed_shards.append(result)
        logger.info(f"✅ Shard '{dataset_id}' complete in {duration:.1f}s")
        return result

    def _fetch_huggingface(self, repo_id: str, local_dir: Path, dataset_id: str) -> int:
        """Download a HuggingFace dataset, saving audio files to local_dir."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("HuggingFace datasets not installed. Run: pip install datasets")
            return 0

        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Streaming HuggingFace dataset: {repo_id}")

        try:
            # Use streaming to avoid downloading everything at once
            ds = load_dataset(repo_id, streaming=True, trust_remote_code=True)

            count = 0
            max_samples = 10000  # Limit per shard to protect disk

            for split_name in ds:
                split_dir = local_dir / split_name
                split_dir.mkdir(exist_ok=True)

                for sample in ds[split_name]:
                    if count >= max_samples:
                        break

                    # HuggingFace audio datasets typically have an 'audio' column
                    if "audio" in sample:
                        audio_data = sample["audio"]
                        if isinstance(audio_data, dict) and "array" in audio_data:
                            import soundfile as sf
                            import numpy as np
                            sr = audio_data.get("sampling_rate", 16000)
                            array = np.array(audio_data["array"], dtype=np.float32)

                            # Build filename from available metadata
                            label = sample.get("label", sample.get("species", "unknown"))
                            label_dir = split_dir / str(label)
                            label_dir.mkdir(exist_ok=True)

                            filepath = label_dir / f"{dataset_id}_{count:06d}.wav"
                            sf.write(str(filepath), array, sr)
                            count += 1

                    if count >= max_samples:
                        break

            logger.info(f"Saved {count} audio files from {repo_id}")
            return count

        except Exception as e:
            logger.error(f"HuggingFace download error for {repo_id}: {e}")
            return 0

    def _fetch_kaggle(self, competition_id: str, local_dir: Path) -> int:
        """Download a Kaggle competition dataset."""
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            cmd = ["kaggle", "competitions", "download", "-c", competition_id, "-p", str(local_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
                return sum(1 for f in local_dir.rglob("*")
                           if f.is_file() and f.suffix.lower() in audio_exts)
        except Exception as e:
            logger.error(f"Kaggle download failed: {e}")
        return 0

    def run_priority_pipeline(
        self,
        max_shards: int = 5,
        species_filter: str = None,
        module_filter: str = None,
    ):
        """
        Run the full shard pipeline in priority order.
        Downloads, extracts, and purges each dataset one at a time.
        """
        sorted_datasets = sorted(
            DATASET_REGISTRY.items(), key=lambda x: x[1]["priority"]
        )

        processed = 0
        for ds_id, ds in sorted_datasets:
            if processed >= max_shards:
                break
            if ds_id in self.history.get("completed", []):
                logger.info(f"Skipping {ds_id} (already completed)")
                continue
            if species_filter and ds["species"] != species_filter:
                continue
            if module_filter and module_filter not in ds["target_modules"]:
                continue

            result = self.pull_and_extract(ds_id)
            if not result.success:
                logger.error(f"Shard {ds_id} failed: {result.error}")
                self.history["failed"].append(ds_id)
                self._save_history()
            processed += 1

        print("\n" + "=" * 70)
        print("📊 Pipeline Summary")
        print("=" * 70)
        for r in self.completed_shards:
            status = "✅" if r.success else "❌"
            print(
                f"  {status} {r.shard_id:<30s} | "
                f"{r.num_features_extracted:>6d} features | "
                f"{r.disk_freed_gb:>5.1f} GB freed | "
                f"{r.duration_seconds:>6.1f}s"
            )


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Google Drive Shard Manager")
    parser.add_argument("--remote", type=str, default="gdrive", help="rclone remote name")
    parser.add_argument("--list", action="store_true", help="List all datasets in registry")
    parser.add_argument("--pull", type=str, help="Pull and extract a specific dataset by ID")
    parser.add_argument("--pull-all", action="store_true", help="Process all datasets in priority order")
    parser.add_argument("--species", type=str, help="Filter by species (dogs, cats, birds, etc.)")
    parser.add_argument("--module", type=str, help="Filter by target module")
    parser.add_argument("--max-shards", type=int, default=5, help="Max shards to process in --pull-all")
    parser.add_argument("--no-extract", action="store_true", help="Download only, skip feature extraction")
    parser.add_argument("--keep-raw", action="store_true", help="Don't delete raw audio after extraction")
    parser.add_argument("--backup", action="store_true", help="Backup features to Google Drive")
    parser.add_argument("--min-free-gb", type=float, default=15.0, help="Minimum free disk space (GB)")
    parser.add_argument("--bwlimit", type=str, default="0", help="rclone bandwidth limit (e.g., '10M')")
    parser.add_argument("--check-setup", action="store_true", help="Verify rclone installation and remote")
    args = parser.parse_args()

    manager = GDriveShardManager(
        remote_name=args.remote,
        min_free_gb=args.min_free_gb,
        bandwidth_limit=args.bwlimit,
    )

    if args.check_setup:
        print("\n🔍 Checking rclone setup...")
        if check_rclone_installed():
            print("  ✅ rclone is installed")
        else:
            print("  ❌ rclone not found. Install with: winget install Rclone.Rclone")
            return

        if check_rclone_remote_exists(args.remote):
            print(f"  ✅ Remote '{args.remote}' is configured")
        else:
            print(f"  ❌ Remote '{args.remote}' not found. Run: rclone config")
            return

        safe, free_gb = manager.check_disk_safe()
        print(f"  {'✅' if safe else '❌'} Disk space: {free_gb:.1f} GB free")
        return

    if args.list:
        manager.list_registry(species_filter=args.species, module_filter=args.module)
        return

    if args.pull:
        manager.pull_and_extract(
            args.pull,
            extract=not args.no_extract,
            delete_raw=not args.keep_raw,
            backup_features=args.backup,
        )
        return

    if args.pull_all:
        manager.run_priority_pipeline(
            max_shards=args.max_shards,
            species_filter=args.species,
            module_filter=args.module,
        )
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
