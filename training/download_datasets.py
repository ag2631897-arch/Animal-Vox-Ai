"""
AnimalVox AI — Direct Dataset Downloader
==========================================
Downloads datasets directly using huggingface_hub, Zenodo, and direct URLs.
Handles zip archives, audiofolder formats, and streaming datasets properly.

This bypasses the datasets library streaming issues with non-standard formats
and directly downloads + extracts audio files for the feature extraction pipeline.

Usage:
    py training/download_datasets.py --dataset catmeows
    py training/download_datasets.py --dataset dogspeak --max-samples 5000
    py training/download_datasets.py --dataset all --max-samples 2000
"""

import os
import sys
import json
import shutil
import zipfile
import logging
import argparse
import requests
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetDownloader")

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

STAGING_DIR = DATA_DIR / "shard_staging"
FEATURES_DIR = PROCESSED_DATA_DIR / "shard_features"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def get_free_disk_gb(path: Path = DATA_DIR) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress bar."""
    try:
        from tqdm import tqdm
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# Dataset-Specific Downloaders
# ═══════════════════════════════════════════════════════════════

def download_catmeows(staging_dir: Path, max_samples: int = 10000) -> Tuple[Path, int]:
    """
    Download CatMeows dataset from Zenodo (original source).
    440 meows from 21 cats, 3 contexts (brushing, food, isolation).
    """
    dataset_dir = staging_dir / "catmeows"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    zenodo_url = "https://zenodo.org/records/4008297/files/CatMeows.zip?download=1"
    zip_path = dataset_dir / "CatMeows.zip"
    
    logger.info("Downloading CatMeows from Zenodo...")
    if not download_file(zenodo_url, zip_path, "CatMeows.zip"):
        # Fallback: try HuggingFace hub direct download
        logger.info("Trying HuggingFace Hub fallback...")
        try:
            from huggingface_hub import hf_hub_download
            zip_path_hf = hf_hub_download(
                repo_id="zeddez/CatMeows",
                filename="dataset.zip",
                repo_type="dataset",
                local_dir=str(dataset_dir),
            )
            zip_path = Path(zip_path_hf)
        except Exception as e:
            logger.error(f"HuggingFace fallback also failed: {e}")
            return dataset_dir, 0
    
    # Extract zip
    logger.info("Extracting CatMeows archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dataset_dir)
        zip_path.unlink(missing_ok=True)  # Delete zip to save space
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
    
    # Count audio files
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
    audio_files = list(dataset_dir.rglob("*"))
    audio_files = [f for f in audio_files if f.is_file() and f.suffix.lower() in audio_exts]
    
    # Organize by context (parse filename convention: C_NNNNN_BB_SS_OOOOO_RXX)
    for af in audio_files:
        name = af.stem
        parts = name.split("_")
        if len(parts) >= 1:
            context = parts[0]  # B=brushing, F=food, I=isolation
            context_map = {"B": "brushing", "F": "waiting_food", "I": "isolation"}
            context_name = context_map.get(context, "unknown")
            
            ctx_dir = dataset_dir / context_name
            ctx_dir.mkdir(exist_ok=True)
            dest = ctx_dir / af.name
            if not dest.exists():
                af.rename(dest)
    
    final_count = sum(1 for f in dataset_dir.rglob("*")
                      if f.is_file() and f.suffix.lower() in audio_exts)
    logger.info(f"✅ CatMeows: {final_count} audio files organized by context")
    return dataset_dir, final_count


def download_dogspeak(staging_dir: Path, max_samples: int = 5000) -> Tuple[Path, int]:
    """
    Download DogSpeak dataset from HuggingFace.
    77,202 bark sequences from 156 dogs across 5 breeds:
    chihuahua, german_shepherd, husky, pitbull, shiba_inu.
    
    Uses metadata.csv directly to map each file to dog_id and breed,
    avoiding slow repo tree listing and datasets streaming issues.
    Downloads files in parallel with ThreadPoolExecutor.
    """
    dataset_dir = staging_dir / "dogspeak"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading DogSpeak dataset (target {max_samples} balanced samples)...")
    
    try:
        from huggingface_hub import hf_hub_download
        import csv
        import shutil
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        repo_id = "ArlingtonCL2/DogSpeak_Dataset"
        
        # Step 1: Download metadata.csv
        logger.info("Fetching DogSpeak metadata.csv...")
        meta_path = hf_hub_download(
            repo_id=repo_id,
            filename="metadata.csv",
            repo_type="dataset",
        )
        
        # Step 2: Parse rows and group by breed for balanced sampling
        breed_rows = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "").strip()
                breed = row.get("breed", "unknown").strip().lower().replace(" ", "_")
                dog_id = row.get("dog_id", "").strip()
                if fname and dog_id:
                    breed_rows.setdefault(breed, []).append((fname, dog_id, breed))
        
        logger.info(f"Parsed DogSpeak breeds: { {b: len(rows) for b, rows in breed_rows.items()} }")
        
        # Calculate per-breed target
        num_breeds = len(breed_rows)
        samples_per_breed = max(1, max_samples // num_breeds)
        
        selected_samples = []
        import random
        random.seed(42)
        for breed, rows in breed_rows.items():
            random.shuffle(rows)
            selected_samples.extend(rows[:samples_per_breed])
        
        logger.info(f"Selected {len(selected_samples)} total samples across {num_breeds} breeds (~{samples_per_breed}/breed)")
        
        # Step 3: Download audio files in parallel
        download_tasks = []
        for fname, dog_id, breed in selected_samples:
            remote_path = f"dogspeak_released/{dog_id}/{fname}"
            dest_dir = dataset_dir / breed
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / fname
            download_tasks.append((remote_path, dest_file))
        
        logger.info(f"Downloading {len(download_tasks)} WAV files with 8 parallel workers...")
        
        def _download_single(task):
            remote_p, dest_f = task
            if dest_f.exists() and dest_f.stat().st_size > 0:
                return True
            try:
                cached_f = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_p,
                    repo_type="dataset",
                )
                shutil.copy2(cached_f, dest_f)
                return True
            except Exception:
                return False

        downloaded = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_single, task) for task in download_tasks]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                else:
                    failed += 1
                if (i + 1) % 250 == 0 or (i + 1) == len(download_tasks):
                    logger.info(f"  [{i+1}/{len(download_tasks)}] Downloaded: {downloaded}, Failed: {failed}")
        
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ DogSpeak downloaded: {final_count} bark samples across {num_breeds} breeds")
        return dataset_dir, final_count
        
    except Exception as e:
        logger.error(f"DogSpeak download error: {e}")
        import traceback
        traceback.print_exc()
        return dataset_dir, 0


def download_barkopedia_emotion(staging_dir: Path, max_samples: int = 2000) -> Tuple[Path, int]:
    """
    Download Barkopedia Dog Emotion Classification from HuggingFace.
    Downloads husky & shiba emotion labels and audio files in parallel.
    Maps arousal (High, Medium, Low) and valence (Positive, Negative, Neutral)
    to emotional state categories.
    """
    dataset_dir = staging_dir / "barkopedia_emotion"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading Barkopedia Dog Emotion dataset (target max {max_samples})...")
    
    try:
        from huggingface_hub import hf_hub_download
        import csv
        import shutil
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        repo_id = "ArlingtonCL2/BarkopediaDogEmotionClassification_Data"
        
        # Emotion mapping from Circumplex model (arousal, valence)
        def map_emotion(arousal: str, valence: str) -> str:
            a = arousal.strip().lower()
            v = valence.strip().lower()
            if a == "high" and v == "positive":
                return "playfulness"
            elif a in ("medium", "low") and v == "positive":
                return "contentment"
            elif a == "high" and v == "negative":
                return "stress_acute"
            elif a in ("medium", "low") and v == "negative":
                return "stress_chronic"
            elif a == "high" and v == "neutral":
                return "alertness"
            elif a == "medium" and v == "neutral":
                return "curiosity"
            else:
                return "neutral_rest"

        label_files = [
            ("husky_train_labels.csv", "train/husky"),
            ("shiba_train_labels.csv", "train/shiba"),
            ("husky_validation_labels.csv", "validation/husky"),
            ("shiba_validation_labels.csv", "validation/shiba"),
        ]

        tasks = []
        for csv_name, folder_prefix in label_files:
            try:
                csv_path = hf_hub_download(repo_id=repo_id, filename=csv_name, repo_type="dataset")
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        audio_id = row.get("audio_id", "").strip()
                        arousal = row.get("arousal", "medium")
                        valence = row.get("valence", "neutral")
                        if audio_id:
                            emotion_label = map_emotion(arousal, valence)
                            remote_path = f"{folder_prefix}/{audio_id}.wav"
                            dest_dir = dataset_dir / emotion_label
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest_path = dest_dir / f"{audio_id}.wav"
                            tasks.append((remote_path, dest_path))
            except Exception as e:
                logger.warning(f"Could not load {csv_name}: {e}")

        if max_samples and len(tasks) > max_samples:
            import random
            random.seed(42)
            random.shuffle(tasks)
            tasks = tasks[:max_samples]

        logger.info(f"Downloading {len(tasks)} Barkopedia audio files with 8 parallel workers...")

        def _download_task(task):
            remote_p, dest_p = task
            if dest_p.exists() and dest_p.stat().st_size > 0:
                return True
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote_p, repo_type="dataset")
                shutil.copy2(cached, dest_p)
                return True
            except Exception:
                return False

        downloaded = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_task, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                if (i + 1) % 200 == 0 or (i + 1) == len(tasks):
                    logger.info(f"  [{i+1}/{len(tasks)}] Downloaded: {downloaded}")

        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Barkopedia Emotion downloaded: {final_count} audio files")
        return dataset_dir, final_count

    except Exception as e:
        logger.error(f"Barkopedia download failed: {e}")
        return dataset_dir, 0


def download_wildlife_bioacoustics(staging_dir: Path, max_samples: int = 1500) -> Tuple[Path, int]:
    """
    Download wildlife bioacoustics from cgeorgiaw/animal-sounds on HuggingFace:
      - orca (marine mammal / whale vocalizations)
      - birds (NIPS4Bplus bioacoustic benchmark)
      - zebra_finch (avian vocalizations)
      - giant_otter (wild aquatic bioacoustics)
      - egyptian_fruit_bat (bat echolocation)
    """
    dataset_dir = staging_dir / "wildlife_bioacoustics"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Wildlife Bioacoustics from cgeorgiaw/animal-sounds...")

    try:
        from huggingface_hub import HfApi, hf_hub_download
        import shutil
        from concurrent.futures import ThreadPoolExecutor, as_completed

        repo_id = "cgeorgiaw/animal-sounds"
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset")

        # Select species targets
        species_targets = {
            "whales": [f for f in all_files if f.startswith("orca/") and f.endswith(".wav")],
            "birds": (
                [f for f in all_files if f.startswith("birds/") and f.endswith(".wav")] +
                [f for f in all_files if f.startswith("zebra_finch/") and f.endswith(".wav")][:500]
            ),
            "otters": [f for f in all_files if f.startswith("giant_otter/") and f.endswith(".wav")][:500],
            "bats": [f for f in all_files if f.startswith("egyptian_fruit_bat/") and f.endswith(".wav")][:500],
            "primates": [f for f in all_files if f.startswith("macaque/") and f.endswith(".wav")][:500],
        }

        download_tasks = []
        for sp_label, files in species_targets.items():
            sp_dir = dataset_dir / sp_label
            sp_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                dest = sp_dir / Path(f).name
                download_tasks.append((f, dest))

        logger.info(f"Downloading {len(download_tasks)} wildlife files across {list(species_targets.keys())}...")

        def _download_task(task):
            remote_p, dest_p = task
            if dest_p.exists() and dest_p.stat().st_size > 0:
                return True
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote_p, repo_type="dataset")
                shutil.copy2(cached, dest_p)
                return True
            except Exception:
                return False

        downloaded = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_task, t) for t in download_tasks]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                if (i + 1) % 200 == 0 or (i + 1) == len(download_tasks):
                    logger.info(f"  [{i+1}/{len(download_tasks)}] Downloaded: {downloaded}")

        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Wildlife Bioacoustics downloaded: {final_count} audio files")
        return dataset_dir, final_count

    except Exception as e:
        logger.error(f"Wildlife Bioacoustics download failed: {e}")
        return dataset_dir, 0


def download_esc50(staging_dir: Path, max_samples: int = 10000) -> Tuple[Path, int]:
    """
    Download ESC-50 environmental sound dataset (GitHub release).
    Extracts all 2,000 clips and organizes them into structured species & anti-spoof folders
    using meta/esc50.csv:
      - frogs: frog (40 clips)
      - birds: crow, rooster, hen, chirping_birds (160 clips)
      - dogs: dog (40 clips)
      - cats: cat (40 clips)
      - farm_animals: cow, pig, sheep (120 clips)
      - insects: insects, crickets (80 clips)
      - human_anti_spoof: crying_baby, sneezing, coughing, footsteps, laughing, breathing, snoring (280 clips)
      - soundscape: rain, sea_waves, crackling_fire, thunderstorm, wind (200 clips)
    """
    dataset_dir = staging_dir / "esc50"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    temp_extract = staging_dir / "esc50_temp"
    temp_extract.mkdir(parents=True, exist_ok=True)
    
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    zip_path = staging_dir / "esc50.zip"
    
    logger.info("Downloading ESC-50 archive from GitHub...")
    if not download_file(url, zip_path, "ESC-50"):
        return dataset_dir, 0
    
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(temp_extract)
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"ESC-50 extraction error: {e}")
        return dataset_dir, 0

    # Locate meta/esc50.csv and audio/ folder
    csv_candidates = list(temp_extract.rglob("esc50.csv"))
    if not csv_candidates:
        logger.error("esc50.csv not found in extracted archive")
        return dataset_dir, 0
    
    esc_csv = csv_candidates[0]
    audio_dir = esc_csv.parent.parent / "audio"

    # Category to species / anti-spoof mapping
    category_map = {
        "frog": "frogs",
        "crow": "birds",
        "rooster": "birds",
        "hen": "birds",
        "chirping_birds": "birds",
        "dog": "dogs",
        "cat": "cats",
        "cow": "farm_animals",
        "pig": "farm_animals",
        "sheep": "farm_animals",
        "insects": "insects",
        "crickets": "insects",
        "crying_baby": "human_anti_spoof",
        "sneezing": "human_anti_spoof",
        "coughing": "human_anti_spoof",
        "footsteps": "human_anti_spoof",
        "laughing": "human_anti_spoof",
        "breathing": "human_anti_spoof",
        "snoring": "human_anti_spoof",
        "rain": "soundscape_ambient",
        "sea_waves": "soundscape_ambient",
        "crackling_fire": "soundscape_ambient",
        "thunderstorm": "soundscape_ambient",
        "wind": "soundscape_ambient",
    }

    import csv
    import shutil
    organized = 0
    with open(esc_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "")
            cat = row.get("category", "")
            mapped_folder = category_map.get(cat)
            if mapped_folder and filename:
                src_file = audio_dir / filename
                if src_file.exists():
                    dest_dir = dataset_dir / mapped_folder
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dest_dir / filename)
                    organized += 1

    # Cleanup temp extract folder to save space
    shutil.rmtree(temp_extract, ignore_errors=True)

    logger.info(f"✅ ESC-50: {organized} files organized into structured species & anti-spoof folders")
    return dataset_dir, organized


def download_xenocanto_birds(staging_dir: Path, max_samples: int = 400) -> Tuple[Path, int]:
    """
    Download high-quality bird vocalizations from Xeno-Canto API v3.
    Uses XENO_CANTO_API_KEY from .env.
    Filters for Grade-A quality songs, calls, and alarms across diverse avian families.
    """
    dataset_dir = staging_dir / "xenocanto_birds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("XENO_CANTO_API_KEY", "")
    if not key:
        logger.error("XENO_CANTO_API_KEY not found in environment")
        return dataset_dir, 0
        
    logger.info(f"Querying Xeno-canto API v3 for high-quality bird vocalizations (target: {max_samples})...")
    
    queries = [
        "grp:birds+q:A+cnt:Brazil",
        "gen:Turdus+q:A",
        "grp:birds+q:A+cnt:Colombia",
        "grp:birds+q:A+cnt:Costa+Rica",
    ]
    
    recs = []
    seen_ids = set()
    for q in queries:
        if len(recs) >= max_samples:
            break
        try:
            url = f"https://xeno-canto.org/api/3/recordings?key={key}&query={q}"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("recordings", []):
                    rec_id = item.get("id")
                    if rec_id and rec_id not in seen_ids:
                        seen_ids.add(rec_id)
                        recs.append(item)
                        if len(recs) >= max_samples:
                            break
        except Exception as e:
            logger.warning(f"Xeno-canto query {q} failed: {e}")
            
    logger.info(f"Found {len(recs)} unique Xeno-canto bird recordings. Downloading...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def _download_rec(item):
        rec_id = item.get("id")
        f_url = item.get("file")
        call_type = str(item.get("type", "call")).lower()
        if "song" in call_type:
            sub = "song"
        elif "alarm" in call_type:
            sub = "alarm"
        else:
            sub = "call"
            
        target_dir = dataset_dir / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / f"xc_{rec_id}.mp3"
        
        if dest_file.exists() and dest_file.stat().st_size > 0:
            return True
        try:
            r = requests.get(f_url, stream=True, timeout=20)
            if r.status_code == 200:
                with open(dest_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            pass
        return False
        
    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_download_rec, item) for item in recs]
        for i, fut in enumerate(as_completed(futures)):
            if fut.result():
                downloaded += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(recs):
                logger.info(f"  [{i+1}/{len(recs)}] Downloaded: {downloaded} bird clips")
                
    final_count = sum(1 for f in dataset_dir.rglob("*.mp3") if f.is_file())
    logger.info(f"✅ Xeno-canto birds downloaded: {final_count} audio files")
    return dataset_dir, final_count


def download_xenocanto_frogs(staging_dir: Path, max_samples: int = 250) -> Tuple[Path, int]:
    """
    Download high-quality anuran (frog) recordings from Xeno-Canto API v3.
    """
    dataset_dir = staging_dir / "xenocanto_frogs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("XENO_CANTO_API_KEY", "")
    if not key:
        return dataset_dir, 0
        
    logger.info(f"Querying Xeno-canto for frog vocalizations (target: {max_samples})...")
    url = f"https://xeno-canto.org/api/3/recordings?key={key}&query=grp:frogs+q:A"
    recs = []
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            recs = resp.json().get("recordings", [])[:max_samples]
    except Exception as e:
        logger.error(f"Xeno-canto frogs query failed: {e}")
        return dataset_dir, 0
        
    logger.info(f"Downloading {len(recs)} frog recordings from Xeno-canto...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def _download_frog(item):
        rec_id = item.get("id")
        f_url = item.get("file")
        target_dir = dataset_dir / "call"
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_file = target_dir / f"frog_{rec_id}.mp3"
        if dest_file.exists() and dest_file.stat().st_size > 0:
            return True
        try:
            r = requests.get(f_url, stream=True, timeout=20)
            if r.status_code == 200:
                with open(dest_file, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            pass
        return False
        
    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_download_frog, item) for item in recs]
        for i, fut in enumerate(as_completed(futures)):
            if fut.result():
                downloaded += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(recs):
                logger.info(f"  [{i+1}/{len(recs)}] Downloaded: {downloaded} frog clips")
                
    final_count = sum(1 for f in dataset_dir.rglob("*.mp3") if f.is_file())
    logger.info(f"✅ Xeno-canto frogs downloaded: {final_count} audio files")
    return dataset_dir, final_count


def download_barkopedia_detection(staging_dir: Path, max_samples: int = 600) -> Tuple[Path, int]:
    """
    Download Barkopedia Dog Vocal Detection from HuggingFace.
    Contains 17,836 vocalizations across dog breeds (shiba inu, etc.).
    """
    dataset_dir = staging_dir / "barkopedia_detection"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "ArlingtonCL2/Barkopedia-Dog-Vocal-Detection"
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
        wav_files = [f for f in all_files if f.startswith("strong_audios/") and f.endswith(".wav")]
        
        import random
        random.seed(42)
        random.shuffle(wav_files)
        selected = wav_files[:max_samples]
        
        logger.info(f"Downloading {len(selected)} dog bark samples from Barkopedia Dog Vocal Detection...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil
        
        def _download_task(remote_p):
            fname = Path(remote_p).name
            parts = fname.rsplit("_", 2)
            breed = parts[1] if len(parts) > 1 else "canine"
            dest_dir = dataset_dir / breed
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_p = dest_dir / fname
            if dest_p.exists() and dest_p.stat().st_size > 0:
                return True
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote_p, repo_type="dataset", token=token)
                shutil.copy2(cached, dest_p)
                return True
            except Exception:
                return False
                
        downloaded = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_task, t) for t in selected]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(selected):
                    logger.info(f"  [{i+1}/{len(selected)}] Downloaded: {downloaded} bark samples")
                    
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Barkopedia Dog Detection downloaded: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"Barkopedia detection download failed: {e}")
        return dataset_dir, 0


def download_watkins_marine(staging_dir: Path, max_samples: int = 400) -> Tuple[Path, int]:
    """
    Download marine mammal vocalizations (Dolphins & Whales) from DBD-research-group/beans_watkins.
    Extracts embedded audio bytes directly from Parquet files without large downloads.
    """
    dataset_dir = staging_dir / "watkins_marine"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "DBD-research-group/beans_watkins"
        logger.info("Fetching Watkins Marine Mammal parquets from HuggingFace...")
        
        extracted = 0
        parquet_files = [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
            "data/test-00000-of-00001.parquet",
            "data/valid-00000-of-00001.parquet",
            "data/train_low-00000-of-00001.parquet",
        ]
        
        for pq_file in parquet_files:
            if extracted >= max_samples:
                break
            cached = hf_hub_download(repo_id=repo_id, filename=pq_file, repo_type="dataset", token=token)
            df = pd.read_parquet(cached)
            for _, row in df.iterrows():
                if extracted >= max_samples:
                    break
                label = str(row.get("label", "marine_mammal")).lower()
                path_val = row.get("path")
                raw_bytes = None
                fname = f"watkins_{extracted}.wav"
                
                if isinstance(path_val, dict) and "bytes" in path_val:
                    raw_bytes = path_val["bytes"]
                    if "path" in path_val and path_val["path"]:
                        fname = Path(str(path_val["path"])).name
                elif isinstance(path_val, bytes):
                    raw_bytes = path_val
                    
                # Group as dolphin or whale
                if any(k in label for k in ["dolphin", "grampus", "stenella", "delphinus", "tursiops", "globicephala"]):
                    sp_dir = dataset_dir / "dolphins"
                else:
                    sp_dir = dataset_dir / "whales"
                sp_dir.mkdir(parents=True, exist_ok=True)
                dest = sp_dir / fname
                
                if raw_bytes and len(raw_bytes) > 100:
                    with open(dest, "wb") as f:
                        f.write(raw_bytes)
                    extracted += 1
                    
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Watkins Marine Mammals extracted: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"Watkins marine download failed: {e}")
        return dataset_dir, 0


def download_avian_bioacoustics(staging_dir: Path, max_samples: int = 500) -> Tuple[Path, int]:
    """
    Download avian bioacoustics from cgeorgiaw/animal-sounds on HuggingFace:
      - NIPS4Bplus benchmark bird vocalizations (687 available)
      - Zebra finch calls (3,405 available)
    """
    dataset_dir = staging_dir / "avian_bioacoustics"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "cgeorgiaw/animal-sounds"
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
        
        nips_birds = [f for f in all_files if f.startswith("birds/") and f.endswith(".wav")]
        zebra_finch = [f for f in all_files if f.startswith("zebra_finch/") and f.endswith(".wav")]
        
        import random
        random.seed(42)
        random.shuffle(nips_birds)
        random.shuffle(zebra_finch)
        
        half = max_samples // 2
        selected = nips_birds[:half] + zebra_finch[:half]
        logger.info(f"Downloading {len(selected)} avian bioacoustic files (NIPS4Bplus + Zebra Finch)...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil
        
        def _download_task(remote_p):
            subfolder = "wild_birds" if remote_p.startswith("birds/") else "songbirds"
            dest_dir = dataset_dir / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_p = dest_dir / Path(remote_p).name
            if dest_p.exists() and dest_p.stat().st_size > 0:
                return True
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote_p, repo_type="dataset", token=token)
                shutil.copy2(cached, dest_p)
                return True
            except Exception:
                return False
                
        downloaded = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_task, t) for t in selected]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(selected):
                    logger.info(f"  [{i+1}/{len(selected)}] Downloaded: {downloaded} bird clips")
                    
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Avian Bioacoustics downloaded: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"Avian bioacoustics download failed: {e}")
        return dataset_dir, 0


def download_rfcx_frogs(staging_dir: Path, max_samples: int = 250) -> Tuple[Path, int]:
    """
    Download Rainforest Connection (RFCX) frog soundscape recordings from DBD-research-group/beans_rfcx.
    Extracts embedded audio bytes directly from Parquet files.
    """
    dataset_dir = staging_dir / "rfcx_frogs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "DBD-research-group/beans_rfcx"
        logger.info("Fetching RFCX Rainforest frog & tropical soundscape parquets from HuggingFace...")
        
        extracted = 0
        parquet_files = [f"data/train-{i:05d}-of-00033.parquet" for i in range(33)] + [
            "data/test-00000-of-00001.parquet",
            "data/valid-00000-of-00001.parquet",
        ]
        
        for pq_file in parquet_files:
            if extracted >= max_samples:
                break
            cached = hf_hub_download(repo_id=repo_id, filename=pq_file, repo_type="dataset", token=token)
            df = pd.read_parquet(cached)
            for _, row in df.iterrows():
                if extracted >= max_samples:
                    break
                path_val = row.get("path")
                raw_bytes = None
                fname = f"rfcx_{extracted}.wav"
                
                if isinstance(path_val, dict) and "bytes" in path_val:
                    raw_bytes = path_val["bytes"]
                elif isinstance(path_val, bytes):
                    raw_bytes = path_val
                    
                sp_dir = dataset_dir / "frogs"
                sp_dir.mkdir(parents=True, exist_ok=True)
                dest = sp_dir / fname
                
                if raw_bytes and len(raw_bytes) > 100:
                    with open(dest, "wb") as f:
                        f.write(raw_bytes)
                    extracted += 1
                    
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ RFCX Frogs extracted: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"RFCX frogs download failed: {e}")
        return dataset_dir, 0


def download_barkopedia_age(staging_dir: Path, max_samples: int = 500) -> Tuple[Path, int]:
    """
    Download Barkopedia Dog Age Group Classification dataset from HuggingFace.
    Contains 4,920 canine vocalization clips labeled by age group.
    """
    dataset_dir = staging_dir / "barkopedia_age"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "hlx1021/Barkopedia_DOG_AGE_GROUP_CLASSIFICATION_DATASET"
        api = HfApi()
        all_files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
        wav_files = [f for f in all_files if f.endswith(".wav")]
        
        import random
        random.seed(42)
        random.shuffle(wav_files)
        selected = wav_files[:max_samples]
        
        logger.info(f"Downloading {len(selected)} dog age group audio clips from Barkopedia...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import shutil
        
        def _download_task(remote_p):
            fname = Path(remote_p).name
            sub = Path(remote_p).parent.name or "canine"
            dest_dir = dataset_dir / sub
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_p = dest_dir / fname
            if dest_p.exists() and dest_p.stat().st_size > 0:
                return True
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=remote_p, repo_type="dataset", token=token)
                shutil.copy2(cached, dest_p)
                return True
            except Exception:
                return False
                
        downloaded = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_download_task, t) for t in selected]
            for i, fut in enumerate(as_completed(futures)):
                if fut.result():
                    downloaded += 1
                if (i + 1) % 100 == 0 or (i + 1) == len(selected):
                    logger.info(f"  [{i+1}/{len(selected)}] Downloaded: {downloaded} clips")
                    
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ Barkopedia Dog Age Group downloaded: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"Barkopedia age group download failed: {e}")
        return dataset_dir, 0


def download_beans_dogs(staging_dir: Path, max_samples: int = 1500) -> Tuple[Path, int]:
    """
    Download BEANS Dog Benchmark vocalizations from DBD-research-group/beans_dogs.
    Extracts embedded audio bytes directly from Parquet files without large downloads.
    """
    dataset_dir = staging_dir / "beans_dogs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "DBD-research-group/beans_dogs"
        logger.info("Fetching BEANS Dogs benchmark parquets from HuggingFace...")
        
        extracted = 0
        parquet_files = [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
            "data/valid-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
            "data/train_low-00000-of-00001.parquet",
        ]
        
        for pq_file in parquet_files:
            if extracted >= max_samples:
                break
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=pq_file, repo_type="dataset", token=token)
                df = pd.read_parquet(cached)
                for _, row in df.iterrows():
                    if extracted >= max_samples:
                        break
                    label = str(row.get("label", "canine")).lower()
                    path_val = row.get("path")
                    raw_bytes = None
                    fname = f"beansdog_{extracted}.wav"
                    
                    if isinstance(path_val, dict) and "bytes" in path_val:
                        raw_bytes = path_val["bytes"]
                        if "path" in path_val and path_val["path"]:
                            fname = Path(str(path_val["path"])).name
                    elif isinstance(path_val, bytes):
                        raw_bytes = path_val
                    
                    sp_dir = dataset_dir / label
                    sp_dir.mkdir(parents=True, exist_ok=True)
                    dest = sp_dir / fname
                    
                    if raw_bytes and len(raw_bytes) > 100:
                        with open(dest, "wb") as f:
                            f.write(raw_bytes)
                        extracted += 1
            except Exception as e:
                logger.warning(f"Error loading {pq_file}: {e}")
                
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ BEANS Dogs extracted: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"BEANS dogs download failed: {e}")
        return dataset_dir, 0


def download_audioset_animals(staging_dir: Path, max_samples: int = 3000) -> Tuple[Path, int]:
    """
    Download strongly annotated animal vocalizations from Google AudioSet
    via enyoukai/AudioSet-Strong-Animals.
    
    Contains ~17,800 clips covering:
      - Dogs: Bark, Howl, Growling, Yip, Dog
      - Cats: Cat, Meow, Purr, Hiss
      - Birds: Bird vocalization, Chirp, tweet, Squawk, Coo, Cluck
      - Frogs: Frog, Croak
      - Wildlife: Roaring cats, insects, crickets
    """
    dataset_dir = staging_dir / "audioset_animals"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        from dotenv import load_dotenv
        import soundfile as sf
        import io
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        
        repo_id = "enyoukai/AudioSet-Strong-Animals"
        logger.info(f"Streaming strongly annotated AudioSet Animal clips (target max {max_samples})...")
        
        parquet_files = [
            f"data/train-{i:05d}-of-00009.parquet" for i in range(9)
        ] + [
            "data/eval-00000-of-00002.parquet",
            "data/eval-00001-of-00002.parquet",
        ]
        
        extracted = 0
        for pq_file in parquet_files:
            if extracted >= max_samples:
                break
            try:
                cached = hf_hub_download(repo_id=repo_id, filename=pq_file, repo_type="dataset", token=token)
                df = pd.read_parquet(cached)
                for _, row in df.iterrows():
                    if extracted >= max_samples:
                        break
                    
                    # Extract primary animal category from events
                    events = row.get("events", [])
                    event_names = [e.get("event_name", "").lower() for e in events if isinstance(e, dict)]
                    all_text = " ".join(event_names)
                    
                    if any(k in all_text for k in ["bark", "dog", "growl", "howl", "yip"]):
                        sub = "dogs"
                    elif any(k in all_text for k in ["cat", "meow", "purr", "hiss"]):
                        sub = "cats"
                    elif any(k in all_text for k in ["bird", "chirp", "tweet", "squawk", "coo", "cluck", "chicken", "rooster"]):
                        sub = "birds"
                    elif any(k in all_text for k in ["frog", "croak"]):
                        sub = "frogs"
                    else:
                        sub = "wildlife"
                    
                    audio_entry = row.get("audio")
                    raw_bytes = None
                    vid = row.get("video_id", f"audioset_{extracted}")
                    fname = f"{vid}.wav"
                    
                    if isinstance(audio_entry, dict) and "bytes" in audio_entry:
                        raw_bytes = audio_entry["bytes"]
                    elif isinstance(audio_entry, bytes):
                        raw_bytes = audio_entry
                    
                    if raw_bytes and len(raw_bytes) > 100:
                        dest_dir = dataset_dir / sub
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_dir / fname
                        
                        try:
                            audio_data, sr = sf.read(io.BytesIO(raw_bytes))
                            sf.write(str(dest_path), audio_data, sr)
                            extracted += 1
                        except Exception:
                            with open(dest_path, "wb") as f:
                                f.write(raw_bytes)
                            extracted += 1
                
                logger.info(f"  Processed {pq_file}: {extracted} animal clips extracted so far")
            except Exception as e:
                logger.warning(f"Error loading {pq_file}: {e}")
                
        final_count = sum(1 for f in dataset_dir.rglob("*.wav") if f.is_file())
        logger.info(f"✅ AudioSet Animals extracted: {final_count} audio files")
        return dataset_dir, final_count
    except Exception as e:
        logger.error(f"AudioSet animals download failed: {e}")
        return dataset_dir, 0


# ═══════════════════════════════════════════════════════════════
# Feature Extraction Integration
# ═══════════════════════════════════════════════════════════════

def extract_and_cleanup(dataset_id: str, dataset_dir: Path, species: str, delete_raw: bool = True, upload_to_gdrive: bool = True):
    """Extract features from downloaded dataset, upload raw audio and features to Google Drive, then purge local raw audio."""
    from training.feature_extractor import extract_features_from_directory
    import subprocess
    
    feature_path = FEATURES_DIR / f"features_{dataset_id}.npz"
    
    # 1. Extract 266-dim dense features first (local features ready immediately)
    logger.info(f"Extracting 266-dim features from {dataset_dir}...")
    result_path, count = extract_features_from_directory(
        audio_dir=dataset_dir,
        output_path=feature_path,
        species=species,
        dataset_id=dataset_id,
    )
    
    if result_path and count > 0:
        size_mb = result_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Features saved: {result_path.name} ({count} samples, {size_mb:.1f} MB)")
        
        # 2. Upload feature shard to Google Drive
        if upload_to_gdrive:
            try:
                logger.info(f"☁️ Uploading feature shard {result_path.name} to gdrive:AnimalVoxDatasets/features/...")
                subprocess.run(
                    ["rclone", "copy", str(result_path), "gdrive:AnimalVoxDatasets/features/", "--transfers", "2"],
                    timeout=120
                )
            except Exception as e:
                logger.warning(f"Google Drive feature upload warning: {e}")

        # 3. Archive & sync raw audio to Google Drive 5TB repository
        if upload_to_gdrive:
            try:
                zip_file = STAGING_DIR / f"{dataset_id}_raw.zip"
                logger.info(f"📦 Packaging raw audio to {zip_file.name} for fast cloud upload...")
                shutil.make_archive(str(STAGING_DIR / f"{dataset_id}_raw"), "zip", dataset_dir)
                logger.info(f"☁️ Syncing {zip_file.name} to gdrive:AnimalVoxDatasets/raw/...")
                subprocess.run(
                    ["rclone", "copy", str(zip_file), "gdrive:AnimalVoxDatasets/raw/", "--transfers", "4"],
                    timeout=600
                )
                logger.info(f"☁️ Raw archive {zip_file.name} backed up on Google Drive.")
                if zip_file.exists():
                    zip_file.unlink()
            except Exception as e:
                logger.warning(f"Google Drive raw archive upload warning: {e}")

        # 4. Delete local raw audio to preserve local disk space
        if delete_raw:
            raw_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
            raw_size_mb = raw_size / (1024 * 1024)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            logger.info(f"🗑️  Deleted local raw audio ({raw_size_mb:.1f} MB) → Stored on Google Drive & kept local features ({size_mb:.1f} MB)")
    else:
        logger.warning(f"Feature extraction produced no results for {dataset_id}")


# ═══════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════

DOWNLOADERS = {
    "catmeows": ("cats", download_catmeows),
    "dogspeak": ("dogs", download_dogspeak),
    "esc50": ("multi", download_esc50),
    "wildlife": ("multi", download_wildlife_bioacoustics),
    "barkopedia_emotion": ("dogs", download_barkopedia_emotion),
    "barkopedia_detection": ("dogs", download_barkopedia_detection),
    "barkopedia_age": ("dogs", download_barkopedia_age),
    "watkins_marine": ("multi", download_watkins_marine),
    "avian_bioacoustics": ("birds", download_avian_bioacoustics),
    "rfcx_frogs": ("frogs", download_rfcx_frogs),
    "xenocanto_birds": ("birds", download_xenocanto_birds),
    "xenocanto_frogs": ("frogs", download_xenocanto_frogs),
    "beans_dogs": ("dogs", download_beans_dogs),
    "audioset_animals": ("multi", download_audioset_animals),
}


def main():
    parser = argparse.ArgumentParser(description="AnimalVox AI — Direct Dataset Downloader")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DOWNLOADERS.keys()) + ["all"],
        help="Which dataset to download"
    )
    parser.add_argument("--max-samples", type=int, default=5000, help="Max samples per dataset")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw audio after extraction")
    parser.add_argument("--no-extract", action="store_true", help="Download only, skip extraction")
    args = parser.parse_args()
    
    free_gb = get_free_disk_gb()
    logger.info(f"💾 Disk free: {free_gb:.1f} GB")
    
    if free_gb < 10:
        logger.error("Less than 10 GB free. Aborting to protect disk.")
        return
    
    datasets_to_process = list(DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]
    
    for ds_id in datasets_to_process:
        species, downloader = DOWNLOADERS[ds_id]
        
        print("\n" + "━" * 60)
        logger.info(f"📥 Downloading: {ds_id}")
        print("━" * 60)
        
        dataset_dir, count = downloader(STAGING_DIR, max_samples=args.max_samples)
        
        if count > 0 and not args.no_extract:
            extract_and_cleanup(
                dataset_id=ds_id,
                dataset_dir=dataset_dir,
                species=species,
                delete_raw=not args.keep_raw,
            )
        elif count == 0:
            logger.warning(f"No files downloaded for {ds_id}")
    
    # Show final state
    print("\n" + "=" * 60)
    print("📊 Feature Shards Available:")
    print("=" * 60)
    for npz in sorted(FEATURES_DIR.glob("features_*.npz")):
        import numpy as np
        data = np.load(npz, allow_pickle=True)
        n = len(data["embeddings"])
        size_mb = npz.stat().st_size / (1024 * 1024)
        print(f"  📊 {npz.name:<40s} {n:>6d} samples  {size_mb:>6.1f} MB")
    
    free_after = get_free_disk_gb()
    print(f"\n  💾 Disk free: {free_after:.1f} GB")
    print("=" * 60)
    print("\nNext step: Train models with:")
    print("  py training/train_from_shards.py --module all --epochs 30")


if __name__ == "__main__":
    main()
