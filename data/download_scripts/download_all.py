"""
AnimalVox AI — Data Download Scripts
Download datasets from xeno-canto, Kaggle, AudioSet, and NOAA.
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def download_xeno_canto(species_list: list = None, max_per_species: int = 500, quality: str = "A"):
    """
    Download bird recordings from xeno-canto.org (free API, no key needed).
    
    Args:
        species_list: List of species names (e.g., ["Turdus migratorius"])
        max_per_species: Maximum recordings per species
        quality: Minimum quality rating (A, B, C, D, E)
    """
    if species_list is None:
        species_list = [
            "Turdus migratorius",      # American Robin
            "Corvus brachyrhynchos",    # American Crow
            "Parus major",             # Great Tit
            "Mimus polyglottos",       # Northern Mockingbird
            "Melospiza melodia",       # Song Sparrow
            "Cyanocitta cristata",     # Blue Jay
            "Cardinalis cardinalis",   # Northern Cardinal
            "Strix varia",            # Barred Owl
            "Haliaeetus leucocephalus", # Bald Eagle
            "Corvus corax",           # Common Raven
        ]

    output_dir = RAW_DATA_DIR / "birds" / "xeno_canto"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://xeno-canto.org/api/2/recordings"

    for species in species_list:
        logger.info(f"Downloading {species}...")
        species_dir = output_dir / species.replace(" ", "_")
        species_dir.mkdir(exist_ok=True)

        params = {"query": f'{species} q>:{quality}', "page": 1}
        downloaded = 0

        while downloaded < max_per_species:
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                data = resp.json()
            except Exception as e:
                logger.error(f"API error for {species}: {e}")
                break

            recordings = data.get("recordings", [])
            if not recordings:
                break

            for rec in recordings:
                if downloaded >= max_per_species:
                    break
                try:
                    audio_url = rec.get("file", "")
                    if not audio_url:
                        continue
                    if not audio_url.startswith("http"):
                        audio_url = "https:" + audio_url

                    filename = f"{rec['id']}_{rec.get('type', 'unknown').replace(' ', '_')}.mp3"
                    filepath = species_dir / filename

                    if filepath.exists():
                        downloaded += 1
                        continue

                    audio_resp = requests.get(audio_url, timeout=60)
                    with open(filepath, 'wb') as f:
                        f.write(audio_resp.content)

                    # Save metadata
                    meta_path = species_dir / f"{rec['id']}_meta.json"
                    with open(meta_path, 'w') as f:
                        json.dump(rec, f, indent=2)

                    downloaded += 1
                except Exception as e:
                    logger.warning(f"Failed to download {rec.get('id', '?')}: {e}")

            # Next page
            if int(data.get("page", 1)) >= int(data.get("numPages", 1)):
                break
            params["page"] += 1

        logger.info(f"  Downloaded {downloaded} recordings for {species}")


def download_kaggle_birdclef(competition: str = "birdclef-2024"):
    """Download BirdCLEF dataset from Kaggle (requires KAGGLE_KEY)."""
    output_dir = RAW_DATA_DIR / "birds" / "birdclef"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        os.system(f'kaggle competitions download -c {competition} -p "{output_dir}"')
        logger.info(f"BirdCLEF downloaded to {output_dir}")
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info("Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")


def download_dcase_dogs():
    """Download DCASE 2020 Task 5 dog behavior audio."""
    output_dir = RAW_DATA_DIR / "dogs" / "dcase"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"DCASE dog data directory: {output_dir}")
    logger.info("Download from: https://zenodo.org/record/3665023")
    logger.info("Manual download required — dataset contains ~10k clips")


def download_noaa_marine():
    """Download NOAA DCLDE marine mammal recordings."""
    output_dir = RAW_DATA_DIR / "marine" / "dclde"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"NOAA marine data directory: {output_dir}")
    logger.info("Download from: https://www.fisheries.noaa.gov/resource/document/dclde")
    logger.info("Also see: https://whoicf2.whoi.edu/science/B/whalesounds/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== AnimalVox AI Data Download ===")
    print("1. Downloading xeno-canto bird recordings...")
    download_xeno_canto(max_per_species=50)
    print("2. DCASE dog data (manual download needed)")
    download_dcase_dogs()
    print("3. NOAA marine data (manual download needed)")
    download_noaa_marine()
    print("Done!")
