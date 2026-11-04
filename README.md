<div align="center">

<!-- LOGO / BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A6B6B,100:27AE60&height=200&section=header&text=AnimalVox%20AI&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Animal%20Vocalization%20Intelligence%20Platform&descAlignY=58&descSize=18" width="100%"/>

<br/>

<p>
  <img src="https://img.shields.io/badge/Status-In%20Development-27AE60?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-0A6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Cost-100%25%20Free-1A6B3A?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
</p>

<h3>
  <em>Listen to the wild. Understand it.</em>
</h3>

<p align="center">
  AnimalVox AI is a behavioral bioacoustics platform that translates animal vocalizations — bird calls, dog barks, dolphin whistles, whale songs — into meaningful human language, grounded in real ethological science.
</p>

<br/>

[**📙 Implementation Plan**](./docs/AnimalVox_Implementation_Plan.docx) • [**📕 Project Explainer**](./docs/AnimalVox_Project_Explainer.docx)

<br/>

</div>

---

## 🌿 What Is AnimalVox AI?

AnimalVox AI is **not** a gimmick translator. Animals don't have words — but they absolutely have *meaning* in their sounds.

When a robin produces a rapid series of thin, high-pitched calls, ornithologists know from 80 years of field research that this is an **aerial predator alarm**. AnimalVox AI learns to recognize this pattern and outputs:

> *"Freeze — there's a hawk directly overhead. Don't move a muscle."*

Every translation is grounded in **peer-reviewed ethology** (the science of animal behavior). The system uses deep learning to classify vocalizations into behavioral states, then translates those states into natural human language.

---

## ✨ Key Features

- 🎙️ **Record or upload** any animal audio (`.wav`, `.mp3`, `.flac`, `.ogg`)
- 🧠 **5-layer AI pipeline** — acoustic encoding → behavioral classification → context reasoning → NLG
- 🐦 **Multi-species support** — Birds, Dogs, Cats, Dolphins, Whales, Frogs
- 📊 **Behavioral state detection** — 12 behavior classes with intensity + urgency scores
- 🔗 **Sequential context** — understands call *series*, not just single calls
- 🌐 **Web interface** — React/Vite frontend served by the FastAPI backend
- 💸 **100% Free** — zero infrastructure cost, fully open source

---

## 🏗️ Architecture

```
Raw Audio (.wav / .mp3 / .flac)
        │
        ▼
┌─────────────────────────────────────────┐
│         PREPROCESSING LAYER             │
│  Resample → Normalize → Segment → Gate  │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼                ▼
┌────────────┐  ┌─────────────┐  ┌──────────────┐
│   BEATs    │  │  Mel-Spec   │  │  Prosodic    │
│  Encoder   │  │  EfficientB3│  │  Extractor   │
│ (768-dim)  │  │ (CNN feats) │  │ (10 features)│
└─────┬──────┘  └──────┬──────┘  └──────┬───────┘
      └─────────────────┴────────────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │    FUSION + CLASSIFIER  │
           │  Multi-label: B00-B11   │
           │  Intensity + Urgency    │
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   CONTEXT REASONING    │
           │  Sequence Transformer  │
           │  Ethology KB (ChromaDB)│
           └────────────┬───────────┘
                        │
                        ▼
           ┌────────────────────────┐
           │   NLG — Mistral 7B     │
           │  Fine-tuned translator │
           └────────────┬───────────┘
                        │
                        ▼
         "Freeze — hawk above. Don't move."
```

---

## 🐾 Supported Animals

| Animal | Vocalizations Analyzed | Example Translation |
|--------|----------------------|---------------------|
| 🐦 **Birds** | Alarm calls, mating songs, territorial warnings, contact calls, distress | *"This is MY tree. I have been here three seasons and I am not leaving."* |
| 🐕 **Dogs** | Barks, growls, whines, howls, yelps | *"Something is at the door. I don't know what it is. EVERYONE WAKE UP."* |
| 🐱 **Cats** | Meows, purrs, hisses, chirps, chatters | *"I can see that bird and I cannot get to it and it is TORTURE."* |
| 🐬 **Dolphins** | Clicks, signature whistles, burst-pulses | *"It's me — I'm over here. Come find me. This is my call."* |
| 🐋 **Whales** | Songs, contact calls, echolocation | *"I am here. I am strong. Listen to how long I can sing."* |
| 🐸 **Frogs** | Advertisement calls, release calls, rain choruses | *"I am the loudest male in this pond and I am READY."* |

---

## 🧬 Behavior Classes

The system classifies 12 universal behavioral states across species:

| ID | Behavior | Description |
|----|----------|-------------|
| B00 | `alarm_aerial` | Aerial predator warning |
| B01 | `alarm_ground` | Ground predator warning |
| B02 | `territorial_warning` | Defend territory from rivals |
| B03 | `mating_call` | Attract mate / breeding signal |
| B04 | `contact_call` | Maintain group cohesion |
| B05 | `distress_call` | Pain, capture, extreme fear |
| B06 | `feeding_call` | Food found / foraging |
| B07 | `play_vocalization` | Social play / excitement |
| B08 | `submission_signal` | Deference / appeasement |
| B09 | `aggression_warning` | Direct threat / imminent attack |
| B10 | `social_bonding` | Affiliation maintenance |
| B11 | `navigation_echolocation` | Spatial orientation / hunting |

---

## 🛠️ Tech Stack

### Core ML
| Component | Technology | License |
|-----------|-----------|---------|
| Audio Encoder | [BEATs](https://github.com/microsoft/unilm/tree/master/beats) (Microsoft) | MIT |
| Spectrogram CNN | EfficientNet-B3 (torchvision) | Apache 2.0 |
| Prosodic Analysis | librosa + parselmouth | MIT / GPL |
| Sequence Model | Custom 4-layer Transformer | MIT |
| LLM (Translation) | Mistral 7B (QLoRA fine-tuned) | Apache 2.0 |
| Vector DB | ChromaDB | Apache 2.0 |

### Training & Infra
| Service | Use | Cost |
|---------|-----|------|
| Google Colab | GPU training (T4 / A100) | Free |
| Kaggle Notebooks | Parallel GPU training (P100) | Free |
| HuggingFace Hub | Model hosting + Spaces | Free |
| Groq API | Production LLM inference | Free tier |
| Weights & Biases | Experiment tracking | Free tier |
| GitHub Actions | CI/CD auto-deploy | Free |

---

## 📦 Datasets

| Dataset | Animals | Size | Source |
|---------|---------|------|--------|
| xeno-canto | Birds | 700k+ recordings | [xeno-canto.org](https://xeno-canto.org) |
| BirdCLEF 2024 | Birds | 180k clips | [Kaggle](https://kaggle.com/competitions/birdclef-2024) |
| Macaulay Library | Birds | 1M+ recordings | [Cornell Lab](https://www.macaulaylibrary.org) |
| DCASE 2020 Task 5 | Dogs | ~10k clips | [dcase.community](https://dcase.community) |
| AudioSet (filtered) | Dogs / Cats | ~15k clips | [Google Research](https://research.google.com/audioset/) |
| DCLDE 2022 | Dolphins / Whales | 50k+ recordings | [NOAA / Zenodo](https://doi.org/10.5281/zenodo.7738698) |
| Watkins Sound DB | Marine Mammals | 1,600+ sounds | [WHOI](https://cis.whoi.edu/science/B/whalesounds/) |
| Earth Species Project | Multi-species | Growing | [earthspecies.org](https://earthspecies.org) |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
CUDA-capable GPU (or use Google Colab / Kaggle)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/animalvox-ai.git
cd animalvox-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy the example env file
cp .env.example .env

# Fill in your API keys (see docs/AnimalVox_API_Keys_Guide.docx)
nano .env
```

Required keys:
```env
XENO_CANTO_API_KEY=xc_your_key_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
FREESOUND_API_KEY=your_freesound_client_id
HUGGINGFACE_TOKEN=hf_your_token_here
GROQ_API_KEY=gsk_your_key_here
WANDB_API_KEY=your_wandb_key_here
```

### Download Datasets

```bash
# Download bird dataset (xeno-canto + BirdCLEF)
python data/download_scripts/download_xeno_canto.py

# Download dog/cat clips
python data/download_scripts/download_audioclips.py

# Download marine mammal data
python data/download_scripts/download_marine.py
```

### Preprocess

```bash
# Standardize all audio + generate spectrograms
python data/preprocessing/standardize_audio.py
python data/preprocessing/generate_spectrograms.py
```

### Train

```bash
# Open training notebooks in Colab or Kaggle:
# training/train_birds.ipynb      ← Start here
# training/train_dogs.ipynb
# training/train_marine.ipynb
# training/finetune_mistral.ipynb
```

### Run the App Locally

AnimalVox uses the React frontend in `frontend/` as its primary interface. Build it once,
then FastAPI serves both the API and the frontend from one process.

```bash
# Terminal 1: build the React frontend
cd frontend
pnpm install
pnpm run build

# Terminal 2: start the API and primary web interface
cd ..
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000
# Open: http://localhost:8000
```

For frontend-only development, keep the API running on port 8000 and use `pnpm run dev`
inside `frontend/`; Vite proxies `/api` requests to the backend.

---

## 📁 Repository Structure

```
animalvox-ai/
├── data/
│   ├── download_scripts/          # Dataset downloaders
│   │   ├── download_xeno_canto.py
│   │   ├── download_birdclef.py
│   │   ├── download_audioclips.py
│   │   └── download_marine.py
│   ├── preprocessing/             # Audio standardization + augmentation
│   │   ├── standardize_audio.py
│   │   ├── generate_spectrograms.py
│   │   └── augmentation.py
│   └── label_maps/                # Behavior label mappings per species
│       ├── birds_behavior_map.json
│       ├── dogs_behavior_map.json
│       └── marine_behavior_map.json
│
├── models/
│   ├── acoustic_encoder/          # BEATs fine-tuning per species group
│   ├── behavioral_classifier/     # Multi-label classifier + prosodic extractor
│   ├── sequence_model/            # Call-series Transformer
│   └── nlg_layer/                 # Mistral 7B fine-tuned translator
│
├── knowledge_base/
│   ├── ethology_db.py             # ChromaDB interface
│   ├── populate_db.py             # Seed the vector database
│   └── seed_data/                 # JSON ethology entries (500+ species entries)
│
├── training/
│   ├── train_birds.ipynb          # Colab notebook — bird encoder
│   ├── train_dogs.ipynb           # Colab notebook — dog/cat encoder
│   ├── train_marine.ipynb         # Colab notebook — marine encoder
│   ├── finetune_mistral.ipynb     # QLoRA fine-tuning notebook
│   └── configs/                   # YAML hyperparameter configs
│
├── inference/
│   ├── pipeline.py                # Full end-to-end inference
│   └── api.py                     # FastAPI REST endpoint
│
├── app/
│   ├── frontend/                    # Primary React/Vite web interface
│   │   ├── src/                     # Screens, API client, components and state
│   │   └── package.json             # Frontend scripts and dependencies
│
├── evaluation/
│   ├── benchmark.py               # Model evaluation scripts
│   └── human_eval_tool.py         # Human evaluation interface
│
├── docs/                          # Project documentation
│   ├── AnimalVox_PRD.docx
│   ├── AnimalVox_TRD.docx
│   ├── AnimalVox_Implementation_Plan.docx
│   ├── AnimalVox_Project_Explainer.docx
│   └── AnimalVox_API_Keys_Guide.docx
│
├── .env.example                   # Environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🗺️ Roadmap

```
Phase 1  ████████████░░░░░░░░  Data Pipeline          2 weeks
Phase 2  ░░░░░░░░░░░░░░░░░░░░  Acoustic Encoder       2 weeks
Phase 3  ░░░░░░░░░░░░░░░░░░░░  Behavioral Classifier  1 week
Phase 4  ░░░░░░░░░░░░░░░░░░░░  Ethology KB            2 weeks
Phase 5  ░░░░░░░░░░░░░░░░░░░░  LLM Translator         2 weeks
Phase 6  ░░░░░░░░░░░░░░░░░░░░  Sequence Context       1 week
Phase 7  ░░░░░░░░░░░░░░░░░░░░  Web + Mobile App       1 week
Phase 8  ░░░░░░░░░░░░░░░░░░░░  Feedback Loop          Ongoing
```

| Phase | Status |
|-------|--------|
| Phase 1 — Data Pipeline | 🔄 In Progress |
| Phase 2 — Acoustic Encoder | ⏳ Planned |
| Phase 3 — Behavioral Classifier | ⏳ Planned |
| Phase 4 — Ethology Knowledge Base | ⏳ Planned |
| Phase 5 — LLM Translation Layer | ⏳ Planned |
| Phase 6 — Sequential Context Model | ⏳ Planned |
| Phase 7 — Web & Mobile App | ⏳ Planned |
| Phase 8 — Feedback & Retraining | ⏳ Planned |

---

## 🎯 Target Metrics

| Metric | Target |
|--------|--------|
| Bird call classification (top-1 accuracy) | > 85% |
| Dog behavior F1 (macro-averaged) | > 80% |
| Marine mammal call detection | > 78% |
| Web inference latency | < 3 seconds |
| Translation naturalness (human eval) | > 4.2 / 5.0 |

---

## 🔬 Related Research

This project builds on the work of:

- **[Earth Species Project](https://earthspecies.org)** — Applying AI to decode animal communication across all species
- **[Cornell Lab of Ornithology](https://www.birds.cornell.edu)** — Macaulay Library + BirdNET species identification
- **[Project CETI](https://www.projectceti.org)** — Decoding sperm whale communication using ML
- **[NOAA Fisheries](https://www.fisheries.noaa.gov)** — Marine mammal acoustic monitoring
- **[BirdCLEF](https://www.kaggle.com/competitions/birdclef-2024)** — Bird vocalization recognition challenge

---

## ⚠️ Honest Limitations

AnimalVox AI is scientifically grounded, but be aware:

- **Animals don't have language.** Translations are behavioral interpretations, not word-for-word translations.
- **Individual variation exists.** The model works at the population level — two individual dogs may sound different.
- **Recording quality matters.** Distant or noisy recordings reduce accuracy significantly.
- **Species coverage is limited.** Only 6 animal groups at launch — expanding via community contributions.
- **Not a veterinary tool.** Cannot diagnose health conditions from vocalizations.

---

## 🤝 Contributing

Contributions are welcome — especially for new species models and ethology knowledge base entries.

```bash
# Fork the repo, create a branch
git checkout -b feature/add-wolf-species

# Make your changes, commit
git commit -m "feat: add wolf vocalization model and ethology entries"

# Push and open a Pull Request
git push origin feature/add-wolf-species
```

Areas where help is most needed:
- 🐺 New species ethology entries (JSON format in `knowledge_base/seed_data/`)
- 🎵 Dataset curation and labeling
- 📱 Mobile app (Capacitor/React Native)
- 🌍 Translations of the app UI into other languages

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

All datasets used are publicly available and used in accordance with their respective licenses. Model weights (BEATs, Mistral 7B) are used under their original Apache 2.0 / MIT licenses.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Implementation Plan](./docs/AnimalVox_Implementation_Plan.docx) | 8-phase build roadmap with week-by-week tasks |
| [Project Explainer](./docs/AnimalVox_Project_Explainer.docx) | Plain-language explanation of the whole system |
| [API Keys Guide](./docs/AnimalVox_API_Keys_Guide.docx) | Every external service, how to register, where to get keys |

---

<div align="center">

<br/>

**Built with 🌿 for the animals that share our world**

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:27AE60,100:0A6B6B&height=100&section=footer" width="100%"/>

</div>
