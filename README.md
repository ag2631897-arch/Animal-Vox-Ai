<div align="center">

<!-- LOGO / BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0A6B6B,100:27AE60&height=200&section=header&text=AnimalVox%20AI&fontSize=60&fontColor=ffffff&fontAlignY=38&desc=Animal%20Vocalization%20Intelligence%20Platform&descAlignY=58&descSize=18" width="100%"/>

<br/>

<p>
  <img src="https://img.shields.io/badge/Status-Active%20Development-27AE60?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-0A6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/React-18%20%2B%20Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
</p>

<h3>
  <em>Listen to the wild. Understand it.</em>
</h3>

<p align="center">
  AnimalVox AI is a behavioral bioacoustics platform that analyzes and translates animal vocalizations — birds, dogs, cats, dolphins, whales, and frogs — into human-readable behavioral insights grounded in ethological science.
</p>

</div>

---

## 🌿 What Is AnimalVox AI?

AnimalVox AI is **not** a gimmick translator. Animals don't have human words, but their vocalizations carry measurable **behavioral and emotional intent**.

When a bird produces a rapid series of thin, high-pitched calls, ornithological research identifies this as an **aerial predator alarm**. AnimalVox AI combines pretrained audio tagging, acoustic-prosodic deep learning, embedding-based anti-spoofing, and ethological knowledge retrieval to translate vocalizations into clear behavioral interpretations:

> *"Freeze — there's a hawk directly overhead. Don't move a muscle."*

---

## ✨ Key Features

- 🎙️ **Record or Upload Audio** — Supports `.wav`, `.mp3`, `.flac`, `.ogg`, and `.webm` with live microphone capture and streaming translation responses
- 🛡️ **Stage 0 Anti-Spoofing & Audio Gatekeeper** — Uses a pretrained **PANNs CNN14** AudioSet 527-class sound-event tagger (`models/audio_tagger.py`), an **embedding-based anti-spoofing classifier** (`models/anti_spoofing/`), and a **music detector** (`models/music_detector.py`) to reject human speech, human imitations, music, and background SFX
- 🐾 **Automatic Species Detection & Auto-Routing** — Identifies Birds, Dogs, Cats, Dolphins, Whales, and Frogs automatically (`models/animal_detector/`)
- 🧠 **Multi-Tier Intelligence Pipeline** — Dual-stream acoustic encoding, 12-class behavioral classification (`B00`–`B11`) with intensity/urgency scoring, sequential call-series reasoning, ChromaDB ethology retrieval, and Groq-powered NLG synthesis
- 📊 **5 Pet Intelligence Modules**:
  - **Module 01 — Health Monitoring** (`models/health_monitor/`): Acoustic anomaly tracking and weekly health signals
  - **Module 02 — Breed Identification** (`models/breed_identifier/`): Acoustic + optional visual multimodal breed inference
  - **Module 03 — Emotional & Behavioral Tracking** (`models/emotional_tracker/`): Weekly emotional trends and timelines
  - **Module 04 — Wellbeing Signals Dashboard** (`models/wellbeing_signals/`): Aggregated pet wellbeing telemetry
  - **Module 05 — Animal Detection & Auto-Routing** (`models/animal_detector/`): Ambient species detection
- 🔐 **Multi-User Auth & Data Isolation** — Email/password JWT authentication, Google OAuth 2.0, and **NeonDB PostgreSQL** persistence (`database/`) with per-user pet/history isolation and client-side **IndexedDB** audio clip storage
- ⚡ **Production Hardening & Deployment** — In-memory inference caching (`inference/cache.py`), IP/route rate limiting (`inference/security.py`), Dockerfiles + Compose, and Vercel SPA configuration

---

## 🏗️ Architecture

```
Raw Audio (.wav / .mp3 / .flac / .ogg / .webm)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│        STAGE 0: ANTI-SPOOFING & SOUND GATEKEEPER      │
│  PANNs CNN14 (AudioSet 527) + Embedding Anti-Spoofing │
│  + Music Detector → Reject Human Voice / Imitation    │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│         STAGE 1 & 2: PREPROCESSING & FEATURES         │
│  Resample (16 kHz) → Noise Gate → Segment             │
│  BEATs / Dual-Stream + Mel-Spec CNN + Prosodic Feats  │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│       STAGE 3 & 4: CLASSIFICATION & CONTEXT           │
│  Auto Species Routing + Multi-label Behaviors (B00-11)│
│  Call Sequence Transformer + Ethology KB (ChromaDB)   │
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│     STAGE 5: NLG TRANSLATION & MODULE ENRICHMENT      │
│  Groq Streaming Translator + Health / Breed /         │
│  Emotion / Wellbeing Signal Updates                   │
└───────────────────────────────────────────────────────┘
```

---

## 🐾 Supported Animals

| Animal | Vocalizations Analyzed | Example Translation |
|--------|------------------------|---------------------|
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
| `B00` | `alarm_aerial` | Aerial predator warning |
| `B01` | `alarm_ground` | Ground predator warning |
| `B02` | `territorial_warning` | Defend territory from rivals |
| `B03` | `mating_call` | Attract mate / breeding signal |
| `B04` | `contact_call` | Maintain group cohesion |
| `B05` | `distress_call` | Pain, capture, extreme fear |
| `B06` | `feeding_call` | Food found / foraging |
| `B07` | `play_vocalization` | Social play / excitement |
| `B08` | `submission_signal` | Deference / appeasement |
| `B09` | `aggression_warning` | Direct threat / imminent attack |
| `B10` | `social_bonding` | Affiliation maintenance |
| `B11` | `navigation_echolocation` | Spatial orientation / hunting |

---

## 🛠️ Tech Stack

### Backend & ML
| Component | Technology |
|-----------|-----------|
| API Server | FastAPI + Uvicorn |
| Deep Learning | PyTorch + torchaudio + torchvision + timm |
| Audio Tagging | PANNs CNN14 (AudioSet 527 classes, 16 kHz) |
| Acoustic & Prosodic Analysis | librosa, soundfile, parselmouth, audiomentations |
| Sequence & Classification | Custom Dual-Stream Classifier, Spectrogram CNN, Call Sequence Transformer |
| Vector Database | ChromaDB (`knowledge_base/`) |
| LLM Translation | Groq API (`groq`) with structured ethology prompt templates + offline fallbacks |
| Persistence & Auth | NeonDB PostgreSQL (`psycopg2`), JWT, bcrypt, Google OAuth 2.0 |

### Frontend & Deployment
| Component | Technology |
|-----------|-----------|
| Web UI | React 18, Vite 5, Tailwind CSS, Framer Motion, Zustand, React Router |
| Client Audio Persistence | IndexedDB (`frontend/src/lib/audioStorage.js`) |
| Containers | Docker (`Dockerfile.backend`, `Dockerfile.frontend`) + Docker Compose |
| Frontend Hosting | Vercel (`vercel.json`, `frontend/vercel.json`) |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (with `npm` or `pnpm`)
- Optional: CUDA-capable GPU for faster model inference/training

### 1. Clone & Install Backend Dependencies

```bash
git clone https://github.com/ag2631897-arch/Animal-Vox-Ai.git
cd Animal-Vox-Ai

python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Key environment variables in `.env.example`:

```env
# Application
NODE_ENV=production
PORT=3000
API_BACKEND=http://127.0.0.1:8000

# NeonDB PostgreSQL & Session Security
DATABASE_URL=postgresql://user:password@ep-small-cell-pooler.aws.neon.tech/neondb?sslmode=require
JWT_SECRET=generate_a_long_random_secret_string_for_production
COOKIE_SECURE=true

# Google OAuth 2.0 (optional)
GOOGLE_CLIENT_ID=your_google_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_google_client_secret

# CORS & External Services
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
GROQ_API_KEY=your_groq_api_key_here

# Optional Model / Dataset / Tracking Tokens
HF_TOKEN=your_huggingface_token_here
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=animalvox-ai
```

### 3. Build & Run Locally

**Option A — Single FastAPI server serving API + built React frontend:**

```bash
# 1. Build the React frontend
cd frontend
npm install
npm run build

# 2. Start the FastAPI backend
cd ..
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000
```

**Option B — Separate Vite dev server (hot reload) + FastAPI API:**

```bash
# Terminal 1: FastAPI backend on port 8000
python -m uvicorn inference.api:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend server on port 3000
cd frontend
npm run dev:vite
# Open http://localhost:3000
```

**Option C — Docker Compose:**

```bash
docker compose up --build
```

- Frontend container runs on `http://localhost:3000`
- Backend API container runs on `http://localhost:8000`

---

## 🔌 Core API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and model/pipeline status |
| `POST` | `/api/v1/translate` | Full audio analysis & behavioral translation |
| `POST` | `/api/v1/translate/stream` | Streaming translation endpoint |
| `POST` | `/api/v1/reset` | Reset session sequence state |
| `GET` | `/api/v1/species` | List supported species groups |
| `GET` | `/api/v1/health/weekly/{pet_id}` | Weekly pet health metrics |
| `GET` | `/api/v1/emotion/weekly/{pet_id}` | Weekly emotional timeline |
| `GET` | `/api/v1/wellbeing/{pet_id}` | Pet wellbeing dashboard signals |
| `GET` / `POST` | `/api/v1/user/state`, `/api/v1/user/sync` | Per-user state retrieval and sync |
| `POST` / `DELETE` | `/api/v1/pets`, `/api/v1/pets/{pet_id}` | Create or remove user pet profiles |
| `POST` | `/api/v1/translations` | Persist translation history entries |
| `GET` / `POST` | `/api/v1/auth/*` | Signup, login, Google OAuth, logout, and session check |
| `GET` | `/api/v1/cache/stats` | Inference cache metrics |

---

## 📁 Repository Structure

```
Animal-Vox-Ai/
├── config.py                      # Central audio, model, anti-spoofing, and inference config
├── requirements.txt               # Python dependencies
├── Dockerfile.backend             # Backend FastAPI Docker image
├── Dockerfile.frontend            # Frontend Node/Vite Docker image
├── docker-compose.yml             # Multi-service container orchestration
├── package.json                   # Root workspace helper script for Vercel
├── vercel.json                    # Root Vercel monorepo build & SPA routing config
├── .env.example                   # Environment variable template
│
├── database/                      # NeonDB PostgreSQL layer
│   ├── connection.py              # Connection pool & schema initialization
│   ├── auth.py                    # JWT, bcrypt, and Google OAuth 2.0
│   └── user_data.py               # Per-user ownership & telemetry isolation
│
├── inference/                     # Production API & inference orchestration
│   ├── api.py                     # FastAPI REST & streaming server
│   ├── pipeline.py                # End-to-end 5-stage + 5-module inference pipeline
│   ├── cache.py                   # Inference response caching
│   └── security.py                # Rate limiting & request protection
│
├── models/                        # Bioacoustic ML models & gatekeepers
│   ├── audio_tagger.py            # Pretrained PANNs CNN14 AudioSet 527-class tagger
│   ├── music_detector.py          # Music/SFX gatekeeper
│   ├── dual_stream_classifier.py  # Dual-stream acoustic classifier
│   ├── weight_manager.py          # Model weight loading & management
│   ├── acoustic_encoder/          # Acoustic feature encoder
│   ├── animal_detector/           # Module 05: Species detection & auto-routing
│   ├── anti_spoofing/             # Stage 0: Embedding-based human/imitation rejection
│   ├── behavioral_classifier/     # Multi-label behavior & prosodic classifier
│   ├── breed_identifier/          # Module 02: Acoustic + visual breed identifier
│   ├── emotional_tracker/         # Module 03: Emotional tracking
│   ├── health_monitor/            # Module 01: Health anomaly monitor
│   ├── wellbeing_signals/         # Module 04: Wellbeing dashboard aggregator
│   ├── sequence_model/            # Call-series Transformer
│   └── nlg_layer/                 # Ethology prompt templates & Groq translator
│
├── knowledge_base/                # Ethology knowledge base
│   ├── ethology_db.py             # ChromaDB interface
│   ├── populate_db.py             # Seed script
│   └── seed_data/                 # Species ethology JSON entries
│
├── data/                          # Datasets, downloaders, and preprocessing
│   ├── download_scripts/          # Dataset & weight downloaders
│   ├── preprocessing/             # Audio standardization, spectrograms, augmentation
│   ├── label_maps/                # Behavior label mappings
│   └── sample_audio/              # Sample audio clips
│
├── training/                      # Training & dataset streaming pipelines
│   ├── bootstrap_anti_spoofing.py # Anti-spoofing training data bootstrap
│   ├── train_anti_spoofing*.py    # Anti-spoofing & embedding classifier training
│   ├── train_from_shards.py       # Sharded dataset training
│   ├── train_max_accuracy.py      # High-accuracy training pipeline
│   └── stream_*_pipeline.py       # AudioSet & universal animal stream pipelines
│
├── frontend/                      # React 18 + Vite 5 web application
│   ├── src/                       # Pages, screens, UI components, hooks, store, IndexedDB
│   ├── public/                    # Static assets, SEO sitemap, robots.txt
│   ├── server.js                  # Production Node proxy/SSR server
│   ├── vercel.json                # Vercel frontend output & rewrite config
│   └── package.json               # Frontend scripts and dependencies
│
├── evaluation/                    # Evaluation scripts
└── test_*.py                      # End-to-end & regression test suites
```

---

## ⚠️ Limitations & Disclaimers

- **Animals don't have human language.** Translations represent behavioral and emotional interpretations grounded in ethology, not literal speech.
- **Not a veterinary diagnostic tool.** Health and wellbeing scores are informational acoustic signals and do not replace professional veterinary care.
- **Recording environment matters.** Heavy wind, distant microphones, or overlapping noise lower classification confidence.

---

## 📄 License

This project is licensed under the **MIT License**. Datasets and pretrained backbones (such as PANNs CNN14 and BEATs) are subject to their respective original licenses.

---

<div align="center">

**Built with 🌿 for the animals that share our world**

</div>
