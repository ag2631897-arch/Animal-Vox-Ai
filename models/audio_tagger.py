"""
AnimalVox AI — Pretrained AudioSet Sound-Event Tagger (PANNs CNN14, 16 kHz)
===========================================================================
Frozen, pretrained encoder trained on AudioSet's 527 sound classes
(Kong et al., 2020, "PANNs: Large-Scale Pretrained Audio Neural Networks").

Provides unified, calibrated per-segment and clip-level probabilities for:
- Supported animals: dogs, cats, birds, frogs, marine (whales/dolphins)
- Unsupported animals: horse, cow/cattle, pig, sheep, goat, chicken/rooster, etc.
- Human vocalizations: conversational speech, non-speech human voice (laugh, cough,
  shout, scream, grunt, groan, sigh, whistle, crying, singing)
- Music & sound effects (SFX)
"""

from __future__ import annotations

import csv
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANNS_DIR = PROJECT_ROOT / "models" / "weights" / "panns"
CHECKPOINT = PANNS_DIR / "Cnn14_16k_mAP=0.438.pth"
LABELS_CSV = PANNS_DIR / "class_labels_indices.csv"

SAMPLE_RATE = 16000
SEGMENT_SEC = 1.0
HOP_SEC = 0.5


# ── CNN14 architecture (reference implementation, 16 kHz config) ──────────

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    def __init__(self, sample_rate=16000, window_size=512, hop_size=160,
                 mel_bins=64, fmin=50, fmax=8000, classes_num=527):
        super().__init__()
        from torchlibrosa.stft import Spectrogram, LogmelFilterBank

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

    def forward(self, waveform: torch.Tensor):
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)            # (B, 1, T, mel)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        for blk in (self.conv_block1, self.conv_block2, self.conv_block3,
                    self.conv_block4, self.conv_block5):
            x = F.dropout(blk(x, pool_size=(2, 2)), p=0.2, training=False)
        x = F.dropout(self.conv_block6(x, pool_size=(1, 1)), p=0.2, training=False)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        embedding = F.relu_(self.fc1(x))
        clipwise = torch.sigmoid(self.fc_audioset(embedding))
        return clipwise, embedding


# ── AudioSet class groups ────────────────────────────────────────────────

GROUPS: Dict[str, List[str]] = {
    # Supported species
    "dogs": ["Dog", "Bark", "Yip", "Howl", "Bow-wow", "Growling",
             "Whimper (dog)", "Canidae, dogs, wolves"],
    "cats": ["Cat", "Purr", "Meow", "Hiss", "Caterwaul"],
    "birds": ["Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet",
              "Squawk", "Pigeon, dove", "Coo", "Crow", "Caw", "Owl", "Hoot",
              "Bird flight, flapping wings", "Gull, seagull"],
    "frogs": ["Frog", "Croak"],
    "marine": ["Whale vocalization"],
    # Unsupported animals (still animals — must return unsupported_species, not human)
    "unsupported_animal": [
        "Livestock, farm animals, working animals", "Horse", "Clip-clop",
        "Neigh, whinny", "Cattle, bovinae", "Moo", "Pig", "Oink",
        "Goat", "Bleat", "Sheep", "Fowl", "Chicken, rooster", "Cluck",
        "Crowing, cock-a-doodle-doo", "Turkey", "Gobble", "Duck", "Quack",
        "Goose", "Honk", "Roaring cats (lions, tigers)", "Roar"],
    # Humans
    "speech": ["Speech", "Male speech, man speaking", "Female speech, woman speaking",
               "Child speech, kid speaking", "Conversation", "Narration, monologue",
               "Babbling", "Speech synthesizer", "Whispering", "Chatter",
               "Hubbub, speech noise, speech babble", "Television", "Radio"],
    "human_voice_nonspeech": [
        "Shout", "Bellow", "Whoop", "Yell", "Battle cry", "Children shouting",
        "Screaming", "Laughter", "Baby laughter", "Giggle", "Snicker",
        "Belly laugh", "Chuckle, chortle", "Crying, sobbing", "Baby cry, infant cry",
        "Whimper", "Wail, moan", "Sigh", "Groan", "Grunt", "Humming",
        "Whistling", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort",
        "Cough", "Throat clearing", "Sneeze", "Sniff", "Yodeling", "Chant",
        "Mantra", "Male singing", "Female singing", "Child singing",
        "Synthetic singing", "Rapping", "Beatboxing"],
    "music": ["Music", "Musical instrument", "Singing", "Song", "Choir",
              "Background music", "Theme music", "Pop music", "Hip hop music",
              "Rock music", "Electronic music", "Techno", "Dance music",
              "Soundtrack music", "Classical music", "Jingle (music)",
              "Guitar", "Piano", "Drum", "Drum kit", "Orchestra",
              "Bass guitar", "Electric guitar", "Percussion",
              "Video game music", "Lullaby", "Christmas music",
              "Music of Bollywood", "Music of Asia", "Saxophone", "Harp",
              "Brass instrument", "Folk music", "Sitar", "Dubstep", "Drum and bass"],
    "sfx": ["Sound effect", "Whoosh, swoosh, swish", "Boom", "Explosion",
            "Gunshot, gunfire", "Shatter", "Smash, crash"],
}

UNSUPPORTED_ANIMAL_MAP: List[Tuple[str, List[str]]] = [
    ("cow / cattle", ["Cattle, bovinae", "Moo"]),
    ("horse", ["Horse", "Neigh, whinny", "Clip-clop"]),
    ("pig", ["Pig", "Oink"]),
    ("goat / sheep", ["Goat", "Bleat", "Sheep"]),
    ("chicken / rooster", ["Chicken, rooster", "Cluck", "Crowing, cock-a-doodle-doo", "Fowl"]),
    ("duck / goose", ["Duck", "Quack", "Goose", "Honk"]),
    ("turkey", ["Turkey", "Gobble"]),
    ("lion / tiger", ["Roaring cats (lions, tigers)", "Roar"]),
]

GENERIC_ANIMAL = ["Animal", "Domestic animals, pets", "Wild animals"]
HUMAN_VOICE = ["Human voice"]

SUPPORTED_SPECIES = ("dogs", "cats", "birds", "frogs", "marine")


@dataclass
class SegmentTags:
    start_sec: float
    end_sec: float
    rms: float
    groups: Dict[str, float]          # max prob within each group
    animal: float                     # max over supported+unsupported+generic
    supported_animal: float           # max over supported species + generic_animal
    human_voice: float
    top: List[tuple]                  # top-5 (label, prob)


@dataclass
class TagResult:
    clip_groups: Dict[str, float]
    clip_top: List[tuple]
    segments: List[SegmentTags] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    raw_probs: Optional[np.ndarray] = None

    @property
    def max_supported_species(self) -> Tuple[str, float]:
        """Return (species_key, max_prob) across clip and segments."""
        best_sp = "dogs"
        best_val = -1.0
        for sp in SUPPORTED_SPECIES:
            c_val = self.clip_groups.get(sp, 0.0)
            s_val = max((s.groups.get(sp, 0.0) for s in self.segments), default=0.0)
            val = max(c_val, 0.85 * s_val)
            if val > best_val:
                best_val = val
                best_sp = sp
        return best_sp, float(best_val)

    @property
    def any_animal_prob(self) -> float:
        _, sup_val = self.max_supported_species
        c_gen = self.clip_groups.get("generic_animal", 0.0)
        s_gen = max((s.groups.get("generic_animal", 0.0) for s in self.segments), default=0.0)
        c_uns = self.clip_groups.get("unsupported_animal", 0.0)
        s_uns = max((s.groups.get("unsupported_animal", 0.0) for s in self.segments), default=0.0)
        return float(max(sup_val, c_gen, 0.85 * s_gen, c_uns, 0.85 * s_uns))


class AudioTagger:
    """Singleton wrapper around frozen PANNs CNN14 (16 kHz)."""

    _instance: Optional["AudioTagger"] = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.available = False
        self.model = None
        self.labels: List[str] = []
        self.name_to_idx: Dict[str, int] = {}
        self.group_idx: Dict[str, List[int]] = {}
        self.generic_idx: List[int] = []
        self.human_voice_idx: List[int] = []
        self._tag_cache: Dict[tuple, TagResult] = {}
        if not CHECKPOINT.exists() or not LABELS_CSV.exists():
            logger.warning(f"AudioTagger disabled: checkpoint missing at {CHECKPOINT}")
            return
        try:
            self.labels = self._load_labels()
            model = Cnn14()
            ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device).eval()
            self.model = model
            self.name_to_idx = {n: i for i, n in enumerate(self.labels)}
            for g, names in GROUPS.items():
                idx = [self.name_to_idx[n] for n in names if n in self.name_to_idx]
                self.group_idx[g] = idx
            self.generic_idx = [self.name_to_idx[n] for n in GENERIC_ANIMAL if n in self.name_to_idx]
            self.human_voice_idx = [self.name_to_idx[n] for n in HUMAN_VOICE if n in self.name_to_idx]
            self.available = True
            logger.info("AudioTagger: ✅ PANNs CNN14 (16 kHz, AudioSet mAP 0.438) loaded")
        except Exception as e:
            self.model = None
            logger.warning(f"AudioTagger unavailable ({e}). "
                           f"Expected checkpoint at {CHECKPOINT}")

    @classmethod
    def get(cls, device: str = "cpu") -> "AudioTagger":
        with cls._lock:
            if cls._instance is None:
                cls._instance = AudioTagger(device=device)
            return cls._instance

    @staticmethod
    def _load_labels() -> List[str]:
        with open(LABELS_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return [r[2] for r in rows[1:]]

    def _forward(self, batch: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(batch.astype(np.float32)).to(self.device)
            probs, emb = self.model(x)
        return probs.cpu().numpy(), emb.cpu().numpy()

    def _summarise(self, probs: np.ndarray) -> Dict[str, float]:
        out = {g: float(probs[idx].max()) if idx else 0.0
               for g, idx in self.group_idx.items()}
        out["generic_animal"] = float(probs[self.generic_idx].max()) if self.generic_idx else 0.0
        out["human_voice"] = float(probs[self.human_voice_idx].max()) if self.human_voice_idx else 0.0
        return out

    def _top(self, probs: np.ndarray, k: int = 5) -> List[tuple]:
        order = np.argsort(probs)[::-1][:k]
        return [(self.labels[i], round(float(probs[i]), 3)) for i in order]

    @staticmethod
    def _trim_zero_padding(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Remove synthetic zero-padding added by 5-second fixed windowing."""
        nz = np.where(np.abs(audio) > 1e-4)[0]
        if len(nz) == 0:
            return audio
        pad = int(0.05 * sr)
        start = max(0, int(nz[0]) - pad)
        end = min(len(audio), int(nz[-1]) + 1 + pad)
        return audio[start:end]

    def tag(self, audio: np.ndarray, sr: int = SAMPLE_RATE,
            segment_sec: float = SEGMENT_SEC, hop_sec: float = HOP_SEC,
            min_rms: float = 0.004) -> TagResult:
        """Clip-level + per-segment AudioSet tags on trimmed active audio."""
        if not self.available:
            raise RuntimeError("AudioTagger not available")
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = np.asarray(audio, dtype=np.float32)

        cache_key = (
            hashlib.blake2b(audio.tobytes(), digest_size=16).hexdigest(),
            sr, segment_sec, hop_sec, min_rms,
        )
        cached = self._tag_cache.get(cache_key)
        if cached is not None:
            return cached

        # Strip synthetic zero-padding so silence doesn't dilute CNN14 pooling
        active = self._trim_zero_padding(audio, SAMPLE_RATE)
        if len(active) == 0:
            active = audio

        # If active vocalization is a short burst (< 1.2s, e.g., single dog bark),
        # tile it with a 50ms gap to ~1.5s so CNN14 gets full temporal support.
        min_clip_samples = int(1.2 * SAMPLE_RATE)
        if 0 < len(active) < min_clip_samples:
            gap = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
            unit = np.concatenate([active, gap])
            reps = max(2, int(np.ceil((1.5 * SAMPLE_RATE) / max(1, len(unit)))))
            clip_input = np.tile(unit, reps)[:int(1.8 * SAMPLE_RATE)]
        else:
            clip_input = active

        if len(clip_input) < SAMPLE_RATE:
            clip_input = np.pad(clip_input, (0, SAMPLE_RATE - len(clip_input)))

        clip_probs, clip_emb = self._forward(clip_input[None, :])
        clip_probs = clip_probs[0]
        result = TagResult(
            clip_groups=self._summarise(clip_probs),
            clip_top=self._top(clip_probs),
            embedding=clip_emb[0],
            raw_probs=clip_probs,
        )

        seg = int(segment_sec * SAMPLE_RATE)
        hop = int(hop_sec * SAMPLE_RATE)
        seg_source = active if len(active) >= seg else clip_input
        starts = list(range(0, max(1, len(seg_source) - seg + 1), hop))
        if starts and starts[-1] + seg < len(seg_source):
            starts.append(len(seg_source) - seg)
        windows, meta = [], []
        for s in starts:
            w = seg_source[s:s + seg]
            if len(w) < seg:
                w = np.pad(w, (0, seg - len(w)))
            rms = float(np.sqrt(np.mean(w ** 2)))
            if rms < min_rms:
                continue
            windows.append(w)
            meta.append((s / SAMPLE_RATE, (s + seg) / SAMPLE_RATE, rms))
        for i in range(0, len(windows), 16):
            probs, _ = self._forward(np.stack(windows[i:i + 16]))
            for p, (t0, t1, rms) in zip(probs, meta[i:i + 16]):
                g = self._summarise(p)
                sup_animal = max([g[s] for s in SUPPORTED_SPECIES] + [g["generic_animal"]])
                animal = max(sup_animal, g["unsupported_animal"])
                result.segments.append(SegmentTags(
                    start_sec=t0, end_sec=t1, rms=rms, groups=g,
                    animal=animal, supported_animal=sup_animal,
                    human_voice=g["human_voice"], top=self._top(p)))
        if len(self._tag_cache) >= 32:
            self._tag_cache.pop(next(iter(self._tag_cache)), None)
        self._tag_cache[cache_key] = result
        return result

    def detect_unsupported_animal(self, tag_res: TagResult) -> Optional[str]:
        """
        If clip is clearly an unsupported animal (e.g., cow, horse, pig, sheep, rooster)
        and NOT a supported species, return its human-readable name.
        """
        if tag_res.raw_probs is None:
            return None
        _, sup_prob = tag_res.max_supported_species
        uns_prob = max(
            tag_res.clip_groups.get("unsupported_animal", 0.0),
            max((s.groups.get("unsupported_animal", 0.0) for s in tag_res.segments), default=0.0),
        )
        # Only flag unsupported if unsupported_animal is strong and clearly beats supported species
        if uns_prob < 0.45 or sup_prob >= uns_prob * 0.85:
            return None

        best_name = None
        best_score = 0.0
        for animal_name, label_names in UNSUPPORTED_ANIMAL_MAP:
            idxs = [self.name_to_idx[n] for n in label_names if n in self.name_to_idx]
            if not idxs:
                continue
            score = float(tag_res.raw_probs[idxs].max())
            if score > best_score:
                best_score = score
                best_name = animal_name
        if best_score >= 0.35:
            return best_name
        return None

    def isolate_animal_from_speech(
        self, audio: np.ndarray, sr: int = SAMPLE_RATE, tag_res: Optional[TagResult] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Segment-wise speech masking for mixed audio (human talking + real animal).
        Only removes segments when BOTH genuine human speech segments AND genuine
        animal segments exist in the clip. Never strips pure animal recordings.
        """
        if not self.available:
            return audio, False
        if tag_res is None:
            tag_res = self.tag(audio, sr=sr)
        if len(tag_res.segments) < 2:
            return audio, False

        speech_segs = [
            s for s in tag_res.segments
            if (s.groups.get("speech", 0.0) >= 0.35 or s.groups.get("human_voice_nonspeech", 0.0) >= 0.45)
            and s.supported_animal < 0.25
        ]
        animal_segs = [
            s for s in tag_res.segments
            if s.supported_animal >= 0.25 or s.animal >= 0.30
        ]

        if not speech_segs or not animal_segs:
            return audio, False

        active = self._trim_zero_padding(audio, sr)
        keep_mask = np.zeros(len(active), dtype=bool)
        for s in animal_segs:
            i0 = max(0, int(s.start_sec * sr))
            i1 = min(len(active), int(s.end_sec * sr))
            keep_mask[i0:i1] = True

        kept = active[keep_mask]
        if len(kept) >= int(0.25 * sr) and float(np.sqrt(np.mean(kept ** 2))) >= 0.015:
            # Pad back to original length if caller expects fixed length
            if len(kept) < len(audio):
                kept = np.pad(kept, (0, len(audio) - len(kept)))
            return kept.astype(np.float32), True

        return audio, False


def get_tagger(device: str = "cpu") -> AudioTagger:
    return AudioTagger.get(device=device)
