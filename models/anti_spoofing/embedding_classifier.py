"""
AnimalVox AI — Embedding-Based Anti-Spoofing Classifier
=========================================================
Replaces rule-based formant analysis with a learned classifier operating
on frozen BEATs encoder embeddings.

Architecture:
    Audio → 1-second segments → Frozen BEATs Encoder (256-dim) → MLP (5 classes)

Classes:
    0: real_animal       — Genuine animal vocalization
    1: human_imitation   — Human imitating an animal sound
    2: human_speech      — Normal human conversational speech
    3: other_human       — Cough, laugh, clap, whistle, etc.
    4: noise             — Music, background noise, silence, non-vocal

Decision logic (per-segment, then aggregated):
    - "animal"    → translate
    - "human"     → reject (with reason)
    - "uncertain" → prompt re-record

Scientific basis:
    BEATs (Microsoft, 2022) is pretrained on AudioSet (2M+ clips) and
    encodes spectral texture, temporal modulation and harmonic structure.
    Human vocal tracts produce consistent embedding clusters even when
    imitating animal sounds, because the underlying resonance geometry
    differs from animal anatomy (syrinx, shorter tracts, nasal coupling).
    A simple logistic regression on these embeddings separates the classes
    with a few hundred samples per class.
"""

import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Class definitions ──────────────────────────────────────────────
CLASS_NAMES = ["real_animal", "human_imitation", "human_speech", "other_human", "noise"]
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class SegmentPrediction:
    """Prediction for a single 1-second audio segment."""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]
    segment_index: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class AntiSpoofDecision:
    """Aggregated clip-level decision from per-segment analysis."""
    outcome: str           # "animal" | "human" | "uncertain"
    reason: str
    confidence: float
    segment_predictions: List[SegmentPrediction]
    dominant_class: str
    animal_ratio: float    # Fraction of active segments classified as real_animal
    human_ratio: float     # Fraction classified as any human class
    imitation_ratio: float # Fraction classified as human_imitation specifically
    is_trained: bool = False


# ── MLP Classifier on BEATs Embeddings ─────────────────────────────
class SegmentEmbeddingClassifier(nn.Module):
    """
    5-class MLP classifier on frozen 256-dim BEATs embeddings.

    Architecture: 256 → 128 → ReLU → Dropout → 64 → ReLU → 5
    Parameters: ~33K (trains in seconds on CPU with a few hundred samples).
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BEATs embedding tensor (B, 256)
        Returns:
            Logits (B, 5) for [real_animal, human_imitation, human_speech, other_human, noise]
        """
        return self.classifier(x)


# ── Segment-Wise Anti-Spoofing Analyzer ────────────────────────────
class SegmentWiseAntiSpoof:
    """
    Per-segment anti-spoofing using frozen BEATs embeddings + trained MLP.

    Pipeline:
        1. Split audio into 1-second overlapping segments
        2. Extract BEATs embedding (256-dim) for each active segment
        3. Classify each segment independently (5 classes)
        4. Aggregate segment predictions into clip-level decision
        5. Return: "animal" | "human" | "uncertain"

    When the classifier has no trained weights, returns "uncertain"
    to let the caller fall back to a heuristic or pass through.
    """

    SEGMENT_DURATION_SEC = 1.0    # seconds per segment
    SEGMENT_HOP_SEC = 0.5        # overlap hop (50%)
    MIN_ACTIVE_RMS = 0.012       # minimum RMS to consider a segment active
    MIN_SEGMENT_SEC = 0.3        # minimum remaining audio to create a segment

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.has_trained_weights = False
        self.metadata: Dict = {}

        # Frozen BEATs encoder (shared singleton — no extra memory)
        try:
            from training.feature_extractor import get_beats_encoder
            self.encoder = get_beats_encoder(device=device)
        except Exception as e:
            logger.warning(f"Could not load BEATs encoder for anti-spoof: {e}")
            self.encoder = None

        # Trainable MLP classifier (33K params)
        self.classifier = SegmentEmbeddingClassifier(
            embed_dim=256, num_classes=NUM_CLASSES, hidden_dim=128
        ).to(device).eval()

        # Attempt to load trained weights
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_weights, self.metadata = load_checkpoint(
                self.classifier, "anti_spoofing_embedding", device=device
            )
            if self.has_trained_weights:
                acc = self.metadata.get("metrics", {}).get("val_accuracy", "N/A")
                fpr = self.metadata.get("metrics", {}).get("animal_false_positive_rate", "N/A")
                logger.info(
                    f"SegmentWiseAntiSpoof: ✅ TRAINED embedding classifier loaded "
                    f"(Acc: {acc}, Animal FPR: {fpr})"
                )
        except Exception as e:
            logger.debug(f"No trained embedding classifier weights found: {e}")

        if not self.has_trained_weights:
            logger.info(
                "SegmentWiseAntiSpoof: No trained weights. "
                "Embedding classifier will return 'uncertain' until trained. "
                "Run: python training/train_anti_spoofing_embedding.py"
            )

    # ── Segmentation ───────────────────────────────────────────────
    def _segment_audio(self, audio: np.ndarray, sr: int) -> List[Tuple[np.ndarray, float, float]]:
        """
        Split audio into overlapping 1-second segments.
        Returns list of (segment_audio, start_sec, end_sec).
        """
        seg_samples = int(self.SEGMENT_DURATION_SEC * sr)
        hop_samples = int(self.SEGMENT_HOP_SEC * sr)

        if len(audio) < seg_samples:
            padded = np.zeros(seg_samples, dtype=audio.dtype)
            padded[:len(audio)] = audio
            return [(padded, 0.0, len(audio) / sr)]

        segments = []
        start = 0
        while start + seg_samples <= len(audio):
            seg = audio[start: start + seg_samples]
            segments.append((seg, start / sr, (start + seg_samples) / sr))
            start += hop_samples

        # Remaining tail
        if start < len(audio) and (len(audio) - start) >= int(self.MIN_SEGMENT_SEC * sr):
            remaining = audio[start:]
            padded = np.zeros(seg_samples, dtype=audio.dtype)
            padded[:len(remaining)] = remaining
            segments.append((padded, start / sr, len(audio) / sr))

        return segments

    # ── Embedding extraction ───────────────────────────────────────
    def _extract_embedding(self, segment: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract 256-dim BEATs embedding for a single segment."""
        try:
            from training.feature_extractor import extract_beats_embedding
            return extract_beats_embedding(segment, sr=sr, embed_dim=256)
        except Exception as e:
            logger.debug(f"Embedding extraction failed: {e}")
            return None

    # ── Core analysis ──────────────────────────────────────────────
    def analyze(self, audio: np.ndarray, sr: int = 16000) -> AntiSpoofDecision:
        """
        Run per-segment embedding-based anti-spoofing analysis.

        Returns AntiSpoofDecision with outcome in {"animal", "human", "uncertain"}.
        """
        if not self.has_trained_weights or self.encoder is None:
            return AntiSpoofDecision(
                outcome="uncertain",
                reason="Embedding anti-spoof classifier not yet trained.",
                confidence=0.0,
                segment_predictions=[],
                dominant_class="unknown",
                animal_ratio=0.0,
                human_ratio=0.0,
                imitation_ratio=0.0,
                is_trained=False,
            )

        segments = self._segment_audio(audio, sr)
        predictions: List[SegmentPrediction] = []

        for idx, (seg, start_s, end_s) in enumerate(segments):
            rms = float(np.sqrt(np.mean(seg ** 2)))
            if rms < self.MIN_ACTIVE_RMS:
                continue  # skip silent segments

            emb = self._extract_embedding(seg, sr)
            if emb is None:
                continue

            emb_tensor = torch.from_numpy(emb).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.classifier(emb_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            class_id = int(np.argmax(probs))
            predictions.append(SegmentPrediction(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                confidence=float(probs[class_id]),
                probabilities={name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
                segment_index=idx,
                start_sec=start_s,
                end_sec=end_s,
            ))

        if not predictions:
            return AntiSpoofDecision(
                outcome="uncertain",
                reason="No active audio segments found.",
                confidence=0.0,
                segment_predictions=[],
                dominant_class="noise",
                animal_ratio=0.0,
                human_ratio=0.0,
                imitation_ratio=0.0,
                is_trained=True,
            )

        return self._aggregate(predictions)

    # ── Aggregation logic ──────────────────────────────────────────
    def _aggregate(self, predictions: List[SegmentPrediction]) -> AntiSpoofDecision:
        """
        Aggregate per-segment predictions into a clip-level decision.

        Decision rules (per the user's spec):
        - Block ONLY when imitation segments dominate animal-like events
          by duration or energy.
        - If speech is background, translate the animal segments.
        - Return "uncertain" for ambiguous cases → prompt re-record.
        """
        n = len(predictions)
        if n == 0:
            return AntiSpoofDecision(
                outcome="uncertain", reason="No predictions.", confidence=0.0,
                segment_predictions=[], dominant_class="noise",
                animal_ratio=0.0, human_ratio=0.0, imitation_ratio=0.0,
                is_trained=True,
            )

        counts = {name: 0 for name in CLASS_NAMES}
        for p in predictions:
            counts[p.class_name] += 1

        animal_ratio = counts["real_animal"] / n
        imitation_ratio = counts["human_imitation"] / n
        speech_ratio = counts["human_speech"] / n
        other_human_ratio = counts["other_human"] / n
        noise_ratio = counts["noise"] / n
        human_ratio = imitation_ratio + speech_ratio + other_human_ratio

        dominant_class = max(counts, key=counts.get)

        # ── Rule 1: Imitation dominates → REJECT ──────────────────
        if imitation_ratio >= 0.50:
            return AntiSpoofDecision(
                outcome="human",
                reason=(
                    f"This doesn't sound like an animal. "
                    f"Human imitation detected in {imitation_ratio:.0%} of audio segments."
                ),
                confidence=imitation_ratio,
                segment_predictions=predictions,
                dominant_class=dominant_class,
                animal_ratio=animal_ratio,
                human_ratio=human_ratio,
                imitation_ratio=imitation_ratio,
                is_trained=True,
            )

        # ── Rule 2: Pure speech, no animal → REJECT ──────────────
        if speech_ratio >= 0.60 and animal_ratio < 0.15:
            return AntiSpoofDecision(
                outcome="human",
                reason=(
                    f"No animal vocalization detected. "
                    f"Human speech found in {speech_ratio:.0%} of audio segments."
                ),
                confidence=speech_ratio,
                segment_predictions=predictions,
                dominant_class=dominant_class,
                animal_ratio=animal_ratio,
                human_ratio=human_ratio,
                imitation_ratio=imitation_ratio,
                is_trained=True,
            )

        # ── Rule 3: Other human sounds dominate → REJECT ─────────
        if other_human_ratio >= 0.60 and animal_ratio < 0.15:
            return AntiSpoofDecision(
                outcome="human",
                reason=(
                    f"No animal vocalization detected. "
                    f"Human sounds (cough, laugh, etc.) in {other_human_ratio:.0%} of segments."
                ),
                confidence=other_human_ratio,
                segment_predictions=predictions,
                dominant_class=dominant_class,
                animal_ratio=animal_ratio,
                human_ratio=human_ratio,
                imitation_ratio=imitation_ratio,
                is_trained=True,
            )

        # ── Rule 4: Clear animal → PASS ──────────────────────────
        if animal_ratio >= 0.35 and imitation_ratio < 0.25:
            return AntiSpoofDecision(
                outcome="animal",
                reason=f"Animal vocalization detected in {animal_ratio:.0%} of segments.",
                confidence=animal_ratio,
                segment_predictions=predictions,
                dominant_class=dominant_class,
                animal_ratio=animal_ratio,
                human_ratio=human_ratio,
                imitation_ratio=imitation_ratio,
                is_trained=True,
            )

        # ── Rule 5: Mixed — animal + background speech → PASS ────
        # (Background speech should not block translation of real animal)
        if animal_ratio >= 0.20 and speech_ratio > 0.0 and imitation_ratio < 0.20:
            return AntiSpoofDecision(
                outcome="animal",
                reason=(
                    f"Animal vocalization detected ({animal_ratio:.0%}) "
                    f"with background human speech ({speech_ratio:.0%}). Translating animal segments."
                ),
                confidence=animal_ratio,
                segment_predictions=predictions,
                dominant_class=dominant_class,
                animal_ratio=animal_ratio,
                human_ratio=human_ratio,
                imitation_ratio=imitation_ratio,
                is_trained=True,
            )

        # ── Rule 6: Uncertain — ambiguous ─────────────────────────
        return AntiSpoofDecision(
            outcome="uncertain",
            reason=(
                f"Ambiguous audio: {animal_ratio:.0%} animal, "
                f"{imitation_ratio:.0%} imitation, {speech_ratio:.0%} speech, "
                f"{noise_ratio:.0%} noise. Please re-record with a clearer sample."
            ),
            confidence=max(animal_ratio, human_ratio),
            segment_predictions=predictions,
            dominant_class=dominant_class,
            animal_ratio=animal_ratio,
            human_ratio=human_ratio,
            imitation_ratio=imitation_ratio,
            is_trained=True,
        )

    # ── Diagnostic logging ─────────────────────────────────────────
    def diagnose(self, audio: np.ndarray, sr: int = 16000) -> Dict:
        """
        Detailed diagnostic output for calibration testing.
        Logs per-segment scores for the 30-minute diagnostic protocol.
        """
        decision = self.analyze(audio, sr)
        segments_log = []
        for p in decision.segment_predictions:
            segments_log.append({
                "segment": p.segment_index,
                "start": round(p.start_sec, 2),
                "end": round(p.end_sec, 2),
                "class": p.class_name,
                "confidence": round(p.confidence, 3),
                "probs": {k: round(v, 3) for k, v in p.probabilities.items()},
            })

        return {
            "outcome": decision.outcome,
            "reason": decision.reason,
            "confidence": round(decision.confidence, 3),
            "animal_ratio": round(decision.animal_ratio, 3),
            "human_ratio": round(decision.human_ratio, 3),
            "imitation_ratio": round(decision.imitation_ratio, 3),
            "dominant_class": decision.dominant_class,
            "num_active_segments": len(decision.segment_predictions),
            "is_trained": decision.is_trained,
            "segments": segments_log,
        }
