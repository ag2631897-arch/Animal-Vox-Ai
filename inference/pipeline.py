"""
AnimalVox AI — Enhanced End-to-End Inference Pipeline (SRS v1.1)
=================================================================
Integrates all 5 intelligence modules + anti-spoofing + streaming:

Stage 0: Anti-Spoofing Gatekeeper (human imitation rejection)
Stage 1: Audio Preprocessing
Stage 2: Feature Extraction (BEATs + CNN + Prosodic)
Stage 3: Behavioral Classification
Stage 4: Context Reasoning (Sequence Transformer)
Stage 5: NLG Translation (Groq streaming)

Module Enrichment (parallel):
  - Module 01: Pet Health Monitoring
  - Module 02: Breed Identification
  - Module 03: Emotional & Behavioural Tracking
  - Module 04: Wellbeing Signals Dashboard
  - Module 05: Animal Detection & Auto-Routing
"""

import os
import json
import time
import asyncio
import logging
import re
import numpy as np
import torch
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AUDIO_CONFIG, MODEL_CONFIG, INFERENCE_CONFIG, STREAMING_CONFIG,
    BEHAVIOR_LABELS, SPECIES_GROUPS, NUM_SPECIES_GROUPS,
    MEDICAL_DISCLAIMER
)
from data.preprocessing.standardize_audio import preprocess_audio, preprocess_audio_bytes
from data.preprocessing.generate_spectrograms import generate_mel_spectrogram, mel_to_image
from models.behavioral_classifier.prosodic_extractor import extract_prosodic_features
from models.acoustic_encoder.encoder import AnimalAcousticEncoder
from models.behavioral_classifier.classifier import BehavioralClassifier, SpectrogramCNN
from models.sequence_model.call_sequence_transformer import CallSequenceTransformer
from models.nlg_layer.prompt_templates import build_prompt, SYSTEM_PROMPT
from models.nlg_layer.bioacoustic_translator import generate_human_translation
from knowledge_base.ethology_db import EthologyKnowledgeBase

# New module imports (SRS v1.1)
from models.anti_spoofing.detector import AntiSpoofingGatekeeper
from models.animal_detector.detector import AnimalDetectionPipeline
from models.health_monitor.monitor import HealthMonitor
from models.breed_identifier.identifier import BreedIdentifier
from models.emotional_tracker.tracker import EmotionalTracker
from models.wellbeing_signals.dashboard import WellbeingDashboard

logger = logging.getLogger(__name__)


class AnimalVoxPipeline:
    """
    Complete AnimalVox AI inference pipeline (SRS v1.1).

    Audio → Anti-Spoofing Gate → Auto-Detection → Preprocessing →
    Feature Extraction (3 paths) → Behavioral Classification →
    Context Reasoning → LLM Translation (streaming) →
    Health + Emotion + Breed + Wellbeing enrichment
    """

    def __init__(self, device: str = None, groq_api_key: str = None):
        self.device = device or INFERENCE_CONFIG.device
        self.groq_api_key = groq_api_key
        self.call_history: List[Dict] = []
        self._models_loaded = False
        self._modules_loaded = False

        # Initialize knowledge base
        self.kb = EthologyKnowledgeBase(use_chroma=False)
        self.kb.load_seed_data()
        logger.info(f"Knowledge base loaded: {self.kb.size} entries")

    def load_models(self):
        """Load core AI models (lazy loading for fast startup)."""
        if self._models_loaded:
            return

        logger.info("Loading core models...")
        dev = self.device

        # Path A: Acoustic encoder
        self.encoder = AnimalAcousticEncoder(embed_dim=768, projection_dim=256)
        self.encoder.to(dev).eval()

        # Path B: Spectrogram CNN
        self.cnn = SpectrogramCNN(output_dim=256)
        self.cnn.to(dev).eval()

        # Stage 3: Behavioral classifier
        self.classifier = BehavioralClassifier(input_dim=522, n_cls=12, device=dev)
        self.classifier.to(dev).eval()

        # Stage 6: Sequence transformer
        self.seq_model = CallSequenceTransformer(embed_dim=276, ctx_dim=128)
        self.seq_model.to(dev).eval()

        self._models_loaded = True
        logger.info("Core models loaded successfully")

    def load_modules(self):
        """Load SRS v1.1 intelligence modules (lazy loading)."""
        if self._modules_loaded:
            return

        logger.info("Loading intelligence modules...")
        dev = self.device

        # Stage 0: Anti-Spoofing
        self.anti_spoof = AntiSpoofingGatekeeper(device=dev)

        # Module 05: Animal Detection
        self.animal_detector = AnimalDetectionPipeline(device=dev)

        # Module 01: Health Monitor
        self.health_monitor = HealthMonitor(device=dev)

        # Module 02: Breed Identifier
        self.breed_identifier = BreedIdentifier(device=dev)

        # Module 03: Emotional Tracker
        self.emotional_tracker = EmotionalTracker(device=dev)

        # Module 04: Wellbeing Dashboard
        self.wellbeing = WellbeingDashboard(device=dev)

        self._modules_loaded = True
        logger.info("All intelligence modules loaded")
        self._print_module_status_banner()

    def _print_module_status_banner(self):
        """Log ASCII banner displaying execution modes of all modules."""
        modes = {
            "Behavioral Classifier": "TRAINED (12-Class Neural)" if getattr(self.classifier, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Morton Ethology Rules)",
            "Anti-Spoofing Gate": "TRAINED (ECAPA-TDNN)" if getattr(self.anti_spoof, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Formant / VTL Physics)",
            "Animal Detector": "TRAINED (CNN Auto-Route)" if getattr(self.animal_detector, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Spectral & VAD)",
            "Health Monitor": "TRAINED (8-Flag Multi-Label)" if getattr(self.health_monitor, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Biomarker Rules)",
            "Emotion Tracker": "TRAINED (10-Class Neural)" if getattr(self.emotional_tracker, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Morton's Rules)",
            "Breed Identifier": "TRAINED (Acoustic + Visual)" if getattr(self.breed_identifier, 'has_trained_weights', False) else "ACOUSTIC HEURISTIC (Profile Distance)",
        }
        logger.info("=" * 64)
        logger.info("🐾 AnimalVox AI Intelligence Engine — Module Status:")
        for mod, mode in modes.items():
            logger.info(f"  • {mod:<22}: {mode}")
        logger.info("=" * 64)

    def get_system_status(self) -> Dict[str, Any]:
        """Return real-time weight & fallback status for all modules."""
        from models.weight_manager import get_weights_status
        return {
            "device": self.device,
            "models_loaded": self._models_loaded,
            "modules_loaded": self._modules_loaded,
            "weights_status": get_weights_status(),
        }

    def _extract_features(self, audio_clip: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from all 3 parallel paths."""
        dev = self.device

        # Path A: Audio embedding (BEATs foundation backbone)
        if getattr(self.encoder, 'has_trained_weights', False):
            audio_tensor = torch.from_numpy(audio_clip).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                audio_embed = self.encoder(audio_tensor)  # (1, 256)
        else:
            from training.feature_extractor import extract_beats_embedding
            beats_emb = extract_beats_embedding(audio_clip, sr=AUDIO_CONFIG.sample_rate, embed_dim=256)
            audio_embed = torch.from_numpy(beats_emb).float().unsqueeze(0).to(dev)

        # Path B: Spectrogram CNN
        mel = generate_mel_spectrogram(audio_clip)
        mel_img = mel_to_image(mel, target_size=(224, 224))
        mel_tensor = torch.from_numpy(mel_img).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            cnn_features = self.cnn(mel_tensor)  # (1, 256)

        # Path C: Prosodic features
        prosodic = extract_prosodic_features(audio_clip, sr=AUDIO_CONFIG.sample_rate)
        prosodic_tensor = torch.from_numpy(prosodic).float().unsqueeze(0).to(dev)

        return audio_embed, cnn_features, prosodic_tensor

    def _get_prosodic_dict(self, audio_clip: np.ndarray) -> Dict:
        """Extract prosodic features as a dictionary for module consumption."""
        import librosa
        sr = AUDIO_CONFIG.sample_rate

        # Clips shorter than the fixed five-second model window are zero-padded
        # during preprocessing.  Including that padding makes a short bark
        # appear quiet, long and low-pitched, which in turn can turn a warning
        # into an unrelated behaviour.  Keep the audible span (including the
        # natural pauses within it) for prosodic measurements.
        analysis_audio = audio_clip
        try:
            _, active_interval = librosa.effects.trim(audio_clip, top_db=45)
            active_start, active_end = active_interval
            if active_end - active_start >= int(0.1 * sr):
                analysis_audio = audio_clip[active_start:active_end]
        except Exception:
            logger.debug("Could not trim silent padding before prosody extraction", exc_info=True)

        f0, _, _ = librosa.pyin(
            analysis_audio, fmin=50, fmax=4000, sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        if len(f0_valid) < 2:
            try:
                f0_yin = librosa.yin(analysis_audio, fmin=50, fmax=2000, sr=sr)
                f0_valid = f0_yin[(f0_yin >= 50) & (f0_yin <= 2000)]
            except Exception:
                pass
        if len(f0_valid) == 0:
            f0_valid = np.array([0.0])

        # Harmonics-to-Noise Ratio proxy via spectral flatness in dB
        sf = librosa.feature.spectral_flatness(y=analysis_audio)[0]
        mean_sf = float(np.mean(sf)) + 1e-8
        hnr_db = float(-10.0 * np.log10(mean_sf))
        hnr_db = float(np.clip(hnr_db, 0.0, 35.0))

        zcr = librosa.feature.zero_crossing_rate(analysis_audio)[0]
        rms = librosa.feature.rms(y=analysis_audio)[0]
        centroid = librosa.feature.spectral_centroid(y=analysis_audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=analysis_audio, sr=sr)[0]

        # Call rate: onsets per second
        try:
            onset_frames = librosa.onset.onset_detect(y=analysis_audio, sr=sr)
            duration_s = max(len(analysis_audio) / sr, 0.01)
            call_rate_val = float(len(onset_frames) / duration_s)
        except Exception:
            call_rate_val = 1.0

        # F0 slope: direction of pitch trajectory
        if len(f0_valid) >= 2:
            x = np.arange(len(f0_valid))
            f0_slope = float(np.polyfit(x, f0_valid, 1)[0])
        else:
            f0_slope = 0.0

        try:
            tempo, _ = librosa.beat.beat_track(y=analysis_audio, sr=sr)
            tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 100.0
        except Exception:
            tempo_val = 100.0

        return {
            "mean_f0": float(np.mean(f0_valid)),
            "f0_range": float(np.ptp(f0_valid)),
            "f0_slope": f0_slope,
            "mean_hnr": hnr_db,
            "call_rate": call_rate_val,
            "mean_rms": float(np.mean(rms)) if len(rms) > 0 else 0.0,
            "spectral_centroid": float(np.mean(centroid)) if len(centroid) > 0 else 0.0,
            "mean_call_duration": float(len(analysis_audio) / sr),
            "mean_zcr": float(np.mean(zcr)) if len(zcr) > 0 else 0.0,
            "mean_spectral_rolloff": float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0,
            "tempo": tempo_val,
        }

    def _classify(
        self,
        audio_embed: torch.Tensor,
        cnn_features: torch.Tensor,
        prosodic: torch.Tensor,
        prosodic_dict: Optional[Dict[str, float]] = None,
        species: str = "dogs",
        audio_clip: Optional[np.ndarray] = None
    ) -> Dict:
        """Run behavioral classification on fused features (Neural or Ethological Fallback)."""
        fused = torch.cat([audio_embed, cnn_features, prosodic], dim=-1)

        if hasattr(self.classifier, "classify") and prosodic_dict is not None:
            labels_np, intensity_val, urgency_val = self.classifier.classify(
                fused_tensor=fused,
                prosodic=prosodic_dict,
                species=species,
                audio_clip=audio_clip
            )
        else:
            with torch.no_grad():
                labels, intensity, urgency = self.classifier(fused)
            labels_np = labels.cpu().numpy()[0]
            intensity_val = float(intensity.cpu().numpy()[0][0])
            urgency_val = float(urgency.cpu().numpy()[0][0])

        # Get top behaviors (threshold 0.3)
        detected = []
        for i, score in enumerate(labels_np):
            if score > 0.3:
                detected.append({
                    "id": f"B{i:02d}",
                    "name": BEHAVIOR_LABELS[i],
                    "score": float(score)
                })
        detected.sort(key=lambda x: x["score"], reverse=True)

        if not detected:
            # Always return at least the top prediction
            top_idx = int(np.argmax(labels_np))
            detected.append({
                "id": f"B{top_idx:02d}",
                "name": BEHAVIOR_LABELS[top_idx],
                "score": float(labels_np[top_idx])
            })

        return {
            "behaviors": detected,
            "intensity": intensity_val,
            "urgency": urgency_val,
            "all_scores": labels_np.tolist(),
            "audio_embed": audio_embed,
            "fused_embed": fused
        }

    def _get_sequence_context(self, classification: Dict, species: str) -> Tuple[str, str]:
        """Build context from call history using sequence transformer."""
        # Add current call to history
        self.call_history.append(classification)
        if len(self.call_history) > 5:
            self.call_history = self.call_history[-5:]

        if len(self.call_history) < 2:
            return "First call in session — no prior context", "stable"

        # Build sequence embedding
        seq_embeds = []
        for call in self.call_history:
            scores = torch.tensor(call["all_scores"], dtype=torch.float32)
            intensity = torch.tensor([call["intensity"]])
            urgency = torch.tensor([call["urgency"]])
            audio_e = call["audio_embed"].squeeze(0).cpu() if isinstance(call["audio_embed"], torch.Tensor) else torch.zeros(256)
            species_oh = torch.zeros(NUM_SPECIES_GROUPS)
            sp_id = SPECIES_GROUPS.get(species, {}).get("id", 0)
            species_oh[sp_id] = 1.0
            embed = torch.cat([scores, intensity, urgency, audio_e, species_oh])
            seq_embeds.append(embed)

        # Pad if needed
        while len(seq_embeds) < 5:
            seq_embeds.insert(0, torch.zeros_like(seq_embeds[0]))

        seq_tensor = torch.stack(seq_embeds).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ctx, escalation_logits = self.seq_model(seq_tensor)

        esc_probs = torch.softmax(escalation_logits, dim=-1).cpu().numpy()[0]
        esc_labels = ["de-escalation", "stable", "escalation"]
        esc_pattern = esc_labels[int(np.argmax(esc_probs))]

        ctx_desc = f"Call {len(self.call_history)} of session. Pattern: {esc_pattern}"
        return ctx_desc, esc_pattern

    async def _generate_translation(self, species: str, classification: Dict,
                                     seq_context: str, escalation: str,
                                     emotion: Optional[str] = None,
                                     breed: Optional[str] = None,
                                     species_detail: Optional[str] = None,
                                     acoustic_signature: Optional[str] = None) -> Dict:
        """Generate natural language translation via Groq API or bioacoustic engine."""
        behavior_names = [classification["behaviors"][0]["name"]]
        primary_behavior = behavior_names[0]
        translation_emotion = self._select_translation_emotion(primary_behavior, emotion)
        ethology_ctx = self.kb.get_context_for_llm(species, behavior_names)

        detail = species_detail or breed or "unspecified"
        acoustic_sig = acoustic_signature or "Natural bioacoustic vocalization"

        prompt = build_prompt(
            species=species,
            species_detail=detail,
            behavior_labels=behavior_names,
            intensity=classification["intensity"],
            urgency=classification["urgency"],
            sequence_context=seq_context,
            escalation=escalation,
            ethology_context=ethology_ctx,
            emotional_state=translation_emotion,
            acoustic_signature=acoustic_sig
        )

        # Try Groq API first
        if self.groq_api_key:
            try:
                res = await self._call_groq(prompt)
                if self._is_grounded_translation(res, primary_behavior):
                    # Keep the LLM only for its concise wording.  Structured
                    # labels and context come from the local, auditable
                    # ethology map rather than from creative model prose.
                    fallback = self._fallback_translation(
                        species=species,
                        classification=classification,
                        emotion=translation_emotion,
                        breed=breed,
                    )
                    fallback["translation"] = res["translation"].strip()
                    return fallback
                logger.warning("Discarding unsupported LLM translation; using ethology template")
            except Exception as e:
                logger.warning(f"Groq API failed: {e}, using bioacoustic translation engine")

        # Fallback: bioacoustic ethological translation
        return self._fallback_translation(
            species=species, classification=classification,
            emotion=translation_emotion, breed=breed
        )

    def _select_translation_emotion(self, behavior: str, detected_emotion: Optional[str]) -> str:
        """Use a modelled emotion only when it agrees with the primary call type."""
        behavior = (behavior or "").lower()
        detected = (detected_emotion or "").lower()
        compatible_terms = {
            "alarm_aerial": ("stress", "fear", "alert", "vigil"),
            "alarm_ground": ("stress", "fear", "alert", "vigil", "territor", "frustrat"),
            "territorial_warning": ("territor", "stress", "frustrat", "alert", "vigil"),
            "aggression_warning": ("stress", "frustrat", "territor", "alert", "vigil"),
            "distress_call": ("stress", "fear", "pain"),
            "play_vocalization": ("play", "excit"),
            "social_bonding": ("bond", "content"),
            "contact_call": ("bond", "lonel"),
            "feeding_call": ("hunger", "excit"),
        }
        if detected and any(term in detected for term in compatible_terms.get(behavior, ())):
            return detected_emotion
        return self._infer_emotion(behavior, 0.0)

    @staticmethod
    def _is_grounded_translation(response: Optional[Dict], behavior: str) -> bool:
        """Reject creative LLM prose that is unsupported by one call label."""
        if not isinstance(response, dict):
            return False

        text = str(response.get("translation", "")).strip()
        words = re.findall(r"[A-Za-z']+", text)
        if not 4 <= len(words) <= 40 or len(re.findall(r"[.!?]", text)) > 2:
            return False

        lowered = text.lower()
        # A result should state a probable communicative meaning, not narrate
        # the audio analysis (the failure mode in the original screenshot).
        analysis_words = (
            "bark", "barking", "growl", "growling", "vocalization",
            "high-pitched", "low-pitched", "pitch", "shift", "first",
            "then i", "after i", "call type",
        )
        if any(term in lowered for term in analysis_words):
            return False

        incompatible_terms = {
            "alarm_aerial": ("play", "lonely", "food", "mate"),
            "alarm_ground": ("play", "lonely", "food", "mate", "friend"),
            "territorial_warning": ("play", "lonely", "food", "mate", "friend"),
            "aggression_warning": ("play", "lonely", "food", "mate", "friend"),
            "distress_call": ("play", "food", "mate"),
        }
        return not any(term in lowered for term in incompatible_terms.get(behavior, ()))

    async def _generate_translation_stream(self, species: str, classification: Dict,
                                            seq_context: str, escalation: str) -> AsyncGenerator[str, None]:
        """
        Stream translation tokens via Groq SSE for real-time display.
        
        Yields individual tokens as they arrive from the LLM,
        enabling word-by-word karaoke-style display.
        """
        behavior_names = [classification["behaviors"][0]["name"]]
        ethology_ctx = self.kb.get_context_for_llm(species, behavior_names)

        prompt = build_prompt(
            species=species,
            species_detail=None,
            behavior_labels=behavior_names,
            intensity=classification["intensity"],
            urgency=classification["urgency"],
            sequence_context=seq_context,
            escalation=escalation,
            ethology_context=ethology_ctx
        )

        if not self.groq_api_key:
            # Fallback: yield full translation at once
            result = self._fallback_translation(species, classification)
            for word in result.get("translation", "").split():
                yield word + " "
            return

        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.groq_api_key)
            stream = await client.chat.completions.create(
                model=INFERENCE_CONFIG.groq_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=INFERENCE_CONFIG.groq_temperature,
                max_tokens=INFERENCE_CONFIG.groq_max_tokens,
                stream=True  # Enable streaming
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except Exception as e:
            logger.warning(f"Groq streaming failed: {e}, using fallback")
            result = self._fallback_translation(species, classification)
            for word in result.get("translation", "").split():
                yield word + " "

    async def _call_groq(self, prompt: str) -> Dict:
        """Call Groq API for LLM inference."""
        from groq import AsyncGroq

        client = AsyncGroq(api_key=self.groq_api_key)
        response = await client.chat.completions.create(
            model=INFERENCE_CONFIG.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=INFERENCE_CONFIG.groq_temperature,
            max_tokens=INFERENCE_CONFIG.groq_max_tokens,
            response_format={"type": "json_object"}
        )

        text = response.choices[0].message.content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"translation": text, "vocalization_type": "unknown",
                    "emotion": "unknown", "behavior_context": "", "confidence": "low"}

    def _fallback_translation(self, species: str, classification: Dict,
                               emotion: Optional[str] = None,
                               breed: Optional[str] = None) -> Dict:
        """Bioacoustic ethology-grounded human translation when LLM is unavailable."""
        top_behavior = classification["behaviors"][0]
        behavior_name = top_behavior["name"]
        intensity = classification.get("intensity", 0.5)
        urgency = classification.get("urgency", 0.5)

        # Use comprehensive Bioacoustic Human Translation Engine
        return generate_human_translation(
            species=species,
            behavior=behavior_name,
            intensity=intensity,
            emotion=emotion,
            breed=breed,
            urgency=urgency
        )

    def _infer_emotion(self, behavior: str, intensity: float) -> str:
        """Map behavior to emotional state."""
        emotion_map = {
            "alarm_aerial": "fear / urgent warning",
            "alarm_ground": "alert / protective",
            "territorial_warning": "assertive / dominant",
            "mating_call": "confident / hopeful",
            "contact_call": "social / seeking connection",
            "distress_call": "distressed / fearful",
            "feeding_call": "excited / sharing",
            "play_vocalization": "joyful / excited",
            "submission_signal": "submissive / appeasing",
            "aggression_warning": "aggressive / threatening",
            "social_bonding": "content / affectionate",
            "navigation_echolocation": "focused / exploratory",
        }
        return emotion_map.get(behavior, "unknown")

    def _select_best_clip(self, clips: List[np.ndarray]) -> np.ndarray:
        """Select the 5s clip with the strongest animal vocalization (via PANNs CNN14), or highest RMS."""
        if len(clips) == 1:
            return clips[0]
        try:
            from models.audio_tagger import get_tagger
            tagger = get_tagger(device=self.device)
            if tagger.available:
                best_clip = None
                best_score = -1.0
                for c in clips[:8]:
                    res = tagger.tag(c, sr=AUDIO_CONFIG.sample_rate)
                    mus = res.clip_groups.get("music", 0.0)
                    if mus >= 0.60 and mus > res.any_animal_prob * 1.15:
                        continue
                    score = res.any_animal_prob - 0.15 * res.clip_groups.get("speech", 0.0) - 0.25 * mus
                    if res.any_animal_prob >= 0.20 and score > best_score:
                        best_score = score
                        best_clip = c
                if best_clip is not None:
                    return best_clip
        except Exception as e:
            logger.debug(f"Clip selection fallback to RMS: {e}")
        return max(clips, key=lambda c: float(np.sqrt(np.mean(c**2))))

    async def translate(self, audio_path: str = None, audio_bytes: bytes = None,
                        audio_format: str = "wav", species: str = None,
                        pet_id: str = None, photo_features: torch.Tensor = None,
                        enable_modules: bool = True, filename: str = None) -> Dict:
        """
        Main entry point: translate animal audio to human language.

        Enhanced with anti-spoofing, auto-detection, and all 5 modules.

        Args:
            audio_path: Path to audio file
            audio_bytes: Raw audio bytes
            audio_format: Audio format string
            species: Species key (None = auto-detect via Module 05)
            pet_id: Optional pet profile ID for baseline tracking
            photo_features: Optional visual features for breed ID
            enable_modules: Run enrichment modules (health, emotion, etc.)
            filename: Original upload/recording filename for hints

        Returns dict with: translation, vocalization_type, emotion,
        behavior_context, confidence, behaviors, intensity, urgency,
        timing, and enriched module outputs.
        """
        start_time = time.time()
        self.load_models()
        if enable_modules:
            self.load_modules()

        # Stage 1: Preprocessing
        if audio_path:
            clips = preprocess_audio(audio_path)
            if not filename:
                filename = Path(audio_path).name
        elif audio_bytes:
            clips = preprocess_audio_bytes(audio_bytes, format=audio_format)
        else:
            raise ValueError("Provide audio_path or audio_bytes")

        if not clips:
            return {"error": "no_animal_detected", "animal_detected": False, "translation": "", "message": "No audio content detected"}

        # Select the clip with strongest animal vocalization (or highest RMS)
        clip = self._select_best_clip(clips)

        # ── Stage 0.1: Music / Song / BGM and Non-Animal SFX Filter ────
        from models.music_detector import detect_music_or_sfx
        full_dur = None
        if audio_path and os.path.exists(audio_path):
            try:
                full_dur = librosa.get_duration(path=audio_path)
            except Exception:
                pass
        music_or_sfx, filter_reason = detect_music_or_sfx(
            clip, sr=AUDIO_CONFIG.sample_rate, filename=filename, full_duration=full_dur
        )
        if music_or_sfx == "music_detected":
            elapsed = time.time() - start_time
            return {
                "error": "music_detected",
                "music_detected": True,
                "is_music": True,
                "translation": "",
                "message": "🎵 Music / Songs / BGMs not allowed for translation. AnimalVox AI is designed for real animal vocalizations.",
                "rejection_reason": filter_reason or "Music, songs, and background tracks cannot be translated.",
                "processing_time_ms": int(elapsed * 1000),
            }
        elif music_or_sfx == "no_animal_detected":
            elapsed = time.time() - start_time
            return {
                "error": "no_animal_detected",
                "animal_detected": False,
                "no_animal": True,
                "translation": "",
                "message": "❓ No animal vocalization detected. This audio appears to be a sound effect or synthetic audio.",
                "rejection_reason": filter_reason or "No animal detected. Sound effects, meme audio, or ambient noise cannot be translated.",
                "processing_time_ms": int(elapsed * 1000),
            }

        # ── Stage 0.2: Anti-Spoofing & Mixed Audio Verification ──
        human_speech_removed = False
        spoof_result = None
        if enable_modules:
            # 1. ALWAYS check anti-spoofing on the ORIGINAL clip first
            #    This prevents imitations from being split by VAD into
            #    "human" + "animal-like" segments where the animal-like
            #    segments bypass the spoof check.
            full_spoof = self.anti_spoof.check(clip, sr=AUDIO_CONFIG.sample_rate, filename=filename)
            spoof_result = full_spoof

            # 2. Handle human detection on the original clip
            if full_spoof.is_human:
                elapsed = time.time() - start_time
                if getattr(full_spoof, "is_conversational_speech", False):
                    return {
                        "error": "no_animal_detected",
                        "animal_detected": False,
                        "translation": "",
                        "message": "❓ No animal vocalization detected. The recording contains human speech, but no animal sounds to translate.",
                        "rejection_reason": full_spoof.rejection_reason or "Human conversational speech detected without animal vocalization.",
                        "confidence": full_spoof.confidence,
                        "processing_time_ms": int(elapsed * 1000),
                    }
                else:
                    return {
                        "error": "human_detected",
                        "human_detected": True,
                        "is_human": True,
                        "translation": "",
                        "message": "🚫 This doesn't sound like an animal. AnimalVox AI only translates real animal vocalizations.",
                        "human_probability": full_spoof.human_probability,
                        "rejection_reason": full_spoof.rejection_reason or "Human vocal imitation detected.",
                        "confidence": full_spoof.confidence,
                        "processing_time_ms": int(elapsed * 1000),
                    }

            # 3. Handle uncertain result — prompt re-record
            if getattr(full_spoof, "is_uncertain", False):
                elapsed = time.time() - start_time
                return {
                    "error": "uncertain_audio",
                    "uncertain": True,
                    "translation": "",
                    "message": "🔍 We couldn't confidently identify this as an animal sound. Please try recording again with a clearer sample.",
                    "rejection_reason": full_spoof.rejection_reason or "Ambiguous audio — could not confirm animal vocalization.",
                    "confidence": full_spoof.confidence,
                    "processing_time_ms": int(elapsed * 1000),
                }

            # 4. If clip passed anti-spoof AND has mixed human speech,
            #    isolate the animal vocalization segments via PANNs CNN14.
            cleaned_clip, had_human_speech = self.animal_detector.vad.remove_human_speech(clip)
            if had_human_speech:
                nz = cleaned_clip[np.abs(cleaned_clip) > 1e-4]
                active_rms = float(np.sqrt(np.mean(nz ** 2))) if len(nz) > 0 else 0.0
                if active_rms >= 0.015:
                    cleaned_det = self.animal_detector.detect(cleaned_clip, filename=filename)
                    if cleaned_det.has_animal:
                        clip = cleaned_clip
                        human_speech_removed = True
                        logger.info("Mixed audio: isolated animal vocalization and removed human talking.")

        # Stage 2: Feature extraction (3 parallel paths)
        audio_embed, cnn_features, prosodic_tensor = self._extract_features(clip)
        prosodic_dict = self._get_prosodic_dict(clip)
        fused_266 = torch.cat([audio_embed, prosodic_tensor], dim=-1)

        # ── Module 05: Auto-Detection / Animal Presence Verification ──
        detection_result = None
        if enable_modules:
            detection_result = self.animal_detector.detect(clip, audio_embed=fused_266, filename=filename)

            if detection_result.unsupported_species:
                if species is not None and species in SPECIES_GROUPS:
                    detection_result.unsupported_species = False
                    detection_result.unsupported_animal = ""
                else:
                    elapsed = time.time() - start_time
                    animal_name = detection_result.unsupported_animal or "this animal"
                    return {
                        "error": "unsupported_species",
                        "species_supported": False,
                        "detected_animal": animal_name,
                        "translation": "",
                        "message": f"Species Not Supported Yet: We detected a {animal_name} vocalization, which isn't in our translation database yet.",
                        "rejection_reason": f"The detected species ({animal_name}) is not currently supported in AnimalVox AI. Supported species: Dogs, Cats, Birds, Dolphins, Whales, Frogs.",
                        "processing_time_ms": int(elapsed * 1000),
                    }

            if not detection_result.has_animal:
                elapsed = time.time() - start_time
                return {
                    "error": "no_animal_detected",
                    "animal_detected": False,
                    "translation": "",
                    "message": "No animal vocalization detected in this audio.",
                    "soundscape": detection_result.soundscape,
                    "rejection_reason": "No animal vocalization was found in this audio. AnimalVox AI only translates genuine animal sounds.",
                    "processing_time_ms": int(elapsed * 1000),
                }

            # Use top detection as species if species was not specified
            if species is None:
                if detection_result.species_detections:
                    species = detection_result.species_detections[0]["species"]
                else:
                    species = "dogs"  # Default fallback

        species = species or "dogs"

        # Stage 3: Behavioral classification
        classification = self._classify(
            audio_embed, cnn_features, prosodic_tensor,
            prosodic_dict=prosodic_dict, species=species, audio_clip=clip
        )

        # Stage 4: Context reasoning
        seq_context, escalation = self._get_sequence_context(classification, species)

        # ── Module Enrichment (parallel) ───────────────────────
        health_report = None
        breed_report = None
        emotion_report = None
        wellbeing_report = None

        if enable_modules:
            # Module 01: Health Monitoring (266-dim feature vector)
            try:
                health_report = self.health_monitor.assess(
                    audio=clip, fused_embedding=fused_266,
                    prosodic_features=prosodic_dict, species=species,
                    pet_id=pet_id
                )
            except Exception as e:
                logger.warning(f"Health monitor error: {e}")

            # Module 02: Breed Identification (dogs/cats only, 266-dim)
            try:
                breed_report = self.breed_identifier.identify(
                    fused_embedding=fused_266,
                    prosodic_features=prosodic_dict,
                    species=species,
                    visual_features=photo_features
                )
            except Exception as e:
                logger.warning(f"Breed identifier error: {e}")

            # Module 03: Emotional Tracking (266-dim with Multi-Bout Sequence Smoothing)
            try:
                emotion_report = self.emotional_tracker.assess(
                    fused_embedding=fused_266,
                    prosodic_features=prosodic_dict,
                    species=species,
                    pet_id=pet_id,
                    sequence_history=self.call_history,
                    escalation_pattern=escalation
                )
            except Exception as e:
                logger.warning(f"Emotional tracker error: {e}")

            # A wellbeing score needs a pet profile and history.  Presenting a
            # score for a single anonymous bark looks authoritative but has no
            # meaningful baseline, so only compute it for a named pet.
            if pet_id:
                try:
                    diary_entries = self.emotional_tracker.diary.get_entries(pet_id, days=7)
                    soundscape = detection_result.soundscape if detection_result else ""
                    wellbeing_report = self.wellbeing.assess(
                        health_report=health_report,
                        emotion_report=emotion_report,
                        prosodic_features=prosodic_dict,
                        soundscape=soundscape,
                        diary_entries=diary_entries,
                        pet_id=pet_id
                    )
                except Exception as e:
                    logger.warning(f"Wellbeing dashboard error: {e}")

        # Stage 5: NLG Translation (now enriched with emotional, acoustic, and species context)
        primary_behavior = classification["behaviors"][0]
        model_emotion_name = emotion_report.primary_emotion.state_name if emotion_report else None
        display_emotion = self._select_translation_emotion(
            primary_behavior["name"], model_emotion_name
        )
        breed_name = breed_report.top_predictions[0].breed_name if (breed_report and breed_report.top_predictions) else None

        # Resolve nuanced species detail
        species_detail = breed_name
        fn_lower = (filename or "").lower()
        if not species_detail:
            if species == "whales":
                if any(k in fn_lower for k in ["blue whale", "blue_whale"]):
                    species_detail = "Blue Whale"
                elif any(k in fn_lower for k in ["humpback"]):
                    species_detail = "Humpback Whale"
                elif any(k in fn_lower for k in ["killer whale", "orca"]):
                    species_detail = "Orca / Killer Whale"
                elif any(k in fn_lower for k in ["beluga"]):
                    species_detail = "Beluga Whale"
                elif any(k in fn_lower for k in ["fin whale"]):
                    species_detail = "Fin Whale"
                elif prosodic_dict.get("mean_f0", 100) < 65:
                    species_detail = "Blue / Fin Whale (Infrasonic)"
                else:
                    species_detail = "Humpback Whale"
            elif species == "birds":
                for b_name in ["robin", "sparrow", "canary", "parrot", "crow", "hawk", "eagle", "owl", "finch", "warbler", "songbird"]:
                    if b_name in fn_lower:
                        species_detail = b_name.capitalize()
                        break
                if not species_detail:
                    species_detail = "Songbird" if prosodic_dict.get("f0_range", 0) > 200 else "Wild Avian"
            elif species == "frogs":
                for f_name in ["bullfrog", "treefrog", "tree frog", "toad", "poison dart"]:
                    if f_name in fn_lower:
                        species_detail = f_name.capitalize()
                        break
                if not species_detail:
                    species_detail = "Chorus Frog"
            elif species == "dolphins":
                species_detail = "Bottlenose Dolphin"

        # Generate acoustic description for translation personalization
        f0_val = prosodic_dict.get("mean_f0", 0)
        f0_rng = prosodic_dict.get("f0_range", 0)
        rms_val = prosodic_dict.get("mean_rms", 0)
        hnr_val = prosodic_dict.get("mean_hnr", 0)
        rate_val = prosodic_dict.get("call_rate", 1.0)
        if species == "whales":
            if f0_val < 65:
                acoustic_sig = f"Deep infrasonic low-frequency moan (~{f0_val:.0f} Hz), steady trans-oceanic resonance, HNR {hnr_val:.1f} dB"
            elif f0_rng > 65:
                acoustic_sig = f"Melodic frequency-modulated song phrase (pitch variation ~{f0_rng:.0f} Hz, F0 ~{f0_val:.0f} Hz), rich harmonic theme"
            else:
                acoustic_sig = f"Sustained underwater cetacean vocalization (~{f0_val:.0f} Hz), volume RMS {rms_val:.2f}"
        elif species == "birds":
            acoustic_sig = f"Avian call/song with F0 ~{f0_val:.0f} Hz, modulation range ~{f0_rng:.0f} Hz, rate ~{rate_val:.1f} calls/sec"
        elif species == "dogs":
            acoustic_sig = f"Canine vocalization (~{f0_val:.0f} Hz), burst rate {rate_val:.1f} onsets/sec, acoustic power {rms_val:.2f}"
        elif species == "cats":
            acoustic_sig = f"Feline vocalization with pitch ~{f0_val:.0f} Hz, modulation ~{f0_rng:.0f} Hz"
        else:
            acoustic_sig = f"Acoustic pitch ~{f0_val:.0f} Hz, call rate {rate_val:.1f}/sec"

        translation = await self._generate_translation(
            species=species, classification=classification,
            seq_context=seq_context, escalation=escalation,
            emotion=model_emotion_name, breed=breed_name,
            species_detail=species_detail, acoustic_signature=acoustic_sig
        )

        elapsed = time.time() - start_time

        # ── Build response ────────────────────────────────────
        result = {
            **translation,
            "species": species,
            "species_detail": species_detail,
            "behavior": classification["behaviors"][0]["name"] if classification.get("behaviors") else None,
            "emotion": display_emotion,
            "species_emoji": SPECIES_GROUPS.get(species, {}).get("emoji", "🔊"),
            "behaviors": classification["behaviors"],
            "intensity": classification["intensity"],
            "urgency": classification["urgency"],
            "escalation_pattern": escalation,
            "human_speech_removed": human_speech_removed,
            "processing_time_ms": int(elapsed * 1000),
            "mel_spectrogram": generate_mel_spectrogram(clip).tolist(),
        }

        # Anti-spoofing info
        if spoof_result:
            result["anti_spoofing"] = {
                "is_human": spoof_result.is_human,
                "human_probability": spoof_result.human_probability,
                "confidence": spoof_result.confidence,
            }

        # Auto-detection info
        if detection_result:
            result["auto_detection"] = {
                "soundscape": detection_result.soundscape,
                "species_detections": detection_result.species_detections,
                "num_individuals": detection_result.num_individuals,
                "human_speech_removed": detection_result.human_speech_removed,
                "routing_confident": detection_result.routing_confident,
                "detector_mode": detection_result.detector_mode,
            }

        # Module 01: Health
        if health_report:
            result["health"] = {
                "overall_status": health_report.overall_status,
                "alerts": [
                    {
                        "flag_id": a.flag_id,
                        "flag_name": a.flag_name,
                        "severity": a.severity,
                        "confidence": a.confidence,
                        "veterinary_context": a.veterinary_context,
                    }
                    for a in health_report.alerts
                ],
                "respiratory_normal": health_report.respiratory_normal,
                "vocal_pattern_change": health_report.vocal_pattern_change,
                "baseline_established": health_report.baseline_established,
                "disclaimer": MEDICAL_DISCLAIMER,
            }

        # Module 02: Breed
        if breed_report and breed_report.top_predictions:
            result["breed"] = {
                "acoustic_group": breed_report.acoustic_group,
                "acoustic_confidence": breed_report.acoustic_confidence,
                "top_predictions": [
                    {
                        "breed": p.breed_name,
                        "group": p.group_name,
                        "confidence": p.confidence,
                        "source": p.source,
                    }
                    for p in breed_report.top_predictions
                ],
                "fusion_method": breed_report.fusion_method,
            }

        # Module 03: Emotion
        if emotion_report:
            result["emotion_data"] = {
                "primary": {
                    "state_id": "behavior_aligned",
                    "name": display_emotion,
                    "confidence": primary_behavior["score"],
                    "context": "Aligned with the primary vocal behaviour for this clip.",
                },
                "secondary": [],
                "stress_level": emotion_report.stress_level,
                "mood_stability": emotion_report.mood_stability,
                "model_primary": {
                    "state_id": emotion_report.primary_emotion.state_id,
                    "name": emotion_report.primary_emotion.state_name,
                    "confidence": emotion_report.primary_emotion.confidence,
                },
            }

        # Module 04: Wellbeing
        if wellbeing_report:
            result["wellbeing"] = {
                "composite_score": wellbeing_report.composite_score,
                "composite_label": wellbeing_report.composite_label,
                "trend": wellbeing_report.trend,
                "dimensions": {
                    k: {
                        "name": v.name,
                        "score": v.score,
                        "weight": v.weight,
                        "trend": v.trend,
                    }
                    for k, v in wellbeing_report.dimensions.items()
                },
                "recommendations": wellbeing_report.recommendations,
            }

        return result

    async def translate_stream(self, audio_path: str = None,
                               audio_bytes: bytes = None,
                               audio_format: str = "wav",
                               species: str = None,
                               pet_id: str = None,
                               filename: str = None) -> AsyncGenerator[Dict, None]:
        """
        Streaming translation: yields word-by-word tokens for real-time display.

        First yields metadata (anti-spoofing, detection, species).
        Then yields individual translation tokens for karaoke display.
        Finally yields enrichment module data (health, emotion, wellbeing).
        """
        start_time = time.time()
        self.load_models()
        self.load_modules()

        # Preprocessing
        if audio_path:
            clips = preprocess_audio(audio_path)
            if not filename:
                filename = Path(audio_path).name
        elif audio_bytes:
            clips = preprocess_audio_bytes(audio_bytes, format=audio_format)
        else:
            yield {"type": "error", "message": "No audio provided"}
            return

        if not clips:
            yield {"type": "no_animal", "message": "No audio content detected"}
            return

        # Select the clip with strongest animal vocalization (or highest RMS)
        clip = self._select_best_clip(clips)

        # ── Stage 0.1: Music / Song / BGM and Non-Animal SFX Filter ────
        from models.music_detector import detect_music_or_sfx
        full_dur = None
        if audio_path and os.path.exists(audio_path):
            try:
                full_dur = librosa.get_duration(path=audio_path)
            except Exception:
                pass
        music_or_sfx, filter_reason = detect_music_or_sfx(
            clip, sr=AUDIO_CONFIG.sample_rate, filename=filename, full_duration=full_dur
        )
        if music_or_sfx == "music_detected":
            yield {
                "type": "music_detected",
                "music_detected": True,
                "is_music": True,
                "message": "🎵 Music / Songs / BGMs not allowed for translation. AnimalVox AI is designed for real animal vocalizations.",
                "rejection_reason": filter_reason or "Music, songs, and background tracks cannot be translated.",
            }
            return
        elif music_or_sfx == "no_animal_detected":
            yield {
                "type": "no_animal_detected",
                "animal_detected": False,
                "no_animal": True,
                "message": "❓ No animal vocalization detected. This audio appears to be a sound effect or synthetic audio.",
                "rejection_reason": filter_reason or "No animal detected. Sound effects, meme audio, or ambient noise cannot be translated.",
            }
            return

        # ── Stage 0.2: Anti-Spoofing & Mixed Audio Verification (Stream) ──
        human_speech_removed = False

        # Check anti-spoofing on the ORIGINAL clip first
        full_spoof = self.anti_spoof.check(clip, sr=AUDIO_CONFIG.sample_rate, filename=filename)

        if full_spoof.is_human:
            if getattr(full_spoof, "is_conversational_speech", False):
                yield {
                    "type": "no_animal_detected",
                    "animal_detected": False,
                    "no_animal": True,
                    "message": "❓ No animal vocalization detected. Human speech was detected, but no animal sounds to translate.",
                    "rejection_reason": full_spoof.rejection_reason or "Human speech detected without animal vocalization.",
                }
                return
            else:
                yield {
                    "type": "human_detected",
                    "human_detected": True,
                    "is_human": True,
                    "message": "🚫 This doesn't sound like an animal.",
                    "human_probability": full_spoof.human_probability,
                    "rejection_reason": full_spoof.rejection_reason or "Human speech or vocal imitation detected.",
                }
                return

        if getattr(full_spoof, "is_uncertain", False):
            yield {
                "type": "uncertain_audio",
                "uncertain": True,
                "message": "🔍 We couldn't confidently identify this as an animal sound. Please try recording again.",
                "rejection_reason": full_spoof.rejection_reason or "Ambiguous audio.",
            }
            return

        # If passed, isolate animal segments from background human speech via PANNs CNN14
        cleaned_clip, had_human_speech = self.animal_detector.vad.remove_human_speech(clip)
        if had_human_speech:
            nz = cleaned_clip[np.abs(cleaned_clip) > 1e-4]
            active_rms = float(np.sqrt(np.mean(nz ** 2))) if len(nz) > 0 else 0.0
            if active_rms >= 0.015:
                cleaned_det = self.animal_detector.detect(cleaned_clip, filename=filename)
                if cleaned_det.has_animal:
                    clip = cleaned_clip
                    human_speech_removed = True
                    logger.info("Mixed audio (stream): isolated animal vocalization.")

        # Feature extraction
        audio_embed, cnn_features, prosodic_tensor = self._extract_features(clip)
        prosodic_dict = self._get_prosodic_dict(clip)
        fused_266 = torch.cat([audio_embed, prosodic_tensor], dim=-1)

        # ── Module 05: Auto-Detection / Animal Presence Verification ──
        detection_result = self.animal_detector.detect(clip, audio_embed=fused_266, filename=filename)

        if detection_result.unsupported_species:
            if species is not None and species in SPECIES_GROUPS:
                detection_result.unsupported_species = False
                detection_result.unsupported_animal = ""
            else:
                animal_name = detection_result.unsupported_animal or "this animal"
                yield {
                    "type": "unsupported_species",
                    "species_supported": False,
                    "detected_animal": animal_name,
                    "message": f"Species Not Supported Yet: {animal_name} is not in our translation database.",
                    "rejection_reason": f"Species {animal_name} is not supported yet.",
                }
                return

        if not detection_result.has_animal:
            yield {
                "type": "no_animal",
                "message": "No animal vocalization detected.",
                "rejection_reason": "No animal vocalization was found in this audio."
            }
            return

        if species is None:
            species = detection_result.species_detections[0]["species"] if detection_result.species_detections else "dogs"

        species = species or "dogs"

        # Yield metadata
        yield {
            "type": "metadata",
            "species": species,
            "emoji": SPECIES_GROUPS.get(species, {}).get("emoji", "🔊"),
            "soundscape": detection_result.soundscape if detection_result else "unknown",
        }

        # Feature extraction + classification
        audio_embed, cnn_features, prosodic_tensor = self._extract_features(clip)
        prosodic_dict = self._get_prosodic_dict(clip)
        classification = self._classify(
            audio_embed, cnn_features, prosodic_tensor,
            prosodic_dict=prosodic_dict, species=species, audio_clip=clip
        )
        seq_context, escalation = self._get_sequence_context(classification, species)

        # Stream translation tokens
        yield {"type": "translation_start"}

        phrase_buffer = []
        async for token in self._generate_translation_stream(
            species, classification, seq_context, escalation
        ):
            yield {"type": "token", "content": token}

            # Phrase buffering for TTS
            phrase_buffer.append(token)
            buffer_text = "".join(phrase_buffer)
            words = buffer_text.split()

            # Flush phrase when punctuation or max words reached
            should_flush = (
                len(words) >= STREAMING_CONFIG.phrase_buffer_max_words or
                any(c in token for c in STREAMING_CONFIG.phrase_punctuation_triggers)
            )
            if should_flush and words:
                yield {"type": "phrase", "text": buffer_text.strip()}
                phrase_buffer = []

        # Flush remaining buffer
        if phrase_buffer:
            yield {"type": "phrase", "text": "".join(phrase_buffer).strip()}

        yield {"type": "translation_end"}

        # Module enrichment
        fused_embedding = classification["fused_embed"]

        try:
            health_report = self.health_monitor.assess(
                audio=clip, fused_embedding=fused_embedding,
                prosodic_features=prosodic_dict, species=species, pet_id=pet_id
            )
            yield {
                "type": "health",
                "overall_status": health_report.overall_status,
                "alerts": [{"flag_id": a.flag_id, "flag_name": a.flag_name,
                           "severity": a.severity} for a in health_report.alerts],
                "disclaimer": MEDICAL_DISCLAIMER,
            }
        except Exception as e:
            logger.warning(f"Health module error during stream: {e}")

        try:
            emotion_report = self.emotional_tracker.assess(
                fused_embedding=fused_embedding,
                prosodic_features=prosodic_dict,
                species=species, pet_id=pet_id
            )
            yield {
                "type": "emotion",
                "primary": emotion_report.primary_emotion.state_name,
                "confidence": emotion_report.primary_emotion.confidence,
                "stress_level": emotion_report.stress_level,
                "mood_stability": emotion_report.mood_stability,
            }
        except Exception as e:
            logger.warning(f"Emotion module error during stream: {e}")

        try:
            wellbeing_report = self.wellbeing.assess(
                health_report=health_report if 'health_report' in dir() else None,
                emotion_report=emotion_report if 'emotion_report' in dir() else None,
                prosodic_features=prosodic_dict,
                pet_id=pet_id
            )
            yield {
                "type": "wellbeing",
                "composite_score": wellbeing_report.composite_score,
                "composite_label": wellbeing_report.composite_label,
                "recommendations": wellbeing_report.recommendations,
            }
        except Exception as e:
            logger.warning(f"Wellbeing module error during stream: {e}")

        elapsed = time.time() - start_time
        yield {"type": "done", "processing_time_ms": int(elapsed * 1000)}

    def reset_session(self):
        """Clear call history for new session."""
        self.call_history = []
