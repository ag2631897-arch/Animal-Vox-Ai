"""
AnimalVox AI — End-to-End Inference Pipeline
Orchestrates all 5 stages: preprocessing → feature extraction
→ classification → context reasoning → NLG translation.
"""

import json
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    AUDIO_CONFIG, MODEL_CONFIG, INFERENCE_CONFIG,
    BEHAVIOR_LABELS, SPECIES_GROUPS, NUM_SPECIES_GROUPS
)
from data.preprocessing.standardize_audio import preprocess_audio, preprocess_audio_bytes
from data.preprocessing.generate_spectrograms import generate_mel_spectrogram, mel_to_image
from models.behavioral_classifier.prosodic_extractor import extract_prosodic_features
from models.acoustic_encoder.encoder import AnimalAcousticEncoder
from models.behavioral_classifier.classifier import BehavioralClassifier, SpectrogramCNN
from models.sequence_model.call_sequence_transformer import CallSequenceTransformer
from models.nlg_layer.prompt_templates import build_prompt, SYSTEM_PROMPT
from knowledge_base.ethology_db import EthologyKnowledgeBase

logger = logging.getLogger(__name__)


class AnimalVoxPipeline:
    """
    Complete AnimalVox AI inference pipeline.
    
    Audio → Preprocessing → Feature Extraction (3 paths) →
    Behavioral Classification → Context Reasoning → LLM Translation
    """

    def __init__(self, device: str = None, groq_api_key: str = None):
        self.device = device or INFERENCE_CONFIG.device
        self.groq_api_key = groq_api_key
        self.call_history: List[Dict] = []
        self._models_loaded = False

        # Initialize knowledge base
        self.kb = EthologyKnowledgeBase(use_chroma=False)
        self.kb.load_seed_data()
        logger.info(f"Knowledge base loaded: {self.kb.size} entries")

    def load_models(self):
        """Load all AI models (lazy loading for fast startup)."""
        if self._models_loaded:
            return

        logger.info("Loading models...")
        dev = self.device

        # Path A: Acoustic encoder
        self.encoder = AnimalAcousticEncoder(embed_dim=768, projection_dim=256)
        self.encoder.to(dev).eval()

        # Path B: Spectrogram CNN
        self.cnn = SpectrogramCNN(output_dim=256)
        self.cnn.to(dev).eval()

        # Stage 3: Behavioral classifier
        self.classifier = BehavioralClassifier(input_dim=522, n_cls=12)
        self.classifier.to(dev).eval()

        # Stage 6: Sequence transformer
        self.seq_model = CallSequenceTransformer(embed_dim=276, ctx_dim=128)
        self.seq_model.to(dev).eval()

        self._models_loaded = True
        logger.info("All models loaded successfully")

    def _extract_features(self, audio_clip: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from all 3 parallel paths."""
        dev = self.device

        # Path A: Audio embedding
        audio_tensor = torch.from_numpy(audio_clip).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            audio_embed = self.encoder(audio_tensor)  # (1, 256)

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

    def _classify(self, audio_embed, cnn_features, prosodic) -> Dict:
        """Run behavioral classification on fused features."""
        fused = torch.cat([audio_embed, cnn_features, prosodic], dim=-1)
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
            "audio_embed": audio_embed
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
                                     seq_context: str, escalation: str) -> Dict:
        """Generate natural language translation via Groq API."""
        behavior_names = [b["name"] for b in classification["behaviors"]]
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

        # Try Groq API first
        if self.groq_api_key:
            try:
                return await self._call_groq(prompt)
            except Exception as e:
                logger.warning(f"Groq API failed: {e}, using fallback")

        # Fallback: template-based translation
        return self._fallback_translation(species, classification)

    async def _call_groq(self, prompt: str) -> Dict:
        """Call Groq API for Mistral 7B inference."""
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

    def _fallback_translation(self, species: str, classification: Dict) -> Dict:
        """Template-based fallback when LLM is unavailable."""
        top_behavior = classification["behaviors"][0]
        template = self.kb.get_translation_template(
            species, top_behavior["name"], classification["intensity"]
        )

        if not template:
            template = f"[{species}] Detected {top_behavior['name']} behavior (intensity: {classification['intensity']:.0%})"

        return {
            "translation": template,
            "vocalization_type": top_behavior["name"].replace("_", " ").title(),
            "emotion": self._infer_emotion(top_behavior["name"], classification["intensity"]),
            "behavior_context": f"Detected {', '.join(b['name'] for b in classification['behaviors'])} "
                               f"with {classification['intensity']:.0%} intensity.",
            "confidence": "high" if top_behavior["score"] > 0.7 else "medium" if top_behavior["score"] > 0.4 else "low"
        }

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

    async def translate(self, audio_path: str = None, audio_bytes: bytes = None,
                        audio_format: str = "wav", species: str = "birds") -> Dict:
        """
        Main entry point: translate animal audio to human language.
        
        Returns dict with: translation, vocalization_type, emotion,
        behavior_context, confidence, behaviors, intensity, urgency, timing
        """
        start_time = time.time()
        self.load_models()

        # Stage 1: Preprocessing
        if audio_path:
            clips = preprocess_audio(audio_path)
        elif audio_bytes:
            clips = preprocess_audio_bytes(audio_bytes, format=audio_format)
        else:
            raise ValueError("Provide audio_path or audio_bytes")

        if not clips:
            return {"error": "No audio content detected", "translation": ""}

        # Use the first clip (or best clip in future)
        clip = clips[0]

        # Stage 2: Feature extraction (3 parallel paths)
        audio_embed, cnn_features, prosodic = self._extract_features(clip)

        # Stage 3: Behavioral classification
        classification = self._classify(audio_embed, cnn_features, prosodic)

        # Stage 4: Context reasoning
        seq_context, escalation = self._get_sequence_context(classification, species)

        # Stage 5: NLG Translation
        translation = await self._generate_translation(species, classification, seq_context, escalation)

        elapsed = time.time() - start_time

        return {
            **translation,
            "species": species,
            "species_emoji": SPECIES_GROUPS.get(species, {}).get("emoji", "🔊"),
            "behaviors": classification["behaviors"],
            "intensity": classification["intensity"],
            "urgency": classification["urgency"],
            "escalation_pattern": escalation,
            "processing_time_ms": int(elapsed * 1000),
            "mel_spectrogram": generate_mel_spectrogram(clip).tolist()
        }

    def reset_session(self):
        """Clear call history for new session."""
        self.call_history = []
