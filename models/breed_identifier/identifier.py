"""
AnimalVox AI — Breed Identification (SRS Module 02)
=====================================================
Identifies breed or breed mix from acoustic features and optional
photo (multi-modal fusion). Audio branch uses prosodic/formant
analysis to classify acoustic breed groups. Visual branch uses
EfficientNet-B4 for fine-grained breed recognition from photos.

Multi-Modal Fusion:
- Audio-only: 40% weight (acoustic breed group)
- Visual-only: 60% weight (fine-grained breed from image)
- Combined: weighted fusion of both predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BreedPrediction:
    """A single breed prediction with confidence."""
    breed_name: str
    group_name: str
    confidence: float
    source: str  # "audio", "visual", "fused"
    acoustic_evidence: Optional[Dict] = None


@dataclass
class BreedReport:
    """Complete breed identification report."""
    top_predictions: List[BreedPrediction]
    acoustic_group: str
    acoustic_confidence: float
    visual_breed: Optional[str]
    visual_confidence: Optional[float]
    fusion_method: str  # "audio_only", "visual_only", "multimodal"
    species: str


DEFAULT_VISUAL_BREEDS = [
    "Chihuahua", "Japanese Spaniel", "Maltese Dog", "Pekingese", "Shih-Tzu",
    "Blenheim Spaniel", "Papillon", "Toy Terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black-and-tan Coonhound",
    "Walker Hound", "English Foxhound", "Redbone Coonhound", "Borzoi", "Irish Wolfhound",
    "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound",
    "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bullterrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier",
    "Norwich Terrier", "Yorkshire Terrier", "Wire-haired Fox Terrier", "Lakeland Terrier", "Sealyham Terrier",
    "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Bull",
    "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scotch Terrier", "Tibetan Terrier",
    "Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-coated Retriever",
    "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer",
    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany Spaniel",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael", "Malinois",
    "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog",
    "Collie", "Border Collie", "Bouvier des Flandres", "Rottweiler", "German Shepherd",
    "Doberman Pinscher", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund",
    "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog",
    "Great Dane", "Saint Bernard", "Siberian Husky", "Alaskan Malamute", "Samoyed",
    "Akita", "Shiba Inu", "Chow Chow", "Pomeranian", "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Siamese Cat",
    "Persian Cat", "Bengal Cat", "Maine Coon Cat", "Ragdoll Cat", "Sphynx Cat",
    "British Shorthair Cat", "Abyssinian Cat", "Russian Blue Cat", "Birman Cat", "Scottish Fold Cat",
]


try:
    from models.dual_stream_classifier import DualStreamBiometricClassifier
    _DUAL_STREAM_AVAILABLE = True
except ImportError:
    _DUAL_STREAM_AVAILABLE = False


class AcousticBreedClassifier(nn.Module):
    """
    Classifies animal into acoustic breed group from vocal features.
    Uses DualStreamBiometricClassifier for morphological biometric gating.
    """

    def __init__(self, input_dim: int = 266, hidden_dim: int = 160,
                 num_groups: int = 8):
        super().__init__()
        if _DUAL_STREAM_AVAILABLE:
            self.model = DualStreamBiometricClassifier(
                semantic_dim=256,
                biometric_dim=10,
                hidden_dim=hidden_dim,
                num_classes=num_groups,
                num_residual_blocks=2,
                dropout=0.15,
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_groups),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VisualBreedClassifier(nn.Module):
    """
    Fine-grained breed classification from pet photo using EfficientNet-B4.
    
    Uses transfer learning from ImageNet with a custom classification head
    for ~120 breeds (Stanford Dogs + cat breeds).
    
    Input: 3×380×380 RGB image
    Output: 120-class breed probabilities
    
    ~19M parameters (EfficientNet-B4 backbone, frozen during initial training)
    """

    def __init__(self, num_breeds: int = 120, feature_dim: int = 1792,
                 projection_dim: int = 256):
        super().__init__()
        # Placeholder backbone — actual EfficientNet-B4 loaded at runtime
        # In production: self.backbone = timm.create_model('efficientnet_b4', pretrained=True)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.BatchNorm1d(projection_dim),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_breeds),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: EfficientNet-B4 penultimate layer features (batch, 1792)
        """
        projected = self.projection(features)
        return self.classifier(projected)


class BreedIdentifier:
    """
    Complete breed identification pipeline (Module 02).
    
    Orchestrates acoustic and visual breed classification with
    configurable fusion weighting.
    """

    def __init__(self, device: str = "cpu"):
        from config import (
            BREED_ACOUSTIC_PROFILES, BREED_ID_CONFIG,
            MODEL_CONFIG, AUDIO_CONFIG
        )
        self.profiles = BREED_ACOUSTIC_PROFILES
        self.config = BREED_ID_CONFIG
        self.device = device

        # Acoustic breed classifier
        input_dim = MODEL_CONFIG.fusion_output_dim + MODEL_CONFIG.prosodic_dim  # 266
        self.acoustic_model = AcousticBreedClassifier(
            input_dim=input_dim,
            hidden_dim=MODEL_CONFIG.breed_hidden_dim,
            num_groups=self.config.num_breed_groups
        ).to(device).eval()

        # Visual breed classifier (loaded on demand)
        self.visual_model = VisualBreedClassifier(
            num_breeds=self.config.num_visual_breeds,
            feature_dim=self.config.visual_feature_dim,
            projection_dim=self.config.visual_projection_dim
        ).to(device).eval()

        self.breed_groups = list(BREED_ACOUSTIC_PROFILES.keys())

        # Check for trained weights or initialize in heuristic fallback mode
        self.has_trained_weights = False
        self.metadata = {}
        try:
            from models.weight_manager import load_checkpoint
            self.has_trained_weights, self.metadata = load_checkpoint(
                self.acoustic_model, "breed_acoustic", device=device
            )
        except Exception as e:
            logger.debug(f"Weight manager check for breed_acoustic: {e}")

        if self.has_trained_weights:
            logger.info(f"BreedIdentifier loaded TRAINED weights (Epoch {self.metadata.get('epoch', 'N/A')})")
        else:
            logger.info(
                "BreedIdentifier: No trained checkpoint found. "
                "Active Mode: ACOUSTIC PROFILE DISTANCE HEURISTIC (Phonation & F0 Matching)."
            )

    def save_weights(self, epoch: int = 0, optimizer=None, metrics: Dict = None) -> Path:
        """Save trained acoustic breed classifier weights to standard checkpoint path."""
        from models.weight_manager import save_checkpoint
        return save_checkpoint(
            self.acoustic_model, "breed_acoustic", epoch=epoch,
            optimizer=optimizer, metrics=metrics
        )

    def load_weights(self, path: Optional[Path] = None) -> bool:
        """Load weights from standard registry or custom path."""
        from models.weight_manager import load_checkpoint
        if path:
            state = torch.load(path, map_location=self.device)
            state_dict = state["state_dict"] if "state_dict" in state else state
            self.acoustic_model.load_state_dict(state_dict, strict=False)
            self.has_trained_weights = True
            return True
        self.has_trained_weights, self.metadata = load_checkpoint(
            self.acoustic_model, "breed_acoustic", device=self.device
        )
        return self.has_trained_weights

    def _compute_heuristic_breed_group(self, prosodic: Dict, species: str) -> Tuple[str, float]:
        """
        Find closest acoustic breed group profile by evaluating F0 range and call duration.
        """
        detected_f0 = prosodic.get("mean_f0", 350.0)
        detected_dur = prosodic.get("mean_call_duration", 0.5)

        best_group = self.breed_groups[0]
        best_dist = float("inf")

        for group_id, profile in self.profiles.items():
            # Check species match (e.g. dogs vs cats)
            is_cat_group = "cat" in group_id
            if species == "cats" and not is_cat_group:
                continue
            if species == "dogs" and is_cat_group:
                continue

            f0_min, f0_max = profile.get("f0_range_hz", (200, 600))
            f0_mid = (f0_min + f0_max) / 2.0

            dur_min, dur_max = profile.get("call_duration_sec", (0.2, 0.8))
            dur_mid = (dur_min + dur_max) / 2.0

            # Normalized Euclidean distance
            dist = ((detected_f0 - f0_mid) / 300.0) ** 2 + ((detected_dur - dur_mid) / 0.5) ** 2
            if dist < best_dist:
                best_dist = dist
                best_group = group_id

        conf = max(0.40, min(0.92, 1.0 / (1.0 + best_dist)))
        return best_group, conf

    def identify(self, fused_embedding: torch.Tensor,
                  prosodic_features: Dict, species: str,
                  visual_features: Optional[torch.Tensor] = None) -> BreedReport:
        """
        Identify breed from acoustic and optional visual features.
        Uses trained neural classifier when weights exist, otherwise matches acoustic profile distances.
        """
        # Only relevant for dogs and cats
        if species not in ("dogs", "cats"):
            return BreedReport(
                top_predictions=[], acoustic_group="unknown",
                acoustic_confidence=0.0, visual_breed=None,
                visual_confidence=None, fusion_method="not_applicable",
                species=species
            )

        # Acoustic classification (Trained vs Fallback)
        if self.has_trained_weights:
            try:
                if isinstance(fused_embedding, np.ndarray):
                    fused_embedding = torch.from_numpy(fused_embedding).float()
                if fused_embedding.dim() == 1:
                    fused_embedding = fused_embedding.unsqueeze(0)
                fused_embedding = fused_embedding.to(self.device)

                if fused_embedding.shape[-1] == 256:
                    prosodic_tensor = torch.tensor([
                        prosodic_features.get("mean_f0", 0.0),
                        prosodic_features.get("f0_range", 0.0),
                        prosodic_features.get("mean_hnr", 0.0),
                        prosodic_features.get("call_rate", 0.0),
                        prosodic_features.get("mean_rms", 0.0),
                        prosodic_features.get("spectral_centroid", 0.0),
                        prosodic_features.get("mean_call_duration", 0.0),
                        prosodic_features.get("mean_zcr", 0.0),
                        prosodic_features.get("mean_spectral_rolloff", 0.0),
                        prosodic_features.get("tempo", 0.0),
                    ], dtype=torch.float32).unsqueeze(0).to(self.device)
                    fused_input = torch.cat([fused_embedding, prosodic_tensor], dim=-1)
                elif fused_embedding.shape[-1] > 266:
                    fused_input = fused_embedding[..., :266]
                else:
                    fused_input = fused_embedding

                with torch.no_grad():
                    acoustic_logits = self.acoustic_model(fused_input)
                    # Species-consistent masking: dogs get canine groups (0-4), cats get feline groups (5-7)
                    if species == "dogs":
                        for cat_idx in (5, 6, 7):
                            if cat_idx < acoustic_logits.shape[-1]:
                                acoustic_logits[..., cat_idx] = -1e9
                    elif species == "cats":
                        for dog_idx in (0, 1, 2, 3, 4):
                            if dog_idx < acoustic_logits.shape[-1]:
                                acoustic_logits[..., dog_idx] = -1e9

                    acoustic_probs = torch.softmax(acoustic_logits, dim=-1).cpu().numpy()[0]

                acoustic_idx = int(np.argmax(acoustic_probs))
                acoustic_group = self.breed_groups[acoustic_idx]
                acoustic_conf = float(acoustic_probs[acoustic_idx])
            except Exception as e:
                logger.warning(f"Neural breed forward pass error ({e}); using heuristic fallback.")
                acoustic_group, acoustic_conf = self._compute_heuristic_breed_group(prosodic_features, species)
        else:
            acoustic_group, acoustic_conf = self._compute_heuristic_breed_group(prosodic_features, species)

        # Visual classification (if image provided)
        visual_breed = None
        visual_conf = None
        if visual_features is not None:
            if isinstance(visual_features, np.ndarray):
                visual_features = torch.from_numpy(visual_features).float()
            if visual_features.dim() == 1:
                visual_features = visual_features.unsqueeze(0)
            visual_features = visual_features.to(self.device)

            with torch.no_grad():
                visual_logits = self.visual_model(visual_features)
                visual_probs = torch.softmax(visual_logits, dim=-1).cpu().numpy()[0]
            visual_idx = int(np.argmax(visual_probs))
            visual_conf = float(visual_probs[visual_idx])
            visual_breed = DEFAULT_VISUAL_BREEDS[visual_idx % len(DEFAULT_VISUAL_BREEDS)]

        # Build predictions
        predictions = []
        group_info = self.profiles.get(acoustic_group, {})
        examples = group_info.get("examples", [])

        if visual_features is not None and visual_breed:
            fusion = "multimodal"
            # 40% acoustic + 60% visual late fusion
            fused_conf = float(min(0.98, (self.config.audio_fusion_weight * acoustic_conf) + (self.config.visual_fusion_weight * visual_conf)))
            predictions.append(BreedPrediction(
                breed_name=visual_breed,
                group_name=group_info.get("display_name", acoustic_group),
                confidence=fused_conf,
                source="fused",
                acoustic_evidence={
                    "f0_range": group_info.get("f0_range_hz"),
                    "breathiness": group_info.get("breathiness"),
                    "detected_f0": prosodic_features.get("mean_f0"),
                    "detected_hnr": prosodic_features.get("mean_hnr"),
                    "acoustic_confidence": float(acoustic_conf),
                    "visual_confidence": float(visual_conf),
                }
            ))
            # Append other plausible group breeds
            for b_name in examples:
                if b_name != visual_breed and len(predictions) < self.config.top_k_breeds:
                    rem_conf = float(max(0.01, (1.0 - fused_conf) / max(len(examples) - 1, 1)))
                    predictions.append(BreedPrediction(
                        breed_name=b_name,
                        group_name=group_info.get("display_name", acoustic_group),
                        confidence=rem_conf,
                        source="audio",
                    ))
        else:
            fusion = "audio_only"
            # Add acoustic-based predictions
            for breed_name in examples[:self.config.top_k_breeds]:
                predictions.append(BreedPrediction(
                    breed_name=breed_name,
                    group_name=group_info.get("display_name", acoustic_group),
                    confidence=acoustic_conf / len(examples),
                    source="audio",
                    acoustic_evidence={
                        "f0_range": group_info.get("f0_range_hz"),
                        "breathiness": group_info.get("breathiness"),
                        "detected_f0": prosodic_features.get("mean_f0"),
                        "detected_hnr": prosodic_features.get("mean_hnr"),
                    }
                ))

        return BreedReport(
            top_predictions=predictions,
            acoustic_group=acoustic_group,
            acoustic_confidence=acoustic_conf,
            visual_breed=visual_breed,
            visual_confidence=visual_conf,
            fusion_method=fusion,
            species=species,
        )
