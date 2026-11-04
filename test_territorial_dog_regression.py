"""Regression checks for dog territorial-bark handling.

Run with:
    python test_territorial_dog_regression.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import BEHAVIOR_LABELS
from inference.pipeline import AnimalVoxPipeline
from models.animal_detector.detector import AnimalDetectionPipeline
from models.behavioral_classifier.classifier import BehavioralClassifier
from models.nlg_layer.bioacoustic_translator import generate_human_translation
from models.nlg_layer.prompt_templates import build_prompt


class TerritorialDogRegressionTests(unittest.TestCase):
    def test_territorial_bark_maps_to_territorial_warning(self):
        # Representative traits of a firm, repeated boundary bark: mid-low
        # pitch, audible energy, moderate cadence, and a rough vocal quality.
        prosody = {
            "mean_f0": 300.0,
            "f0_range": 100.0,
            "mean_hnr": 6.0,
            "mean_rms": 0.22,
            "call_rate": 1.8,
            "spectral_centroid": 1700.0,
            "mean_zcr": 0.08,
            "mean_call_duration": 0.6,
            "f0_slope": -0.1,
        }
        classifier = BehavioralClassifier(device="cpu")
        scores = classifier._compute_heuristic_behaviors(prosody, species="dogs")
        primary_behavior = BEHAVIOR_LABELS[int(np.argmax(scores))]

        self.assertEqual(primary_behavior, "territorial_warning")

    def test_territorial_translation_is_short_and_natural(self):
        result = generate_human_translation(
            species="dogs",
            behavior="territorial_warning",
            intensity=0.82,
            urgency=0.88,
        )

        self.assertEqual(
            result["translation"],
            "This is my territory. Stay back and do not come any closer.",
        )
        self.assertNotIn("pack", result["translation"].lower())

    def test_rapid_guard_bark_uses_shared_guarding_meaning(self):
        # Measurements from the original user-uploaded barking clip: a fast,
        # high-energy warning bark. Audio alone cannot prove ownership of a
        # location, so its safe interpretation is a guarding/intruder alert.
        prosody = {
            "mean_f0": 460.4,
            "f0_range": 395.6,
            "mean_hnr": 25.9,
            "mean_rms": 0.23,
            "call_rate": 6.2,
            "spectral_centroid": 974.7,
            "mean_zcr": 0.065,
            "mean_call_duration": 5.0,
            "f0_slope": -0.97,
        }
        classifier = BehavioralClassifier(device="cpu")
        scores = classifier._compute_heuristic_behaviors(prosody, species="dogs")
        primary_behavior = BEHAVIOR_LABELS[int(np.argmax(scores))]
        translation = generate_human_translation(
            species="dogs",
            behavior=primary_behavior,
            intensity=0.70,
            urgency=0.83,
        )

        self.assertEqual(primary_behavior, "alarm_ground")
        self.assertEqual(
            translation["translation"],
            "Stay away from here. I'm on guard and warning you not to come closer.",
        )

    def test_llm_output_with_unrelated_story_is_rejected(self):
        bad_response = {
            "translation": (
                "I bark sharply to warn my pack, then switch to playful barks "
                "because I am lonely."
            )
        }
        good_response = {"translation": "Stay away from here. I am warning you not to come closer."}

        self.assertFalse(AnimalVoxPipeline._is_grounded_translation(bad_response, "alarm_ground"))
        self.assertTrue(AnimalVoxPipeline._is_grounded_translation(good_response, "alarm_ground"))

    def test_conflicting_emotion_does_not_override_warning_meaning(self):
        pipeline = AnimalVoxPipeline(device="cpu")

        display_emotion = pipeline._select_translation_emotion(
            "alarm_ground", "Loneliness / Isolation"
        )

        self.assertEqual(display_emotion, "alert / protective")

    def test_llm_prompt_has_one_primary_interpretation(self):
        prompt = build_prompt(
            species="dogs",
            species_detail=None,
            behavior_labels=["territorial_warning"],
            intensity=0.82,
            urgency=0.88,
        )

        self.assertIn("Primary behavioral interpretation: territorial_warning", prompt)
        self.assertIn("Do not add a backstory", prompt)

    def test_species_checkpoint_does_not_fabricate_soundscape(self):
        detector = AnimalDetectionPipeline(device="cpu")
        self.assertFalse(detector.has_trained_soundscape_weights)
        self.assertFalse(detector.has_trained_count_weights)


if __name__ == "__main__":
    unittest.main(verbosity=2)
