"""
AnimalVox AI — Wellbeing Signals Dashboard (SRS Module 04)
============================================================
Aggregates health, emotional, and behavioural data into four
wellbeing dimensions based on the Five Domains Model of animal welfare.

Dimensions & Weights:
- Physical Distress    (35%) — Domains 1 & 3 (Nutrition + Health)
- Environmental Stress (20%) — Domain 2 (Environment)
- Isolation Stress     (20%) — Domain 4 (Behaviour)
- Contentment          (25%) — Domain 5 (Mental State)

Composite Score: 0-100 (higher = better overall wellbeing)
"""

import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DimensionScore:
    """Score for a single wellbeing dimension."""
    name: str
    score: float           # 0-100
    weight: float          # 0-1
    direction: str         # "lower_is_better" | "higher_is_better"
    contributing_factors: List[str]
    trend: str             # "improving", "stable", "declining"
    trend_delta: float     # Points change over trend window


@dataclass
class WellbeingReport:
    """Complete wellbeing assessment."""
    composite_score: float          # 0-100
    composite_label: str            # "Thriving", "Good", "Fair", "Concern"
    dimensions: Dict[str, DimensionScore]
    trend: str                      # Overall trend
    recommendations: List[str]
    pet_id: Optional[str]
    timestamp: str


class WellbeingScorer:
    """
    Computes the four-dimensional wellbeing score.
    
    Uses acoustic evidence from the health monitor, emotional tracker,
    and behavioural classifier to compute each dimension's raw score,
    then combines them into a single composite score.
    """

    def __init__(self):
        from config import WELLBEING_DIMENSIONS, WELLBEING_CONFIG
        self.dimensions = WELLBEING_DIMENSIONS
        self.config = WELLBEING_CONFIG

    def compute_physical_distress(self, health_alerts: List,
                                  respiratory_normal: bool,
                                  prosodic: Dict) -> float:
        """
        Compute physical distress score (0-100, lower = better).
        
        Sources:
        - Health flag count and severity
        - Respiratory normality
        - Vocal energy changes
        """
        score = 0.0

        # Health flags contribute proportionally to severity
        severity_weights = {"critical": 40, "high": 25, "medium": 15, "low": 8}
        for alert in health_alerts:
            sev = getattr(alert, "severity", "low")
            score += severity_weights.get(sev, 5)

        # Respiratory distress
        if not respiratory_normal:
            score += 20

        # Abnormal RMS (very low = lethargy, very high = distress)
        if prosodic and "mean_rms" in prosodic:
            rms = prosodic.get("mean_rms", 0.1)
            if rms < 0.01:  # Very quiet — possible lethargy
                score += 10
            elif rms > 0.5:  # Very loud — possible distress
                score += 10

        return min(score, 100.0)

    def compute_environmental_stress(self, emotion_report,
                                     soundscape: str = "",
                                     diary_entries: List = None) -> float:
        """
        Compute environmental stress score (0-100, lower = better).
        
        Sources:
        - Territorial signalling frequency (E05)
        - Acute stress episodes (E02)
        - Soundscape type (urban environments score higher)
        """
        score = 0.0

        if emotion_report:
            stress = emotion_report.stress_level
            score += stress * 50  # Stress maps to 0-50 of this dimension

            # Check for territorial signalling
            for e in emotion_report.secondary_emotions:
                if e.state_id == "E05" and e.confidence > 0.3:
                    score += 15
        elif diary_entries:
            stress_vals = [float(e.get("stress_level", 0.0)) for e in diary_entries]
            if stress_vals:
                score += (sum(stress_vals) / len(stress_vals)) * 50
            for entry in diary_entries[:3]:
                for em in entry.get("all_emotions", []):
                    if em.get("id") == "E05" and float(em.get("conf", 0.0)) > 0.3:
                        score += 5

        # Soundscape adjustments
        soundscape_stress = {
            "indoor": 0, "outdoor_garden": 5,
            "urban_park": 15, "forest_wilderness": 10, "marine": 5
        }
        score += soundscape_stress.get(soundscape, 0)

        return min(score, 100.0)

    def compute_isolation_stress(self, emotion_report,
                                 diary_entries: List = None) -> float:
        """
        Compute isolation stress score (0-100, lower = better).
        
        Sources:
        - Loneliness/isolation emotion (E10)
        - Separation distress calls
        - Time-alone patterns from diary
        """
        score = 0.0

        if emotion_report:
            # Direct loneliness detection
            for pred in [emotion_report.primary_emotion] + emotion_report.secondary_emotions:
                if pred.state_id == "E10":
                    score += pred.confidence * 60
                elif pred.state_id == "E08":  # Frustration
                    score += pred.confidence * 20
        elif diary_entries:
            e10_scores = []
            e08_scores = []
            for entry in diary_entries:
                for em in entry.get("all_emotions", []):
                    if em.get("id") == "E10":
                        e10_scores.append(float(em.get("conf", 0.0)))
                    elif em.get("id") == "E08":
                        e08_scores.append(float(em.get("conf", 0.0)))
            if e10_scores:
                score += (sum(e10_scores) / len(e10_scores)) * 60
            if e08_scores:
                score += (sum(e08_scores) / len(e08_scores)) * 20

        # Diary-based isolation patterns
        if diary_entries and len(diary_entries) > 3:
            isolation_count = sum(
                1 for e in diary_entries if e.get("primary_emotion") == "E10"
            )
            isolation_ratio = isolation_count / len(diary_entries)
            score += isolation_ratio * 40

        return min(score, 100.0)

    def compute_contentment(self, emotion_report,
                            diary_entries: List = None) -> float:
        """
        Compute contentment score (0-100, higher = better).
        
        Sources:
        - Contentment emotion (E01)
        - Social bonding (E06)
        - Playfulness (E07)
        - Mood stability
        """
        score = 0.0
        positive_states = {"E01": 0.4, "E06": 0.25, "E07": 0.20, "E09": 0.15}

        if emotion_report:
            for pred in [emotion_report.primary_emotion] + emotion_report.secondary_emotions:
                if pred.state_id in positive_states:
                    weight = positive_states[pred.state_id]
                    score += pred.confidence * weight * 100

            # Mood stability bonus
            score += emotion_report.mood_stability * 20
        elif diary_entries:
            entry_pos_scores = []
            for entry in diary_entries:
                pos_sum = 0.0
                for em in entry.get("all_emotions", []):
                    sid = em.get("id")
                    if sid in positive_states:
                        pos_sum += float(em.get("conf", 0.0)) * positive_states[sid] * 100
                entry_pos_scores.append(pos_sum)
            if entry_pos_scores:
                score += sum(entry_pos_scores) / len(entry_pos_scores)

            # Mood stability from diary entries
            if len(diary_entries) > 1:
                switches = sum(
                    1 for i in range(1, len(diary_entries))
                    if diary_entries[i].get("primary_emotion") != diary_entries[i - 1].get("primary_emotion")
                )
                stability = 1.0 - (switches / (len(diary_entries) - 1))
            else:
                stability = 1.0
            score += stability * 20

        # Diary-based contentment trending
        if diary_entries and len(diary_entries) >= 1:
            content_count = sum(
                1 for e in diary_entries
                if e.get("primary_emotion") in ("E01", "E06", "E07")
            )
            content_ratio = content_count / len(diary_entries)
            score += content_ratio * 30

        return min(score, 100.0)

    def compute_composite(self, dimensions: Dict[str, DimensionScore]) -> float:
        """
        Compute weighted composite wellbeing score.
        
        For "lower_is_better" dimensions, we invert:
          contribution = (100 - raw_score) * weight
        For "higher_is_better" dimensions:
          contribution = raw_score * weight
        """
        composite = 0.0
        for dim_key, dim in dimensions.items():
            if dim.direction == "lower_is_better":
                contribution = (100.0 - dim.score) * dim.weight
            else:
                contribution = dim.score * dim.weight
            composite += contribution
        return round(composite, 1)


class WellbeingHistory:
    """
    Persistent wellbeing score history for trend analysis.
    Stores weekly snapshots as JSON files.
    """

    def __init__(self, profiles_dir: Path):
        self.profiles_dir = profiles_dir

    def save_snapshot(self, pet_id: str, report: WellbeingReport):
        """Save a wellbeing snapshot."""
        history_file = self.profiles_dir / f"{pet_id}_wellbeing_history.json"
        history = self._load_history(history_file)

        history.append({
            "timestamp": str(report.timestamp),
            "composite": float(report.composite_score),
            "dimensions": {
                str(k): {"score": float(v.score), "trend": str(v.trend)}
                for k, v in report.dimensions.items()
            },
        })

        # Keep last 52 weeks
        if len(history) > 52:
            history = history[-52:]

        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save wellbeing snapshot for {pet_id}: {e}")

    def get_trend(self, pet_id: str, window_weeks: int = 4) -> Tuple[str, float]:
        """
        Compute trend from recent history.
        
        Returns (trend_label, delta_points).
        """
        history_file = self.profiles_dir / f"{pet_id}_wellbeing_history.json"
        history = self._load_history(history_file)

        if len(history) < 2:
            return "stable", 0.0

        recent = history[-window_weeks:]
        scores = [float(h["composite"]) for h in recent if "composite" in h]

        if len(scores) < 2:
            return "stable", 0.0

        delta = scores[-1] - scores[0]
        if delta > 5.0:
            return "improving", delta
        elif delta < -5.0:
            return "declining", delta
        return "stable", delta

    def _load_history(self, path: Path) -> List[Dict]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Could not load wellbeing history from {path}: {e}")
                return []
        return []


class WellbeingDashboard:
    """
    Complete wellbeing signals pipeline (Module 04).
    
    Aggregates all module outputs into the four-dimensional
    wellbeing report with trend analysis and recommendations.
    """

    def __init__(self, device: str = "cpu"):
        from config import PET_PROFILES_DIR, WELLBEING_CONFIG
        self.config = WELLBEING_CONFIG
        self.scorer = WellbeingScorer()
        self.history = WellbeingHistory(PET_PROFILES_DIR)

        logger.info("WellbeingDashboard initialized (Module 04)")

    def assess(self, health_report=None, emotion_report=None,
               prosodic_features: Dict = None, soundscape: str = "",
               diary_entries: List = None,
               pet_id: Optional[str] = None,
               save_to_history: bool = True) -> WellbeingReport:
        """
        Generate comprehensive wellbeing report.
        
        Args:
            health_report: Output from HealthMonitor
            emotion_report: Output from EmotionalTracker
            prosodic_features: Prosodic features dict
            soundscape: Detected soundscape category
            diary_entries: Recent diary entries for the pet
            pet_id: Optional pet ID for trend tracking
            save_to_history: Whether to persist a new snapshot to wellbeing history
            
        Returns:
            WellbeingReport with composite score and dimension breakdowns
        """
        prosodic = prosodic_features or {}
        if not prosodic and diary_entries:
            for entry in diary_entries:
                if entry.get("prosodic_snapshot"):
                    prosodic = entry["prosodic_snapshot"]
                    break
        health_alerts = health_report.alerts if health_report else []
        resp_normal = health_report.respiratory_normal if health_report else True

        # Compute four dimensions
        phys = self.scorer.compute_physical_distress(
            health_alerts, resp_normal, prosodic
        )
        env = self.scorer.compute_environmental_stress(
            emotion_report, soundscape, diary_entries
        )
        iso = self.scorer.compute_isolation_stress(
            emotion_report, diary_entries
        )
        cont = self.scorer.compute_contentment(
            emotion_report, diary_entries
        )

        # Get trends if pet has history
        phys_trend, phys_delta = ("stable", 0.0)
        env_trend, env_delta = ("stable", 0.0)
        iso_trend, iso_delta = ("stable", 0.0)
        cont_trend, cont_delta = ("stable", 0.0)
        overall_trend = "stable"
        if pet_id:
            overall_trend, _ = self.history.get_trend(
                pet_id, self.config.trend_window_weeks
            )

        dimensions = {
            "physical_distress": DimensionScore(
                name="Physical Distress", score=phys,
                weight=self.config.physical_weight,
                direction="lower_is_better",
                contributing_factors=[
                    f"{len(health_alerts)} health alert(s)",
                    f"Respiratory {'normal' if resp_normal else 'ABNORMAL'}",
                ],
                trend=phys_trend, trend_delta=phys_delta
            ),
            "environmental_stress": DimensionScore(
                name="Environmental Stress", score=env,
                weight=self.config.environmental_weight,
                direction="lower_is_better",
                contributing_factors=[
                    f"Soundscape: {soundscape or 'unknown'}",
                ],
                trend=env_trend, trend_delta=env_delta
            ),
            "isolation_stress": DimensionScore(
                name="Isolation Stress", score=iso,
                weight=self.config.isolation_weight,
                direction="lower_is_better",
                contributing_factors=[],
                trend=iso_trend, trend_delta=iso_delta
            ),
            "contentment": DimensionScore(
                name="Contentment", score=cont,
                weight=self.config.contentment_weight,
                direction="higher_is_better",
                contributing_factors=[],
                trend=cont_trend, trend_delta=cont_delta
            ),
        }

        composite = self.scorer.compute_composite(dimensions)

        # Label
        if composite >= 80:
            label = "Thriving"
        elif composite >= 60:
            label = "Good"
        elif composite >= 40:
            label = "Fair"
        else:
            label = "Concern"

        # Generate recommendations
        recommendations = self._generate_recommendations(dimensions, composite)

        report = WellbeingReport(
            composite_score=composite,
            composite_label=label,
            dimensions=dimensions,
            trend=overall_trend,
            recommendations=recommendations,
            pet_id=pet_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Save to history
        if pet_id and save_to_history:
            self.history.save_snapshot(pet_id, report)

        return report

    def _generate_recommendations(self, dimensions: Dict[str, DimensionScore],
                                  composite: float) -> List[str]:
        """Generate actionable recommendations based on scores."""
        recs = []

        phys = dimensions.get("physical_distress")
        if phys and phys.score > 50:
            recs.append(
                "⚕ Physical distress signals detected. "
                "Consider a veterinary check-up. "
                "(This is an early-awareness aid, not a diagnosis.)"
            )

        env = dimensions.get("environmental_stress")
        if env and env.score > 40:
            recs.append(
                "🏠 Environmental stress elevated. "
                "Consider noise reduction, safe spaces, or routine stabilisation."
            )

        iso = dimensions.get("isolation_stress")
        if iso and iso.score > 40:
            recs.append(
                "🤝 Isolation signals detected. "
                "Increase social interaction, consider companion or enrichment."
            )

        cont = dimensions.get("contentment")
        if cont and cont.score < 30:
            recs.append(
                "😔 Low contentment baseline. "
                "Increase play sessions, positive interactions, and routine stability."
            )

        if not recs and composite >= 60:
            recs.append(
                "✅ Your pet appears to be doing well! "
                "Keep monitoring to maintain this positive trend."
            )

        return recs
