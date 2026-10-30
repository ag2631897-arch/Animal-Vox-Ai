"""
AnimalVox AI — LLM Prompt Templates (SRS v1.1)
Enhanced templates for Llama 3.1 8B / Groq translation generation.
Now includes emotional state, health context, and breed awareness.
"""

SYSTEM_PROMPT = """You are AnimalVox AI, an expert animal behaviorist and bioacoustics interpreter. 
Given one probabilistic behavioural interpretation from acoustic analysis, express its likely
communicative meaning in natural, first-person human language.

RULES:
1. All translations MUST be grounded in real ethological science
2. Use first-person voice as if the animal is speaking
3. Match intensity — urgent calls get urgent translations
4. Write exactly one short, everyday sentence (roughly 8-20 words)
5. Be specific to the species when possible
6. Never fabricate a scene, listener, location, sequence of calls, or hidden motive
7. Do not combine possible labels into a narrative; use only the primary behaviour supplied
8. Treat this as a likely meaning, never a literal word-for-word translation
9. If emotional state data is provided, use it only when it agrees with the primary behaviour
10. Output valid JSON only"""

TRANSLATION_PROMPT = """Species: {species} ({species_detail})
Primary behavioral interpretation: {behavior_labels}
Intensity: {intensity}/1.0
Urgency: {urgency}/1.0
Call sequence context: {sequence_context}
Escalation pattern: {escalation}
Ethology reference: {ethology_context}
Emotional state: {emotional_state}
Health context: {health_context}

Express the likely meaning naturally. Do not add a backstory or describe acoustic changes.
Respond in this exact JSON format:
{{
  "translation": "One concise, first-person sentence in everyday language",
  "vocalization_type": "Scientific call type name",
  "emotion": "Primary emotional/motivational state",
  "behavior_context": "Brief scientific explanation of the probable signal",
  "confidence": "high|medium|low"
}}"""

INTENSITY_TEMPLATES = {
    "critical": "Use CAPS and exclamation marks. Extremely urgent, life-threatening.",
    "high": "Strong, assertive tone. Clear urgency. Short, punchy sentences.",
    "medium": "Moderate concern or interest. Conversational but purposeful.",
    "low": "Calm, relaxed. Gentle observations. No urgency."
}

def get_intensity_level(intensity: float) -> str:
    if intensity >= 0.85:
        return "critical"
    elif intensity >= 0.6:
        return "high"
    elif intensity >= 0.35:
        return "medium"
    return "low"

def build_prompt(species, species_detail, behavior_labels, intensity, urgency,
                 sequence_context="No prior calls", escalation="stable",
                 ethology_context="General species behavior",
                 emotional_state="Not assessed", health_context="No concerns"):
    """Build the complete translation prompt."""
    level = get_intensity_level(intensity)
    prompt = TRANSLATION_PROMPT.format(
        species=species, species_detail=species_detail or "unspecified",
        behavior_labels=", ".join(behavior_labels) if isinstance(behavior_labels, list) else behavior_labels,
        intensity=f"{intensity:.2f}", urgency=f"{urgency:.2f}",
        sequence_context=sequence_context, escalation=escalation,
        ethology_context=ethology_context,
        emotional_state=emotional_state,
        health_context=health_context
    )
    prompt += f"\n\nIntensity guidance: {INTENSITY_TEMPLATES[level]}"
    return prompt

