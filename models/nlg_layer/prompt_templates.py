"""
AnimalVox AI — LLM Prompt Templates (Stage 5)
Templates for Mistral 7B / Groq translation generation.
"""

SYSTEM_PROMPT = """You are AnimalVox AI, an expert animal behaviorist and bioacoustics interpreter. 
Given an animal's detected behavioral state from acoustic analysis, translate what the animal 
is communicating into natural, first-person human language.

RULES:
1. All translations MUST be grounded in real ethological science
2. Use first-person voice as if the animal is speaking
3. Match intensity — urgent calls get urgent translations
4. Include emotional/motivational context
5. Be specific to the species when possible
6. Never fabricate behaviors not supported by the data
7. Output valid JSON only"""

TRANSLATION_PROMPT = """Species: {species} ({species_detail})
Primary behaviors detected: {behavior_labels}
Intensity: {intensity}/1.0
Urgency: {urgency}/1.0
Call sequence context: {sequence_context}
Escalation pattern: {escalation}
Ethology reference: {ethology_context}

Translate this into what the animal is communicating. Respond in this exact JSON format:
{{
  "translation": "First-person natural language, 2-3 sentences",
  "vocalization_type": "Scientific call type name",
  "emotion": "Primary emotional/motivational state",
  "behavior_context": "Scientific explanation, 2-3 sentences",
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
                 ethology_context="General species behavior"):
    """Build the complete translation prompt."""
    level = get_intensity_level(intensity)
    prompt = TRANSLATION_PROMPT.format(
        species=species, species_detail=species_detail or "unspecified",
        behavior_labels=", ".join(behavior_labels) if isinstance(behavior_labels, list) else behavior_labels,
        intensity=f"{intensity:.2f}", urgency=f"{urgency:.2f}",
        sequence_context=sequence_context, escalation=escalation,
        ethology_context=ethology_context
    )
    prompt += f"\n\nIntensity guidance: {INTENSITY_TEMPLATES[level]}"
    return prompt
