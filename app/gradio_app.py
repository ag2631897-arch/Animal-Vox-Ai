"""
AnimalVox AI — Gradio Web Application (Stage 7)
Premium web UI with audio recording, species selection,
waveform visualization, and translation display.
"""

import os
import asyncio
import json
import numpy as np
import logging
import tempfile
from pathlib import Path

import gradio as gr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SPECIES_GROUPS, BEHAVIOR_LABELS, AUDIO_CONFIG
from inference.pipeline import AnimalVoxPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key
from dotenv import load_dotenv
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# Global pipeline instance
pipeline = AnimalVoxPipeline(device="cpu", groq_api_key=GROQ_KEY if GROQ_KEY else None)

# ── CSS Theme ────────────────────────────────────────────────
CUSTOM_CSS = """
/* Dark premium theme */
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.main-title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8em !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1em;
    margin-top: 0;
    margin-bottom: 24px;
}
.translation-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 28px;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.translation-text {
    font-size: 1.4em;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1.6;
    margin-bottom: 16px;
}
.meta-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 600;
    margin: 4px;
}
.badge-high { background: #dc262620; color: #f87171; border: 1px solid #dc262640; }
.badge-medium { background: #f59e0b20; color: #fbbf24; border: 1px solid #f59e0b40; }
.badge-low { background: #10b98120; color: #34d399; border: 1px solid #10b98140; }
.badge-info { background: #3b82f620; color: #60a5fa; border: 1px solid #3b82f640; }
.behavior-tag {
    display: inline-block;
    background: #6366f120;
    color: #a5b4fc;
    border: 1px solid #6366f140;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    margin: 2px;
}
.science-note {
    background: #1e1b4b;
    border-left: 3px solid #818cf8;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-top: 12px;
    color: #c7d2fe;
    font-size: 0.9em;
}
.stats-row {
    display: flex;
    gap: 12px;
    margin: 12px 0;
}
.stat-item {
    flex: 1;
    background: #1e293b;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.stat-value {
    font-size: 1.5em;
    font-weight: 700;
    color: #818cf8;
}
.stat-label {
    font-size: 0.8em;
    color: #64748b;
    margin-top: 4px;
}
footer { display: none !important; }
"""

# ── Sample Translations (for demo) ──────────────────────────
EXAMPLE_TRANSLATIONS = {
    "birds": {
        "translation": "FREEZE! There's a hawk directly overhead. Don't move a muscle — stay in cover until I give the all-clear!",
        "vocalization_type": "Aerial Predator Alarm",
        "emotion": "fear / urgent warning",
        "behavior_context": "American Robin producing rapid thin 'seet' calls — a well-documented aerial predator alarm call. This pattern triggers an immediate freeze response in all nearby birds. Source: Morse 1980, Cornell Lab of Ornithology.",
        "confidence": "high",
        "behaviors": [{"name": "alarm_aerial", "score": 0.92}, {"name": "distress_call", "score": 0.41}],
        "intensity": 0.88, "urgency": 0.91
    },
    "dogs": {
        "translation": "Come on, come ON! Let's play! Why are you just standing there?? PLAY WITH ME! I've got so much energy!",
        "vocalization_type": "Play Solicitation",
        "emotion": "joyful / excited",
        "behavior_context": "Short, high-pitched repeated barks characteristic of play solicitation in domestic dogs. Often accompanied by a play bow. Source: Bekoff 2001, Bradshaw 2011.",
        "confidence": "high",
        "behaviors": [{"name": "play_vocalization", "score": 0.89}, {"name": "social_bonding", "score": 0.35}],
        "intensity": 0.82, "urgency": 0.25
    },
    "cats": {
        "translation": "I feel completely safe with you right now. This is exactly where I want to be. Don't move — everything is perfect.",
        "vocalization_type": "Contentment Purr",
        "emotion": "content / affectionate",
        "behavior_context": "Slow rhythmic purring at 25-50 Hz indicates social bonding and contentment. Purring can also be self-soothing. Source: Turner & Bateson 2014.",
        "confidence": "medium",
        "behaviors": [{"name": "social_bonding", "score": 0.94}],
        "intensity": 0.75, "urgency": 0.05
    },
    "dolphins": {
        "translation": "It's me — it's me! I'm over here by the reef. Where are you all? This is my whistle — you know it's me!",
        "vocalization_type": "Signature Whistle",
        "emotion": "social / seeking connection",
        "behavior_context": "Bottlenose dolphin signature whistle — a unique ascending whistle pattern that serves as an individual identity call. Each dolphin develops their own signature whistle in the first year of life. Source: Herzing 2011.",
        "confidence": "high",
        "behaviors": [{"name": "contact_call", "score": 0.91}, {"name": "social_bonding", "score": 0.48}],
        "intensity": 0.65, "urgency": 0.42
    },
    "whales": {
        "translation": "I am here. I am strong. Listen to how long and complex my song is. I've been singing for hours. Come find me.",
        "vocalization_type": "Breeding Song",
        "emotion": "confident / hopeful",
        "behavior_context": "Male humpback whale breeding song — complex melodic sequence that can last 10-30 minutes and carry for miles underwater. Song complexity indicates male fitness. Source: Payne & McVay 1971.",
        "confidence": "high",
        "behaviors": [{"name": "mating_call", "score": 0.87}, {"name": "social_bonding", "score": 0.32}],
        "intensity": 0.79, "urgency": 0.30
    },
    "frogs": {
        "translation": "I am right here, I am ready, and I am the LOUDEST male in this pond tonight! Ladies, come find me!",
        "vocalization_type": "Advertisement Call",
        "emotion": "confident / hopeful",
        "behavior_context": "American bullfrog 'jug-o-rum' advertisement call. Lower pitch indicates larger body size. Males call from established territories. Source: Duellman & Trueb 1994.",
        "confidence": "high",
        "behaviors": [{"name": "mating_call", "score": 0.93}],
        "intensity": 0.85, "urgency": 0.55
    }
}


def format_result_html(result: dict) -> str:
    """Format translation result as rich HTML card."""
    emoji = SPECIES_GROUPS.get(result.get("species", ""), {}).get("emoji", "🔊")
    conf = result.get("confidence", "medium")
    badge_cls = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}.get(conf, "badge-medium")

    intensity = result.get("intensity", 0)
    urgency = result.get("urgency", 0)
    behaviors = result.get("behaviors", [])

    behavior_tags = " ".join(
        f'<span class="behavior-tag">{b["name"].replace("_", " ").title()} ({b["score"]:.0%})</span>'
        for b in behaviors
    )

    html = f"""
    <div class="translation-card">
        <div class="translation-text">{emoji} "{result.get('translation', 'Processing...')}"</div>
        <div style="margin: 12px 0;">
            <span class="meta-badge badge-info">🎵 {result.get('vocalization_type', 'Unknown')}</span>
            <span class="meta-badge badge-info">💭 {result.get('emotion', 'Unknown')}</span>
            <span class="meta-badge {badge_cls}">📊 Confidence: {conf.upper()}</span>
        </div>
        <div class="stats-row">
            <div class="stat-item">
                <div class="stat-value">{intensity:.0%}</div>
                <div class="stat-label">Intensity</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{urgency:.0%}</div>
                <div class="stat-label">Urgency</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{result.get('processing_time_ms', 0)}ms</div>
                <div class="stat-label">Processing</div>
            </div>
        </div>
        <div style="margin: 8px 0;">{behavior_tags}</div>
        <div class="science-note">
            🔬 <strong>Scientific Context:</strong> {result.get('behavior_context', 'N/A')}
        </div>
    </div>
    """
    return html


def translate_audio(audio_input, species_choice):
    """Main translation function called by Gradio."""
    if audio_input is None:
        # Return demo translation
        species_key = species_choice.lower().split(" ")[0]
        for key in SPECIES_GROUPS:
            if key.startswith(species_key):
                species_key = key
                break
        demo = EXAMPLE_TRANSLATIONS.get(species_key, EXAMPLE_TRANSLATIONS["birds"])
        demo["species"] = species_key
        demo["processing_time_ms"] = 0
        return format_result_html(demo)

    try:
        # Get species key
        species_key = species_choice.lower().split(" ")[0]
        for key in SPECIES_GROUPS:
            if key.startswith(species_key):
                species_key = key
                break

        # Handle Gradio audio input (tuple of sr, data or filepath)
        if isinstance(audio_input, tuple):
            sr, data = audio_input
            # Save to temp file
            import soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, data, sr)
            audio_path = tmp.name
        elif isinstance(audio_input, str):
            audio_path = audio_input
        else:
            return "<div class='translation-card'><p>Unsupported audio format</p></div>"

        # Run pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            pipeline.translate(audio_path=audio_path, species=species_key)
        )
        loop.close()

        return format_result_html(result)

    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        return f"<div class='translation-card'><p style='color:#f87171'>Error: {str(e)}</p></div>"


def show_demo(species_choice):
    """Show example translation for selected species."""
    species_key = species_choice.lower().split(" ")[0]
    for key in SPECIES_GROUPS:
        if key.startswith(species_key):
            species_key = key
            break
    demo = EXAMPLE_TRANSLATIONS.get(species_key, EXAMPLE_TRANSLATIONS["birds"])
    demo["species"] = species_key
    demo["processing_time_ms"] = 0
    return format_result_html(demo)


def reset_session():
    """Reset call history for new session."""
    pipeline.reset_session()
    return "<div class='translation-card'><p style='color:#34d399'>✅ Session reset. Call history cleared.</p></div>"


# ── Build Gradio App ─────────────────────────────────────────
def create_app():
    species_choices = [
        f"{v['emoji']} {v['display_name']}" for v in SPECIES_GROUPS.values()
    ]

    with gr.Blocks(css=CUSTOM_CSS, title="AnimalVox AI — Animal Translation", theme=gr.themes.Base()) as app:

        # Header
        gr.HTML('<h1 class="main-title">🐾 AnimalVox AI</h1>')
        gr.HTML('<p class="subtitle">Scientifically-grounded animal vocalization translator powered by bioacoustics AI</p>')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎙️ Input")
                species_select = gr.Dropdown(
                    choices=species_choices,
                    value=species_choices[0],
                    label="Select Animal Group",
                    info="Choose the animal you're recording"
                )
                audio_input = gr.Audio(
                    label="Record or Upload Audio",
                    sources=["microphone", "upload"],
                    type="numpy",
                    max_length=60
                )
                with gr.Row():
                    translate_btn = gr.Button("🔬 Translate", variant="primary", size="lg")
                    demo_btn = gr.Button("✨ Show Demo", variant="secondary", size="lg")
                reset_btn = gr.Button("🔄 Reset Session", size="sm")

                gr.Markdown("""
                ### 📋 How It Works
                1. **Select** the animal group
                2. **Record** audio or upload a file
                3. **Click** Translate to analyze
                
                *Supports .wav, .mp3, .flac, .ogg, .m4a*
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 🌍 Translation")
                result_html = gr.HTML(
                    value=format_result_html({**EXAMPLE_TRANSLATIONS["birds"], "species": "birds", "processing_time_ms": 0})
                )

        # About section
        with gr.Accordion("🔬 About AnimalVox AI", open=False):
            gr.Markdown("""
            **AnimalVox AI** is a behavioral bioacoustics intelligence platform that translates animal 
            vocalizations into human language based on **peer-reviewed ethology research**.
            
            Every translation is grounded in decades of scientific study from the Cornell Lab of Ornithology, 
            Bradshaw's dog cognition research, Herzing's dolphin communication studies, and Roger Payne's whale song research.
            
            **This is NOT a fantasy translator** — animals don't have words or grammar. What AnimalVox AI does 
            is interpret the *behavioral meaning* of acoustic signals and express that meaning in natural human language.
            
            | Animal | Confidence | Key Sources |
            |--------|-----------|-------------|
            | 🐦 Birds | High | Cornell Lab, Morse 1980, 80+ years of ornithology |
            | 🐕 Dogs | High | Bradshaw 2011, Yin 2002, extensive behavior research |
            | 🐈 Cats | Medium | Turner & Bateson 2014, McComb et al. 2009 |
            | 🐬 Dolphins | High | Herzing 2011, Janik & Slater 1998, NOAA |
            | 🐋 Whales | Medium-High | Payne 1995, NOAA/MBARI research |
            | 🐸 Frogs | Medium | Duellman & Trueb 1994 |
            
            **100% Free & Open Source** — MIT Licensed
            """)

        # Wire up events
        translate_btn.click(fn=translate_audio, inputs=[audio_input, species_select], outputs=[result_html])
        demo_btn.click(fn=show_demo, inputs=[species_select], outputs=[result_html])
        reset_btn.click(fn=reset_session, outputs=[result_html])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
