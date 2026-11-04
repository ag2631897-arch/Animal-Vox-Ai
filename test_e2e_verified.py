import asyncio
import sys
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path.cwd()))
from inference.pipeline import AnimalVoxPipeline

pipe = AnimalVoxPipeline()

async def test_suite():
    print("=" * 75)
    print("ANIMALVOX AI — BIOACOUSTIC TRANSLATION & FILTER INTEGRATION TEST")
    print("=" * 75)

    test_cases = [
        ("Vine boom (SFX)", r"C:\Users\acer\Downloads\Vine boom sound effect - Business Goose.mp3", "no_animal_detected"),
        ("The Weeknd (Song)", r"C:\Users\acer\Downloads\The Weeknd - Blinding Lights (Official Instrumental) - TheWeekndVEVO.mp3", "music_detected"),
        ("Vampire Heart (BGM)", r"C:\Users\acer\Downloads\vampire heart - Isak Roen.mp3", "music_detected"),
        ("Lampaui (BGM)", r"C:\Users\acer\Downloads\Lampaui - Lelampa.mp3", "music_detected"),
        ("Genuine Husky Bark", r"data\sample_audio\husky_bark.wav", "translated"),
        ("Genuine Chihuahua Bark", r"data\sample_audio\chihuahua_bark.wav", "translated"),
    ]

    for label, path, expected in test_cases:
        print(f"\n[TEST CASE]: {label}")
        print(f"  File: {Path(path).name}")
        res = await pipe.translate(audio_path=path)
        err = res.get("error")
        if err:
            print(f"  Result: REJECTED (error={err})")
            print(f"  Reason: {res.get('rejection_reason') or res.get('message')}")
            assert (expected in (err, "rejected")), f"Expected {expected}, got {err}"
        else:
            print(f"  Result: TRANSLATED (species={res.get('species')})")
            print(f"  Translation: \"{res.get('translation')}\"")
            assert expected == "translated", f"Expected {expected}, got translated"

    # Test Case: Human Voice Mimicking Dog Bark ('Woof!')
    print("\n[TEST CASE]: Human Voice Mimicking Dog Bark ('Woof!')")
    sr = 16000
    t = np.linspace(0, 0.4, int(sr * 0.4))
    # Human voice F0=160Hz with vowel /ʊ/ formants (F1=480Hz, F2=1100Hz)
    human_woof = np.zeros_like(t)
    for h in range(1, 15):
        f = 160 * h
        w = np.exp(-((f - 480)**2) / (2 * 70**2)) + 0.6 * np.exp(-((f - 1100)**2) / (2 * 100**2))
        human_woof += w * np.sin(2 * np.pi * f * t)
    human_woof = human_woof / np.max(np.abs(human_woof)) * 0.5

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, human_woof, sr)
        tmp_human = tmp.name

    try:
        res = await pipe.translate(audio_path=tmp_human, filename="my_woof_imitation.wav")
        print(f"  Result: REJECTED (error={res.get('error')})")
        print(f"  Reason: {res.get('rejection_reason')}")
        assert res.get("error") == "human_detected", f"Expected human_detected, got {res.get('error')}"
    finally:
        if os.path.exists(tmp_human):
            os.unlink(tmp_human)

    # Test Case: Mixed Audio (Human Talking + Real Dog Bark)
    print("\n[TEST CASE]: Mixed Audio (Human Talking + Genuine Dog Bark)")
    dog_bark, _ = librosa.load("data/sample_audio/husky_bark.wav", sr=sr)
    # 1.2s human talking + 0.2s pause + husky bark
    t_speech = np.linspace(0, 1.2, int(sr * 1.2))
    human_talking = np.zeros_like(t_speech)
    for h in range(1, 18):
        f = 130 * h
        w = np.exp(-((f - 520)**2) / (2 * 80**2)) + 0.7 * np.exp(-((f - 1450)**2) / (2 * 110**2))
        human_talking += w * np.sin(2 * np.pi * f * t_speech)
    human_talking = human_talking / np.max(np.abs(human_talking)) * 0.35

    mixed = np.concatenate([human_talking, np.zeros(int(sr * 0.2)), dog_bark])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, mixed, sr)
        tmp_mixed = tmp.name

    try:
        res = await pipe.translate(audio_path=tmp_mixed, filename="dog_and_owner.wav")
        print(f"  Result: {res.get('error') or 'TRANSLATED'}")
        print(f"  Species: {res.get('species')}")
        print(f"  Human Speech Removed: {res.get('human_speech_removed')}")
        print(f"  Translation: \"{res.get('translation')}\"")
        assert res.get("error") is None, f"Expected translation, got error: {res.get('error')}"
        assert res.get("human_speech_removed") is True, "Expected human_speech_removed to be True"
    finally:
        if os.path.exists(tmp_mixed):
            os.unlink(tmp_mixed)

    # Test Case 9: Human Cough at start + Faint Bark Mimicry (Browser Mic Recording)
    print("\n[TEST CASE]: Human Cough at start + Faint Bark Mimicry (Browser Mic Recording)")
    t_cough = np.linspace(0, 0.45, int(sr * 0.45))
    noise = np.random.normal(0, 1, len(t_cough)) * np.exp(-t_cough * 9)
    cough = np.zeros_like(t_cough)
    for f, bw in [(520, 80), (1300, 120), (2600, 180)]:
        w = 2 * np.pi * f / sr
        r = np.exp(-np.pi * bw / sr)
        filtered = np.zeros_like(noise)
        for i in range(2, len(noise)):
            filtered[i] = noise[i] + 2 * r * np.cos(w) * filtered[i-1] - (r**2) * filtered[i-2]
        cough += filtered
    cough = cough / np.max(np.abs(cough)) * 0.55

    t_faint = np.linspace(0, 0.35, int(sr * 0.35))
    faint_bark = np.zeros_like(t_faint)
    for h in range(1, 10):
        f = 160 * h
        w = np.exp(-((f - 480)**2) / (2 * 80**2)) + 0.5 * np.exp(-((f - 1100)**2) / (2 * 120**2))
        faint_bark += w * np.sin(2 * np.pi * f * t_faint)
    faint_bark = faint_bark / np.max(np.abs(faint_bark)) * 0.04

    user_mic = np.concatenate([cough, np.zeros(int(sr * 0.4)), faint_bark, np.zeros(int(sr * 2.5))])
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, user_mic, sr)
        tmp_user_mic = tmp.name

    try:
        res = await pipe.translate(audio_path=tmp_user_mic, filename="recording.webm")
        print(f"  Result: REJECTED (error={res.get('error')})")
        print(f"  Reason: {res.get('rejection_reason') or res.get('message')}")
        assert res.get("error") == "human_detected", f"Expected human_detected, got {res.get('error')}"
    finally:
        if os.path.exists(tmp_user_mic):
            os.unlink(tmp_user_mic)

    print("\n" + "=" * 75)
    print("ALL 9 RIGOROUS TESTS PASSED PERFECTLY!")
    print("=" * 75)

if __name__ == "__main__":
    asyncio.run(test_suite())
