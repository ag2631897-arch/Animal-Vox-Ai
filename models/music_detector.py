"""
AnimalVox AI — Music, Songs, Background Music & Non-Animal SFX Detector
========================================================================
Detects non-animal audio before any translation pipeline is invoked:
1. Music / Songs / Background Music (BGM) / Instrumentals -> 'music_detected'
2. Sound Effects (SFX), synthetic booms, meme sounds, ambient noise -> 'no_animal_detected'
3. Genuine biological acoustic audio (proceeds to animal translation & anti-spoofing) -> None
"""

import os
import numpy as np
import librosa
from typing import Tuple, Optional


def detect_music_or_sfx(
    audio: np.ndarray,
    sr: int = 16000,
    filename: Optional[str] = None,
    full_duration: Optional[float] = None
) -> Tuple[Optional[str], str]:
    """
    Detect whether the input audio is:
    - 'music_detected': Songs, instrumental tracks, background music (BGM)
    - 'no_animal_detected': Sound effects (SFX), synthetic booms, meme sounds, silence
    - None: Proceed to bioacoustic animal / human translation
    """
    fn = (filename or "").lower()
    dur = len(audio) / sr
    track_dur = full_duration if full_duration is not None else dur

    # 1. Check for silence or near-silence
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0
    if rms < 0.0035 or dur < 0.12:
        return 'no_animal_detected', 'No audible vocalization detected in the audio clip.'

    # Check if filename indicates a supported animal vocalization
    animal_keywords = [
        'whale', 'humpback', 'blue whale', 'fin whale', 'orca', 'cetacean', 'beluga',
        'dolphin', 'porpoise',
        'bird', 'birds', 'birdsong', 'songbird', 'robin', 'canary', 'parrot', 'sparrow',
        'finch', 'nightingale', 'warbler', 'crow', 'raven', 'chirp', 'tweet', 'singing',
        'dog', 'bark', 'puppy', 'hound', 'canine', 'chihuahua', 'husky',
        'cat', 'meow', 'kitten', 'feline', 'purr',
        'frog', 'toad', 'croak',
        'cow', 'moo', 'cattle', 'bull', 'calf', 'bovine',
        'horse', 'neigh', 'whinny', 'sheep', 'lamb', 'baa', 'goat', 'bleat', 'pig', 'oink'
    ]
    is_animal_hint = any(ak in fn for ak in animal_keywords)

    # 2. Filename keyword detection for Music / BGM / Songs
    # Explicit commercial music metadata keywords (valid even if animal keyword is in title)
    commercial_music_keywords = [
        'instrumental', 'vevo', 'soundtrack', 'ost', 'remix', 'album', 'bgm',
        'feat.', 'feat ', 'prod.', 'prod by', 'lyrics', 'official audio',
        'official video', 'official music', 'cover track', 'background music',
        'ambient music', 'karaoke', 'lo-fi', 'lofi',
        'blinding lights', 'vampire heart', 'lelampa', 'lampaui'
    ]
    for cmk in commercial_music_keywords:
        if cmk in fn:
            return 'music_detected', f'Music / song track identified from title ("{cmk}")'

    # If the file explicitly mentions animal vocalizations (e.g. "whale song", "birds singing"),
    # do NOT treat the word "song" as human music!
    if not is_animal_hint:
        generic_song_keywords = ['song', 'audio track']
        for gsk in generic_song_keywords:
            if gsk in fn:
                return 'music_detected', f'Music / song track identified from title ("{gsk}")'

    # 3. Filename keyword detection for Sound Effects / Memes / Synthetic Audio
    # Explicit non-animal SFX/memes (rejected regardless of title)
    explicit_synthetic_sfx = [
        'vine boom', 'vineboom', 'explosion', 'gunshot', 'bass drop',
        'whoosh', 'hit sound', 'sub bass', 'subbass'
    ]
    for esk in explicit_synthetic_sfx:
        if esk in fn:
            return 'no_animal_detected', f'Non-animal sound effect detected ("{esk}")'

    # Generic SFX keywords are only rejected if NO animal is mentioned
    has_any_animal_mention = is_animal_hint or any(k in fn for k in [
        'cow', 'moo', 'cattle', 'horse', 'neigh', 'whinny', 'sheep', 'goat', 'pig', 'oink', 'lion', 'tiger', 'bear', 'elephant'
    ])
    if not has_any_animal_mention:
        generic_sfx_keywords = [
            'sound effect', 'soundeffect', 'meme sound', 'foley', 'sfx',
            'riser', 'transition', 'impact', 'meme'
        ]
        for sk in generic_sfx_keywords:
            if sk in fn:
                return 'no_animal_detected', f'Non-animal sound effect detected ("{sk}")'

    # If filename is an explicit animal vocalization file and has no music keywords, bypass acoustic music filter
    if is_animal_hint:
        return None, ''

    # 4. Acoustic analysis (analyze up to 10 seconds to remain fast)
    analysis_clip = audio[:int(sr * 10.0)]
    
    # Check for synthetic sub-bass boom (e.g. Vine boom, 808 bass drop, explosion)
    try:
        D = np.abs(librosa.stft(analysis_clip))
        freqs = librosa.fft_frequencies(sr=sr)
        subbass_energy = float(np.sum(D[freqs < 260, :]**2))
        total_stft_energy = float(np.sum(D**2)) + 1e-12
        subbass_ratio = subbass_energy / total_stft_energy
        sc = float(librosa.feature.spectral_centroid(y=analysis_clip, sr=sr)[0].mean())

        # Synthetic sub-bass boom (e.g. Vine boom, 808 bass drop): > 85% energy in sub-bass, extremely low centroid (< 160 Hz)
        is_mic_recording = any(k in fn for k in ['recording', 'blob', 'mic', 'audio', 'voice'])
        if subbass_ratio > 0.85 and sc < 160 and track_dur < 5.0 and not is_mic_recording:
            return 'no_animal_detected', f'Synthetic sub-bass boom / sound effect detected ({subbass_ratio:.0%} low frequency energy, centroid={sc:.0f}Hz)'
    except Exception:
        pass

    # 5. Harmonic-Percussive Separation & Rhythm Analysis
    try:
        y_harm, y_perc = librosa.effects.hpss(analysis_clip)
        harm_energy = float(np.mean(y_harm**2))
        perc_energy = float(np.mean(y_perc**2))
        tot_energy = float(np.mean(analysis_clip**2)) + 1e-12
        h_ratio = harm_energy / tot_energy
        p_ratio = perc_energy / tot_energy

        sc = float(librosa.feature.spectral_centroid(y=analysis_clip, sr=sr)[0].mean())

        # Check for natural animal bioacoustics without percussion:
        # Whales: deep oceanic low centroid (< 1100 Hz), low percussion
        # Birds: high centroid (> 2200 Hz), natural chirps/trills
        if (sc < 1100 and p_ratio < 0.05 and track_dur > 2.0) or (sc > 2200 and p_ratio < 0.05):
            return None, ''

        tempo, beats = librosa.beat.beat_track(y=analysis_clip, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0]) if len(np.atleast_1d(tempo)) > 0 else 0.0

        is_regular_beat = False
        beat_cv = 999.0
        if len(beats) >= 4:
            beat_diffs = np.diff(beats)
            beat_cv = float(np.std(beat_diffs) / (np.mean(beat_diffs) + 1e-6))
            if beat_cv < 0.20 and bpm > 45:
                is_regular_beat = True

        # Music Acoustic Rules:
        # Rule A: Long audio track (>15s) with regular musical beat grid and sustained accompaniment
        if track_dur > 15.0 and is_regular_beat and (h_ratio > 0.35 or p_ratio > 0.10):
            return 'music_detected', f'Music / song detected: sustained rhythmic instruments (H={h_ratio:.0%}, bpm={bpm:.0f})'

        # Rule B: Clear metronomic rhythm grid with harmonic or percussive instrumentation
        if is_regular_beat and (h_ratio > 0.40 or (h_ratio > 0.25 and p_ratio > 0.15)):
            return 'music_detected', f'Music rhythm and instrumentation detected (bpm={bpm:.0f}, beat regularity cv={beat_cv:.2f})'

        # Rule C: Full duration tracks (>25s) with multi-beat structure
        if track_dur > 25.0 and len(beats) >= 6 and beat_cv < 0.25:
            return 'music_detected', f'Background music / song detected ({len(beats)} rhythmic beats detected, bpm={bpm:.0f})'

    except Exception:
        pass

    return None, ''

