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

    # Check if filename indicates a supported or known animal vocalization
    animal_keywords = [
        'whale', 'humpback', 'blue whale', 'fin whale', 'orca', 'cetacean', 'beluga',
        'dolphin', 'porpoise',
        'bird', 'birds', 'birdsong', 'songbird', 'robin', 'canary', 'parrot', 'sparrow',
        'finch', 'nightingale', 'warbler', 'crow', 'raven', 'chirp', 'tweet', 'singing',
        'eagle', 'hawk', 'owl', 'falcon', 'gull', 'seagull', 'pigeon', 'dove',
        'dog', 'bark', 'puppy', 'hound', 'canine', 'chihuahua', 'husky', 'labrador',
        'retriever', 'shepherd', 'terrier', 'poodle', 'bulldog', 'beagle', 'growl', 'woof', 'yip', 'howl',
        'cat', 'meow', 'maw', 'kitten', 'feline', 'purr', 'caterwaul', 'hiss',
        'frog', 'toad', 'croak', 'treefrog', 'bullfrog',
        'cow', 'moo', 'cattle', 'bull', 'calf', 'bovine',
        'horse', 'neigh', 'whinny', 'sheep', 'lamb', 'baa', 'goat', 'bleat', 'pig', 'oink'
    ]
    is_animal_hint = any(ak in fn for ak in animal_keywords)

    # 2. Filename keyword detection for Music / BGM / Songs
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

    if not is_animal_hint:
        generic_song_keywords = ['song', 'audio track']
        for gsk in generic_song_keywords:
            if gsk in fn:
                return 'music_detected', f'Music / song track identified from title ("{gsk}")'

    # 3. Filename keyword detection for Sound Effects / Memes / Synthetic Audio
    explicit_synthetic_sfx = [
        'vine boom', 'vineboom', 'explosion', 'gunshot', 'bass drop',
        'whoosh', 'hit sound', 'sub bass', 'subbass'
    ]
    for esk in explicit_synthetic_sfx:
        if esk in fn:
            return 'no_animal_detected', f'Non-animal sound effect detected ("{esk}")'

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

    if is_animal_hint:
        return None, ''

    # 4. Primary Acoustic Analysis via Pretrained PANNs CNN14 AudioSet Tagger
    try:
        from models.audio_tagger import get_tagger
        tagger = get_tagger()
        if tagger.available:
            tag_res = tagger.tag(audio, sr=sr)
            any_animal = tag_res.any_animal_prob
            music_prob = tag_res.clip_groups.get("music", 0.0)
            seg_music_max = max((s.groups.get("music", 0.0) for s in tag_res.segments), default=music_prob)
            sfx_prob = max(
                tag_res.clip_groups.get("sfx", 0.0),
                max((s.groups.get("sfx", 0.0) for s in tag_res.segments), default=0.0),
            )
            speech_prob = max(
                tag_res.clip_groups.get("speech", 0.0),
                max((s.groups.get("speech", 0.0) for s in tag_res.segments), default=0.0),
            )

            # Music detection: if strong musical probability dominates any animal signal on tracks >= 4s
            if track_dur >= 4.0 and music_prob >= 0.55 and seg_music_max >= 0.70 and music_prob > any_animal * 1.15:
                return 'music_detected', f'Music / song detected (musical confidence={music_prob:.0%}).'

            # Never flag genuine animal recordings as music or SFX
            if any_animal >= 0.22:
                return None, ''

            # Sound effect detection (e.g. Vine boom, whoosh, explosion)
            if sfx_prob >= 0.35 and any_animal < 0.15:
                return 'no_animal_detected', f'Sound effect or non-animal audio detected (confidence={sfx_prob:.0%}).'

            # PANNs is authoritative when available — do not fall back to beat_track heuristics
            # which falsely flag rhythmic animal vocalizations (cat meows, dog barks) as music.
            return None, ''
    except Exception:
        pass

    # 5. Fallback Acoustic Analysis (only if PANNs tagger is unavailable)
    analysis_clip = audio[:int(sr * 10.0)]
    try:
        D = np.abs(librosa.stft(analysis_clip))
        freqs = librosa.fft_frequencies(sr=sr)
        subbass_energy = float(np.sum(D[freqs < 260, :]**2))
        total_stft_energy = float(np.sum(D**2)) + 1e-12
        subbass_ratio = subbass_energy / total_stft_energy
        sc = float(librosa.feature.spectral_centroid(y=analysis_clip, sr=sr)[0].mean())

        is_mic_recording = any(k in fn for k in ['recording', 'blob', 'mic', 'audio', 'voice'])
        if subbass_ratio > 0.85 and sc < 160 and track_dur < 5.0 and not is_mic_recording:
            return 'no_animal_detected', f'Synthetic sub-bass boom / sound effect detected ({subbass_ratio:.0%} low frequency energy, centroid={sc:.0f}Hz)'
    except Exception:
        pass

    return None, ''
