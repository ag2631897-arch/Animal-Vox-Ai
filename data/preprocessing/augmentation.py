"""
AnimalVox AI — Data Augmentation Pipeline
===========================================
Audio augmentation strategies to improve model robustness
against real-world recording conditions.

Augmentations simulate: field noise, distance variation,
equipment differences, and natural acoustic variability.
"""

import numpy as np
import librosa
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """
    Augmentation pipeline for animal audio clips.

    Implements 6 augmentation strategies:
    1. Gaussian noise injection (simulates field recording noise)
    2. Time stretching (handles natural tempo variation)
    3. Pitch shifting (handles individual variation within species)
    4. Background noise mixing (urban, rain, wind)
    5. SpecAugment (frequency + time masking)
    6. Volume perturbation (simulates distance variation)
    """

    def __init__(self, sr: int = 16000, seed: int = 42):
        self.sr = sr
        self.rng = np.random.RandomState(seed)

    def add_gaussian_noise(
        self,
        audio: np.ndarray,
        snr_range: Tuple[float, float] = (10, 30)
    ) -> np.ndarray:
        """
        Add Gaussian noise at a random SNR level.

        Args:
            audio: Input audio array
            snr_range: Range of SNR values in dB (min, max)

        Returns:
            Noisy audio array
        """
        snr_db = self.rng.uniform(*snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise.astype(audio.dtype)

    def time_stretch(
        self,
        audio: np.ndarray,
        rate_range: Tuple[float, float] = (0.8, 1.2)
    ) -> np.ndarray:
        """
        Stretch/compress audio in time domain.

        Args:
            audio: Input audio array
            rate_range: Range of stretch rates (< 1 = slower, > 1 = faster)

        Returns:
            Time-stretched audio (may change length)
        """
        rate = self.rng.uniform(*rate_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched

    def pitch_shift(
        self,
        audio: np.ndarray,
        semitone_range: Tuple[float, float] = (-2, 2)
    ) -> np.ndarray:
        """
        Shift pitch by random semitones.

        Handles individual variation within species — two dogs of the
        same breed can have different vocal pitches.

        Args:
            audio: Input audio array
            semitone_range: Range of semitone shifts

        Returns:
            Pitch-shifted audio
        """
        n_steps = self.rng.uniform(*semitone_range)
        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sr, n_steps=n_steps
        )
        return shifted

    def add_background_noise(
        self,
        audio: np.ndarray,
        noise: np.ndarray,
        snr_range: Tuple[float, float] = (5, 20)
    ) -> np.ndarray:
        """
        Mix background environmental noise at random SNR.

        Args:
            audio: Input audio array
            noise: Background noise array (same sr)
            snr_range: SNR range in dB

        Returns:
            Audio mixed with background noise
        """
        snr_db = self.rng.uniform(*snr_range)

        # Adjust noise length to match audio
        if len(noise) < len(audio):
            # Repeat noise to fill
            repeats = (len(audio) // len(noise)) + 1
            noise = np.tile(noise, repeats)
        noise = noise[:len(audio)]

        # Scale noise to desired SNR
        signal_rms = np.sqrt(np.mean(audio ** 2))
        noise_rms = np.sqrt(np.mean(noise ** 2))
        if noise_rms > 0:
            desired_noise_rms = signal_rms / (10 ** (snr_db / 20))
            noise = noise * (desired_noise_rms / noise_rms)

        return audio + noise

    def spec_augment(
        self,
        mel_spec: np.ndarray,
        freq_mask_param: int = 20,
        time_mask_param: int = 40,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ) -> np.ndarray:
        """
        Apply SpecAugment: frequency and time masking on mel-spectrogram.

        Reference: Park et al., "SpecAugment" (2019)

        Args:
            mel_spec: Mel-spectrogram of shape (n_mels, T)
            freq_mask_param: Maximum frequency bands to mask
            time_mask_param: Maximum time frames to mask

        Returns:
            Augmented mel-spectrogram
        """
        spec = mel_spec.copy()
        n_mels, T = spec.shape

        # Frequency masking
        for _ in range(num_freq_masks):
            f = self.rng.randint(0, min(freq_mask_param, n_mels))
            f0 = self.rng.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = 0

        # Time masking
        for _ in range(num_time_masks):
            t = self.rng.randint(0, min(time_mask_param, T))
            t0 = self.rng.randint(0, max(1, T - t))
            spec[:, t0:t0 + t] = 0

        return spec

    def volume_perturbation(
        self,
        audio: np.ndarray,
        gain_range: Tuple[float, float] = (0.5, 1.5)
    ) -> np.ndarray:
        """
        Random volume change to simulate distance variation.

        Args:
            audio: Input audio array
            gain_range: Range of gain multipliers

        Returns:
            Volume-adjusted audio (clipped to [-1, 1])
        """
        gain = self.rng.uniform(*gain_range)
        return np.clip(audio * gain, -1.0, 1.0)

    def augment(
        self,
        audio: np.ndarray,
        prob: float = 0.5,
        background_noise: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply random augmentations with given probability.

        Args:
            audio: Input audio clip
            prob: Probability of applying each augmentation
            background_noise: Optional background noise for mixing

        Returns:
            Augmented audio clip
        """
        aug = audio.copy()

        if self.rng.random() < prob:
            aug = self.add_gaussian_noise(aug)

        if self.rng.random() < prob:
            aug = self.time_stretch(aug)

        if self.rng.random() < prob:
            aug = self.pitch_shift(aug)

        if background_noise is not None and self.rng.random() < prob:
            aug = self.add_background_noise(aug, background_noise)

        if self.rng.random() < prob:
            aug = self.volume_perturbation(aug)

        return aug
