"""
EEG Preprocessing for Seizure Prediction

This module preprocesses raw EEG data before sending to the AI model.
Must match the preprocessing used during training.

Pipeline:
1. Common Average Reference (CAR)
2. Bandpass filter (0.5-45 Hz)
3. Segmentation (5 seconds = 1280 samples @ 256 Hz)

Usage:
    from preprocessing import preprocess_eeg, preprocess_single_segment
    
    # For continuous EEG (multiple 5-second segments)
    segments = preprocess_eeg(raw_data, sample_rate=256)
    
    # For a single 5-second segment
    segment = preprocess_single_segment(raw_data, sample_rate=256)
"""

import numpy as np
from scipy import signal


# Standard 23-channel montage for CHB-MIT
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8"
]


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to EEG data.
    Subtracts the mean of all channels from each channel.
    
    Args:
        data: shape (n_samples, 23) or (23, n_samples)
    
    Returns:
        CAR-referenced data, same shape as input
    """
    # Ensure (samples, channels) format
    if data.shape[0] == 23 and data.shape[1] != 23:
        data = data.T
        transposed = True
    else:
        transposed = False
    
    mean_signal = np.mean(data, axis=1, keepdims=True)
    referenced = data - mean_signal
    
    return referenced.T if transposed else referenced


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    sample_rate: int = 256,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter (0.5-45 Hz).
    
    Args:
        data: shape (n_samples, 23)
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        sample_rate: Sampling rate (Hz)
        order: Filter order
    
    Returns:
        Filtered data, same shape as input
    """
    nyquist = sample_rate / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply to each channel
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
    
    return filtered


def segment_eeg(data: np.ndarray, segment_samples: int = 1280) -> np.ndarray:
    """
    Split continuous EEG into 5-second segments (1280 samples @ 256 Hz).
    
    Args:
        data: shape (n_samples, 23)
        segment_samples: samples per segment (1280 = 5 seconds)
    
    Returns:
        segments: shape (n_segments, 1280, 23)
    """
    n_samples = data.shape[0]
    n_segments = n_samples // segment_samples
    
    if n_segments == 0:
        return np.array([])
    
    # Trim to exact multiple
    data = data[:n_segments * segment_samples]
    segments = data.reshape(n_segments, segment_samples, 23)
    
    return segments.astype(np.float32)


def preprocess_eeg(raw_data: np.ndarray, sample_rate: int = 256) -> np.ndarray:
    """
    Full preprocessing pipeline for continuous EEG.
    
    Args:
        raw_data: Raw EEG, shape (n_samples, 23) or (23, n_samples)
        sample_rate: Sampling rate (256 Hz for CHB-MIT)
    
    Returns:
        Preprocessed segments, shape (n_segments, 1280, 23, 1)
        Ready for model.predict()
    """
    # Ensure (samples, channels)
    if raw_data.shape[0] == 23 and raw_data.shape[1] != 23:
        raw_data = raw_data.T
    
    # 1. Common Average Reference
    data = common_average_reference(raw_data)
    
    # 2. Bandpass filter (0.5-45 Hz)
    data = bandpass_filter(data, sample_rate=sample_rate)
    
    # 3. Segment into 5-second windows
    segments = segment_eeg(data)
    
    if len(segments) == 0:
        return np.array([])
    
    # 4. Add channel dimension: (n, 1280, 23) -> (n, 1280, 23, 1)
    return segments[..., np.newaxis].astype(np.float32)


def preprocess_single_segment(raw_data: np.ndarray, sample_rate: int = 256) -> np.ndarray:
    """
    Preprocess a single 5-second EEG segment.
    
    Args:
        raw_data: shape (1280, 23) or (23, 1280) - exactly 5 seconds
        sample_rate: Sampling rate (256 Hz)
    
    Returns:
        Preprocessed segment, shape (1, 1280, 23, 1)
        Ready for model.predict()
    """
    # Ensure (samples, channels)
    if raw_data.shape[0] == 23:
        raw_data = raw_data.T
    
    # 1. Common Average Reference
    data = common_average_reference(raw_data)
    
    # 2. Bandpass filter
    data = bandpass_filter(data, sample_rate=sample_rate)
    
    # 3. Reshape for model: (1, 1280, 23, 1)
    return data.reshape(1, 1280, 23, 1).astype(np.float32)
