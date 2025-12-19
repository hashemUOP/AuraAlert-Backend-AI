"""
Simple EDF Seizure Predictor - Single Function Version

This module provides a single function that does everything:
1. Loads the EDF file
2. Preprocesses the data
3. Loads models (cached)
4. Runs prediction
5. Returns comprehensive results

Usage:
    from simple_edf_predictor import predict_seizure

    result = predict_seizure("path/to/file.edf")
    print(result["seizure_alert"])  # True/False
"""

import os
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union, List

import numpy as np
from scipy.signal import firwin, filtfilt

# ============================================================================
# Configuration
# ============================================================================

MODELS_DIR = os.environ.get(
    "SEIZURE_MODELS_DIR",
    "C:/Users/sara_/PycharmProjects/AuraAlert/backend/ai/checkpoints"
)

THRESHOLD = 0.5
SEGMENT_SAMPLES = 1280  # 5 seconds @ 256 Hz
EXPECTED_CHANNELS = 23

# Global model cache
_models: List[Any] = []


# ============================================================================
# Single Prediction Function
# ============================================================================

def predict_seizure(edf_file: Union[str, bytes, Path]) -> Dict[str, Any]:
    """
    Predict seizure risk from an EDF file.

    This single function handles everything:
    - Loads and parses the EDF file
    - Preprocesses EEG data (CAR + bandpass filter)
    - Segments data into 5-second windows
    - Loads models (cached after first call)
    - Runs ensemble prediction
    - Returns comprehensive results

    Args:
        edf_file: Path to EDF file OR raw bytes from file upload

    Returns:
        Dictionary with prediction results:
        {
            "seizure_alert": bool,          # True = seizure warning, False = stable
            "result": str,                   # "At Risk" or "Stable"
            "probability": float,            # Combined probability (0-1)
            "mean_probability": float,       # Average across segments
            "max_probability": float,        # Maximum segment probability
            "confidence": float,             # Confidence percentage (0-100)
            "alert_level": str,              # "low", "medium", "high", "critical"
            "ai_detected": str,              # Human-readable detection status
            "analysis_time": str,            # Formatted timestamp
            "file_duration_sec": float,      # File duration in seconds
            "file_duration_min": float,      # File duration in minutes
            "segments_analyzed": int,        # Number of 5-sec segments
            "segments_at_risk": int,         # Segments above threshold
            "risk_ratio": float,             # Ratio of at-risk segments
            "n_models": int,                 # Number of models used
            "error": str | None,             # Error message if any
        }
    """
    global _models

    # =========================================================================
    # STEP 1: Import MNE for EDF parsing
    # =========================================================================
    try:
        import mne
        mne.set_log_level('ERROR')
    except ImportError:
        return _error_response("MNE library not installed. Run: pip install mne")

    # =========================================================================
    # STEP 2: Load and parse EDF file
    # =========================================================================
    temp_path = None
    try:
        if isinstance(edf_file, bytes):
            # Handle raw bytes (from file upload)
            with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as f:
                f.write(edf_file)
                temp_path = f.name
            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
        else:
            # Handle file path
            raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
    except Exception as e:
        return _error_response(f"Failed to read EDF: {e}")
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()

    # Extract info
    sample_rate = int(raw.info['sfreq'])
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times

    if n_channels < EXPECTED_CHANNELS:
        return _error_response(f"Need at least {EXPECTED_CHANNELS} channels, got {n_channels}")

    if n_samples < SEGMENT_SAMPLES:
        return _error_response(f"EDF too short: need at least {SEGMENT_SAMPLES / sample_rate:.1f} seconds")

    # Get data (MNE returns Volts, convert to microvolts)
    # Shape: (channels, samples) -> take first 23 channels
    data = raw.get_data()[:EXPECTED_CHANNELS, :] * 1e6

    # Transpose to (samples, channels) for processing
    data = data.T.astype(np.float32)

    # =========================================================================
    # STEP 3: Preprocess - Common Average Reference (CAR)
    # =========================================================================
    data = data - np.mean(data, axis=1, keepdims=True)

    # =========================================================================
    # STEP 4: Preprocess - Bandpass Filter (0.5-45 Hz FIR)
    # =========================================================================
    nyq = sample_rate / 2
    fir_order = 256
    taps = firwin(fir_order + 1, [0.5 / nyq, 45 / nyq], pass_zero=False, window='hamming')

    for ch in range(EXPECTED_CHANNELS):
        data[:, ch] = filtfilt(taps, 1.0, data[:, ch].astype(np.float64)).astype(np.float32)

    # =========================================================================
    # STEP 5: Segment into 5-second windows (50% overlap)
    # =========================================================================
    segments = []
    idx = 0
    step = SEGMENT_SAMPLES // 2  # 50% overlap

    while idx + SEGMENT_SAMPLES <= n_samples:
        segments.append(data[idx:idx + SEGMENT_SAMPLES])
        idx += step

    if len(segments) == 0:
        return _error_response("Could not create segments from data")

    # Reshape for model input: (n_segments, 1280, 23, 1)
    batch = np.array(segments, dtype=np.float32).reshape(len(segments), SEGMENT_SAMPLES, EXPECTED_CHANNELS, 1)

    # =========================================================================
    # STEP 6: Load models (cached after first call)
    # =========================================================================
    if len(_models) == 0:
        try:
            import tensorflow as tf

            models_path = Path(MODELS_DIR)
            print(f"Loading models from {models_path}...")

            for i in range(7):
                model_file = models_path / f"best_model_patientchb01_seizure{i}_preictal60_run42.keras"
                if model_file.exists():
                    try:
                        model = tf.keras.models.load_model(str(model_file), compile=False)
                    except TypeError as e:
                        if "seed" in str(e):
                            model = tf.keras.models.load_model(str(model_file), compile=False, safe_mode=False)
                        else:
                            raise
                    _models.append(model)

            print(f"Loaded {len(_models)} models")

            if len(_models) == 0:
                return _error_response("No models found in " + str(models_path))

        except Exception as e:
            return _error_response(f"Failed to load models: {e}")

    # =========================================================================
    # STEP 7: Run ensemble prediction
    # =========================================================================
    all_preds = []
    for model in _models:
        preds = model.predict(batch, verbose=0, batch_size=32).flatten()
        all_preds.append(preds)

    # Average predictions across models
    probs = np.mean(all_preds, axis=0).tolist()

    # =========================================================================
    # STEP 8: Aggregate results
    # =========================================================================
    mean_prob = float(np.mean(probs))
    max_prob = float(np.max(probs))
    at_risk = sum(1 for p in probs if p >= THRESHOLD)
    risk_ratio = at_risk / len(probs)

    # Determine seizure alert
    seizure_alert = mean_prob >= THRESHOLD or risk_ratio >= 0.3

    # Combined probability
    probability = 0.6 * mean_prob + 0.4 * max_prob if seizure_alert else mean_prob

    # Confidence (distance from decision boundary)
    confidence = round(abs(probability - 0.5) * 200, 1)

    # Alert level
    if probability >= 0.9:
        alert_level = "critical"
    elif probability >= 0.7:
        alert_level = "high"
    elif probability >= 0.5:
        alert_level = "medium"
    else:
        alert_level = "low"

    # =========================================================================
    # STEP 9: Build and return response
    # =========================================================================
    now = datetime.now()

    return {
        # Primary results
        "seizure_alert": bool(seizure_alert),
        "result": "At Risk" if seizure_alert else "Stable",
        "probability": round(probability, 4),

        # Detailed probabilities
        "mean_probability": round(mean_prob, 4),
        "max_probability": round(max_prob, 4),
        "confidence": confidence,
        "alert_level": alert_level,

        # Human-readable
        "ai_detected": "Abnormal patterns detected" if seizure_alert else "Normal patterns",
        "analysis_time": now.strftime("%I:%M %p, %b %d"),

        # File info
        "file_duration_sec": round(n_samples / sample_rate, 1),
        "file_duration_min": round(n_samples / sample_rate / 60, 2),

        # Segment info
        "segments_analyzed": len(probs),
        "segments_at_risk": at_risk,
        "risk_ratio": round(risk_ratio, 3),

        # Model info
        "n_models": len(_models),

        # No error
        "error": None,
    }


def _error_response(message: str) -> Dict[str, Any]:
    """Helper to create error response."""
    return {
        "error": message,
        "seizure_alert": False,
        "result": "Error",
        "probability": 0.0,
        "mean_probability": 0.0,
        "max_probability": 0.0,
        "confidence": 0.0,
        "alert_level": "unknown",
        "ai_detected": "Error occurred",
        "analysis_time": datetime.now().strftime("%I:%M %p, %b %d"),
        "file_duration_sec": 0.0,
        "file_duration_min": 0.0,
        "segments_analyzed": 0,
        "segments_at_risk": 0,
        "risk_ratio": 0.0,
        "n_models": 0,
    }

