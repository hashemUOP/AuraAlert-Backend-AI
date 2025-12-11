"""
Seizure Prediction Inference

This module loads the trained ensemble models and runs predictions.

Usage:
    from predictor import SeizurePredictor
    from preprocessing import preprocess_single_segment
    
    # Initialize (loads 7 models)
    predictor = SeizurePredictor(models_dir="path/to/checkpoints")
    
    # Preprocess and predict
    segment = preprocess_single_segment(raw_eeg_data)
    result = predictor.predict(segment)
    
    # Check result
    if result["seizure_alert"]:
        print("WARNING: Seizure risk detected!")
    else:
        print("Patient is stable")
"""

import os
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class SeizurePredictor:
    """
    Ensemble seizure predictor using 7 LOOCV-trained models.
    
    Models are averaged together for robust predictions.
    """
    
    # Threshold for seizure alert (0.5 = 50%)
    THRESHOLD = 0.5
    
    # Number of predictions to average for smoothing
    SMOOTHING_WINDOW = 3
    
    def __init__(self, models_dir: str):
        """
        Initialize predictor and load models.
        
        Args:
            models_dir: Path to directory containing .keras model files
        """
        self.models_dir = Path(models_dir)
        self.models: List[Any] = []
        self._history: List[float] = []
        
        self._load_models()
    
    def _load_models(self):
        """Load all 7 LOOCV models."""
        import tensorflow as tf
        
        print(f"Loading models from: {self.models_dir}")
        
        for i in range(7):
            model_path = self.models_dir / f"best_model_patientchb01_seizure{i}_preictal60_run42.keras"
            
            if not model_path.exists():
                print(f"WARNING: Model not found: {model_path}")
                continue
            
            print(f"  Loading model {i+1}/7...")
            model = tf.keras.models.load_model(str(model_path))
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models")
        
        if len(self.models) == 0:
            raise RuntimeError("No models loaded! Check models_dir path.")
    
    def predict(self, segment: np.ndarray, apply_smoothing: bool = True) -> Dict[str, Any]:
        """
        Run seizure prediction on preprocessed EEG segment.
        
        Args:
            segment: Preprocessed EEG, shape (1, 1280, 23, 1)
            apply_smoothing: Average with recent predictions for stability
        
        Returns:
            Dictionary with prediction results
        """
        # Ensure correct shape
        if segment.ndim == 3:
            segment = segment[np.newaxis, ...]
        
        # Get prediction from each model
        predictions = []
        for model in self.models:
            prob = model.predict(segment, verbose=0)[0][0]
            predictions.append(float(prob))
        
        # Average all models (ensemble)
        raw_prob = np.mean(predictions)
        
        # Apply temporal smoothing
        if apply_smoothing:
            self._history.append(raw_prob)
            if len(self._history) > self.SMOOTHING_WINDOW:
                self._history = self._history[-self.SMOOTHING_WINDOW:]
            probability = np.mean(self._history)
        else:
            probability = raw_prob
        
        # Determine if seizure alert
        seizure_alert = probability >= self.THRESHOLD
        
        # Calculate confidence (distance from 0.5 threshold)
        confidence = abs(probability - 0.5) * 200  # 0-100%
        
        # Get alert level
        if probability >= 0.9:
            alert_level = "critical"
        elif probability >= 0.7:
            alert_level = "high"
        elif probability >= 0.5:
            alert_level = "medium"
        else:
            alert_level = "low"
        
        # Build response
        now = datetime.now()
        
        return {
            # Primary fields for app UI
            "seizure_alert": seizure_alert,  # True = danger, False = safe
            "result": "At Risk" if seizure_alert else "Stable",
            "confidence": round(confidence, 1),
            "ai_detected": "Abnormal brainwave patterns detected" if seizure_alert 
                          else "Normal brainwave patterns",
            "analysis_time": now.strftime("%I:%M %p, %b %d"),
            "analysis_timestamp": now.isoformat(),
            
            # Detailed info
            "probability": round(probability, 4),
            "raw_probability": round(raw_prob, 4),
            "alert_level": alert_level,
            "individual_predictions": [round(p, 4) for p in predictions],
            "n_models": len(self.models),
        }
    
    def reset_history(self):
        """Clear prediction history (call when starting new session)."""
        self._history = []
    
    def predict_all(self, segments: np.ndarray, apply_smoothing: bool = True) -> Dict[str, Any]:
        """
        Predict on ALL segments from a full file.
        
        Args:
            segments: Shape (num_segments, 1280, 23, 1) from preprocess_eeg()
            apply_smoothing: Apply temporal smoothing
        
        Returns:
            Dictionary with predictions for all segments
        """
        self.reset_history()
        
        predictions = []
        alert_segments = []
        max_prob = 0.0
        
        for i, segment in enumerate(segments):
            # Add batch dimension
            result = self.predict(segment[np.newaxis, ...], apply_smoothing)
            
            # Calculate time
            start_sec = i * 5
            
            pred = {
                "segment": i,
                "time_seconds": start_sec,
                "seizure_alert": result["seizure_alert"],
                "probability": result["probability"],
                "result": result["result"],
                "alert_level": result["alert_level"],
            }
            predictions.append(pred)
            
            if result["seizure_alert"]:
                alert_segments.append(i)
            max_prob = max(max_prob, result["probability"])
        
        return {
            "total_segments": len(predictions),
            "total_duration_seconds": len(predictions) * 5,
            "predictions": predictions,
            "summary": {
                "total_alerts": len(alert_segments),
                "max_probability": round(max_prob, 4),
                "alert_segments": alert_segments,
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get predictor status info."""
        return {
            "models_loaded": len(self.models),
            "models_dir": str(self.models_dir),
            "threshold": self.THRESHOLD,
            "smoothing_window": self.SMOOTHING_WINDOW,
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test with random data
    print("=" * 50)
    print("Seizure Predictor Test")
    print("=" * 50)
    
    # Set your models directory
    models_dir = os.environ.get("MODELS_DIR", "./checkpoints")
    
    try:
        predictor = SeizurePredictor(models_dir)
        
        # Create fake EEG data (5 seconds, 23 channels)
        fake_eeg = np.random.randn(1, 1280, 23, 1).astype(np.float32)
        
        result = predictor.predict(fake_eeg)
        
        print("\nPrediction Result:")
        print(f"  Seizure Alert: {result['seizure_alert']}")
        print(f"  Result: {result['result']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Probability: {result['probability']}")
        print(f"  Alert Level: {result['alert_level']}")
        print(f"  AI Detected: {result['ai_detected']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure MODELS_DIR environment variable points to your checkpoints folder")
