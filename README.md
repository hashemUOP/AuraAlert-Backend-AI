
## Files

| File | Description |
|------|-------------|
| `preprocessing.py` | EEG preprocessing (CAR, bandpass filter, segmentation) |
| `predictor.py` | Model loading and inference |
| `requirements.txt` | Python dependencies |

```python
import numpy as np
from backend.ai_model.preprocessing import preprocess_eeg
from backend.ai_model.predictor import SeizurePredictor

# 1. Initialize predictor ONCE at server startup
predictor = SeizurePredictor(models_dir="path/to/checkpoints")


# 2. In your API endpoint
def predict_full_file(request):
    json_data = json.loads(request.body)
    raw_eeg = np.array(json_data["data"], dtype=np.float32)  # (N, 23)

    # For full file
    segments = preprocess_eeg(raw_eeg)
    result = predictor.predict_all(segments)  # Returns all predictions

    # For single segment
    segment = preprocess_single_segment(raw_eeg)
    result = predictor.predict(segment)  # Returns one prediction
```




## Output Format

```json
{
  "seizure_alert": false,
  "result": "Stable",
  "confidence": 95.0,
  "ai_detected": "Normal brainwave patterns",
  "analysis_time": "11:45 AM, Dec 11",
  "probability": 0.025,
  "alert_level": "low"
}
```


