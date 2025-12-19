# backend/ai/views.py
import traceback
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

# Import your predictor function (ensure simple_edf_predictor.py is on PYTHONPATH)
from .predictor2 import predict_seizure

# Helper: convert numpy scalars/arrays to plain Python types
def _to_plain(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_plain(v) for v in obj)
    if isinstance(obj, _np.generic):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_and_predict_edf(request):
    """
    Accepts multipart/form-data with key 'file'.
    Reads the file into memory (bytes) and passes it to predict_seizure.
    Returns the predictor dict as JSON.
    """
    uploaded_file = request.FILES.get("file")
    if uploaded_file is None:
        return Response({"error": "No file provided under key 'file'."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Read entire EDF into memory (bytes)
        file_bytes = uploaded_file.read()
        print(f"Received EDF: {uploaded_file.name} ({len(file_bytes)} bytes)")

        # Call the predictor (your function accepts bytes or path)
        result = predict_seizure(file_bytes)

        # Convert numpy types -> JSON serializable Python types
        result = _to_plain(result)

        return Response(result, status=status.HTTP_200_OK)

    except Exception as exc:
        # Print full traceback to server logs for debugging
        traceback.print_exc()
        return Response({
            "error": f"Internal server error: {str(exc)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
