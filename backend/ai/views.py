from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser])
def upload_edf(request):
    """
    Accepts an uploaded EDF file via 'file' key and keeps it in memory.
    """
    uploaded_file = request.FILES.get("file")
    if uploaded_file is None:
        return Response({"error": "No file provided under key 'file'."}, status=status.HTTP_400_BAD_REQUEST)

    # Read entire file into a variable (binary)
    file_bytes = uploaded_file.read()  # type: bytes

    # Example: print first 200 bytes in hex for debugging
    print(f"Received EDF file: {uploaded_file.name}")
    print(f"File size: {len(file_bytes)} bytes")
    print(f"First 200 bytes (hex): {file_bytes[:200].hex()}")

    # Now file_bytes contains the full EDF file in memory
    # You can pass it to your processing function:
    # result = predict_seizure_from_edf_bytes(file_bytes)

    return Response({
        "status": "ok",
        "filename": uploaded_file.name,
        "size": len(file_bytes)
    }, status=status.HTTP_201_CREATED)
