from django.urls import path
from .views import upload_and_predict_edf

urlpatterns = [
    path("predict/", upload_and_predict_edf, name="upload_predict_edf"),
]
