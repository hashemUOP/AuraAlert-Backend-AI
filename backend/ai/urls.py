from django.urls import path
from .views import upload_edf

urlpatterns = [
    path('predict/', upload_edf),
]
