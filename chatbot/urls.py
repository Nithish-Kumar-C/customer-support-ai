from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),           # GET  / → chat UI
    path('ask/', views.ask, name='ask'),           # POST /ask/ → get AI answer
    path('upload/', views.upload_pdf, name='upload'),  # POST /upload/ → upload PDF
]
