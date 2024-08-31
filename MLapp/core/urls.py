from django.urls import path

from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("predicted_artist", views.predicted_artist, name="predicted_artist"),

]