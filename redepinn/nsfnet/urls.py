from django.urls import path

from . import views

urlpatterns = [

    path("", views.index , name="index"),
    path("train_1", views.train_1 , name="train_1")

]