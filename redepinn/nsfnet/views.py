from django.shortcuts import render
from django.db import models
# Create your views here.

from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
# RESOLVER A QUESTÃO DA APRESENTAÇÃO DOS DADOS
def index(request):
    return HttpResponse("Página inicial NSFNET")

def train_1(request):
    return HttpResponse(models.Dataset(models.Model))