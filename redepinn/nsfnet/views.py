from django.shortcuts import render
import numpy as np
# Create your views here.

#carregando as bases de dados

train_inil = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')
train_iniv1 = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')
train_inip1 = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')
train_xb1 = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')
train_vb1 = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')
train_pb1 = np.load('C:/Users/User/NSFNET_de_Wita/datasets/train_iniv1.npy')


from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
# RESOLVER A QUESTÃO DA APRESENTAÇÃO DOS DADOS
def index(request):
    return HttpResponse("Página inicial NSFNET")

def train_1(request):
    # v1 = apps.get_model(model_name=models)
    return HttpResponse(train_inil)