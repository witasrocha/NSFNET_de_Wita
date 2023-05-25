from django.db import models
import numpy as np
# Create your models here.

# RESOLVER ESSA QUESTÃO DO CARREGAMENTO DAS BASES DE DADOS
class Dataset(models.Model):
# Load Data
    def load_train_1(self):
        return np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')
        train_iniv1 = np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')
        train_inip1 = np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')
        train_xb1 = np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')
        train_vb1 = np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')
        train_pb1 = np.load('C:/Users/PRH01/PycharmProjects/test/NSFNET_de_Wita/datasets/train_iniv1.npy')



