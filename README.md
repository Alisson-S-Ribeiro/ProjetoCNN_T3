# ProjetoCNN_T3

Nome do Projeto: Implementação e Análise de Classificação com Redes Convolucionais e o dataset CUFS.

Descrição do Projeto: Este projeto abordou a classificação de imagens por meio de redes neurais convolucionais (CNNs), um dos métodos mais eficazes para problemas de visão computacional. O foco foi treinar, avaliar e ajustar modelos com diferentes combinações de hiperparâmetros, ultilizando o dataset CUHK Face Sketch Database (CUFS).

Instalação: Certifique-se de que você possui o Python instalado no seu ambiente,Clone o repositório , use o comando:
"bash
Copiar código
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_REPOSITORIO>."

Estrutura dos Arquivos: A pasta ProjetoRes_def.ipynb contém o código-fonte, enquanto a pasta "Relatório Técnico: Implementação e Análise de Classificação com Redes Convolucionais e o dataset CUFS" contém o relatório.

Tecnologias Utilizadas:
import os
import math
import json
import time
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pickle import encode_long
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, roc_curve, auc, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout.

Autores e Colaboradores: Alisson Santos Ribeiro(documentaçao e codigo) e Irlan Wallace dos Santos Mattos(documentaçao e codigo)
.
