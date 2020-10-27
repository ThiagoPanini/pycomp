"""
---------------------------------------------------
----- TÓPICO: Machine Learning - Transformers -----
---------------------------------------------------
Arquivo de testes para validar as implementações
presentes no módulo ml do pacote bebop

Sumário
-----------------------------------
1. Pipelines
-----------------------------------
"""

# Importando bibliotecacs
from ml.transformers import *
from sklearn.pipeline import Pipeline
import pandas as pd


"""
---------------------------------------------------
-------- 1. PIPELINES DE PRÉ-PROCESSAMENTO --------
---------------------------------------------------
"""

# Lendo dados para utilizar nos testes
data = pd.read_csv('tests/titanic_data/titanic.csv')
print(data.shape)
print(data.head())