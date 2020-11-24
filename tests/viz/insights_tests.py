# Importando módulo
from pycomp.viz.insights import *
import pandas as pd
import os


# Definindo variáveis do projeto
DATA_PATH = 'tests/ml/titanic_data/'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
OUTPUT_PATH = 'tests/viz/output'

# Lendo base de treino e verificando conteúdo
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))

# Plotando e salvando gráficos de rosca
plot_donut_chart(df=df, col='Survived', label_names=['Vítimas', 'Sobreviventes'], colors=['lightcoral', 'lightskyblue'], 
                 save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Embarked', save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Sex', save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Pclass', colors=['brown', 'gold', 'silver'], save=True, output_path=OUTPUT_PATH)

# Plotando e salvando gráficos de pizza
plot_pie_chart(df=df, col='Survived', label_names=['Vítimas', 'Sobreviventes'], colors=['crimson', 'navy'],
               explode=(0.02, 0), save=True, output_path=OUTPUT_PATH)