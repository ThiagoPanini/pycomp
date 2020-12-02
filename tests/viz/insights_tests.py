# Importando módulo
from pycomp.viz.insights import *
import pandas as pd
import os
from warnings import filterwarnings
filterwarnings('ignore')


# Definindo variáveis do projeto
DATA_PATH = 'tests/ml/titanic_data/'
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'
OUTPUT_PATH = 'tests/viz/output'

# Lendo base de treino e verificando conteúdo
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))

# Customizando gráfico de roscas com **kwargs
plot_donut_chart(df=df, col='Survived', label_names={0: 'Vítimas', 1:'Sobreviventes'}, 
                 colors=['crimson', 'darkslateblue'], title='Quantidade de Regitros por Sobrevivência ao Naufrágio',
                 save=True, output_path=OUTPUT_PATH)

# Plotando e salvando gráficos de pizza
plot_pie_chart(df=df, col='Survived', label_names=['Vítimas', 'Sobreviventes'], colors=['crimson', 'navy'],
               explode=(0.02, 0), save=True, output_path=OUTPUT_PATH)

# Plotando e salvando gráfico duplo de rosca              
plot_double_donut_chart(df=df, col1='Survived', col2='Sex', label_names_col1={0: 'Vítimas', 1:'Sobreviventes'}, 
                        label_names_col2={'male': 'Masculino', 'female':'Feminino'}, colors1=['silver', 'navy'], 
                        colors2=['lightcoral', 'lightskyblue'], save=True, output_path=OUTPUT_PATH)

# Plotando gráfico de barras (volumetria - countplot)
plot_countplot(df=df, col='Pclass', order=True, save=True, output_path=OUTPUT_PATH)

"""# Plotando e salvando gráficos de rosca
plot_donut_chart(df=df, col='Survived', label_names=['Vítimas', 'Sobreviventes'], colors=['crimson', 'navy'], 
                 save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Embarked', save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Sex', colors=['lightcoral', 'lightskyblue'], save=True, output_path=OUTPUT_PATH)
plot_donut_chart(df=df, col='Pclass', colors=['brown', 'gold', 'silver'], save=True, output_path=OUTPUT_PATH)

# Plotando e salvando gráficos de pizza
plot_pie_chart(df=df, col='Survived', label_names=['Vítimas', 'Sobreviventes'], colors=['crimson', 'navy'],
               explode=(0.02, 0), save=True, output_path=OUTPUT_PATH)

# Plotando e salvando gráfico duplo de rosca
plot_double_donut_chart(df=df, col1='Survived', col2='Sex', label_names_col1=['Vítimas', 'Sobreviventes'], 
                        colors1=['silver', 'navy'], colors2=['lightcoral', 'lightskyblue'], save=True, output_path=OUTPUT_PATH)"""