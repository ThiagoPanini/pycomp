# Importando módulo
from trainer import ClassificadorBinario

# Importando bibliotecas e lendo dados
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

raw_data = pd.read_csv('tests/titanic_data/titanic.csv')
raw_data.columns = [col.lower().strip() for col in raw_data.columns]
raw_data = raw_data.loc[:, ['survived', 'pclass', 'age', 'sibsp', 'fare']]
raw_data.dropna(inplace=True)

# Separando X e y
X = raw_data.drop('survived', axis=1)
y = raw_data['survived'].values

# Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
features = list(X_train.columns)

# Preparando classificadores
tree_clf = DecisionTreeClassifier()
log_reg = LogisticRegression()
forest_clf = RandomForestClassifier()

set_classifiers = {
    'DecisionTree': {
        'model': tree_clf,
        'params': None
    },
    'LogisticRegression': {
        'model': log_reg,
        'params': None
    },
    'RandomForest': {
        'model': forest_clf,
        'params': None
    }
}

# Variáveis de execução
OUTPUT_PATH = 'tests/ml/output'

# Inicializando objeto
trainer = ClassificadorBinario()

# Fluxo de treino
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path=OUTPUT_PATH)

# Análise gŕafica
print()
trainer.visual_analysis(features=features, output_path=OUTPUT_PATH)


"""
Ideias: salvar o .pkl dos modelos (nova pasta chamada models com os .pkl) [DONE]
Modificar gráficos pra sempre plotar visões pra todos os modelos presentes na classe [DONE]

Criar método para plotagem gráfica dos scores [DONE]
"""