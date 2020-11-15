# Importando módulo
from pycomp.ml.trainer import ClassificadorBinario

# Importando bibliotecas e lendo dados
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

raw_data = pd.read_csv('tests/ml/titanic_data/titanic.csv')
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

# Logistic Regression hyperparameters
logreg_param_grid = {
    'C': np.linspace(0.1, 10, 20),
    'penalty': ['l1', 'l2'],
    'class_weight': ['balanced', None],
    'random_state': [42],
    'solver': ['liblinear']
}

# Decision Trees hyperparameters
tree_param_grid = {
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 5, 10, 20],
    'max_features': np.arange(1, X_train.shape[1]),
    'class_weight': ['balanced', None],
    'random_state': [42]
}

# Random Forest hyperparameters
forest_param_grid = {
    'bootstrap': [True, False],
    'max_depth': [3, 5, 10, 20, 50],
    'n_estimators': [50, 100, 200, 500],
    'random_state': [42],
    'max_features': ['auto', 'sqrt'],
    'class_weight': ['balanced', None]
}

# Montando dicionário de classificadores a serme testados
set_classifiers = {
    'LogisticRegression': {
        'model': log_reg,
        'params': logreg_param_grid
    },
    'DecisionTree': {
        'model': tree_clf,
        'params': tree_param_grid
    },
    'RandomForest': {
        'model': forest_clf,
        'params': forest_param_grid
    }
}

# Variáveis de execução
OUTPUT_PATH = 'tests/ml/output'

# Inicializando objeto
trainer = ClassificadorBinario()

# Fluxo de treino
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path=OUTPUT_PATH, random_search=True)

# Análise gŕafica
trainer.visual_analysis(features=features, model_shap='DecisionTree', output_path=OUTPUT_PATH)

# Retornando informações relevantes de um modelo específico
model = trainer._get_estimator(model_name='RandomForest')
metrics = trainer._get_metrics(model_name='RandomForest')
model_info = trainer._get_model_info(model_name='RandomForest')
classifiers_info = trainer._get_classifiers_info()


"""
Ideias: salvar o .pkl dos modelos (nova pasta chamada models com os .pkl) [DONE]
Modificar gráficos pra sempre plotar visões pra todos os modelos presentes na classe [DONE]

Criar método para plotagem gráfica dos scores [DONE]
"""