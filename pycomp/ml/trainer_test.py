# Importando módulo
from trainer import ClassificadorBinario

# Importando bibliotecas e lendo dados
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('tests/titanic_data/titanic.csv')
raw_data.columns = [col.lower().strip() for col in raw_data.columns]
raw_data = raw_data.loc[:, ['survived', 'pclass', 'age', 'sibsp', 'fare']]
raw_data.dropna(inplace=True)

# Separando X e y
X = raw_data.drop('survived', axis=1)
y = raw_data['survived'].values

# Separando em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)

# Preparando classificadores
tree_clf = DecisionTreeClassifier()
log_reg = LogisticRegression()

set_classifiers = {
    'DecisionTree': {
        'model': tree_clf,
        'params': None
    },
    'LogisticRegression': {
        'model': log_reg,
        'params': None
    }
}

# Inicializando objeto
trainer = ClassificadorBinario()

# Treinando modelo
trainer.fit(set_classifiers, X_train, y_train)

# Avaliando modelo
metrics = trainer.evaluate_performance(X_train, y_train, X_test, y_test, save=True, overwrite=True, path='tests/ml/results/metrics.csv')

# Retornando features mais importantes
features = list(X_train.columns)
top_features = trainer.feature_importance(features, save=True, overwrite=True, path='tests/ml/results/top_features.csv')

# -----------------------------------------------
print()
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path='tests/ml/results')
