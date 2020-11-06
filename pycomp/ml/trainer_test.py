# Importando módulo
from trainer import ClassificadorBinario

# Importando bibliotecas e lendo dados
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
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

# Variáveis de execução
OUTPUT_FILES = 'tests/ml/results/files'
OUTPUT_IMGS = 'tests/ml/results/imgs'

# Inicializando objeto
trainer = ClassificadorBinario()

# Fluxo de treino
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path=OUTPUT_FILES)

# Gerando Matriz de Confusão
print()
trainer.plot_confusion_matrix(output_path=OUTPUT_IMGS)

# Plotando curva ROC
print()
trainer.plot_roc_curve(output_path=OUTPUT_IMGS)

# Plotando gráfico de distribuição de score
print()
trainer.plot_score_distribution(output_path=OUTPUT_IMGS)

# Plotando gráfico de distribuição de score
print()
trainer.plot_score_bins(output_path=OUTPUT_IMGS)

# Plotando análise shap
print()
trainer.shap_analysis(model_name='LogisticRegression', features=features)