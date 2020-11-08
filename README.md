<h1 align="center">
  <a href="https://pypi.org/project/pycomp/#description"><img src="https://i.imgur.com/WcAaq1P.png" alt="pycomp Logo"></a>
</h1>

<div align="center">
  <strong>Fábrica de componentes Python</strong>
</div>
<br/>

<div align="center">
  
![Release](https://img.shields.io/badge/release-ok-brightgreen)
[![PyPI](https://img.shields.io/pypi/v/pycomp?color=blueviolet)](https://pypi.org/project/pycomp/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pycomp?color=9cf)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycomp?color=green)
![PyPI - Status](https://img.shields.io/pypi/status/pycomp)

</div>
<br/>

Literalmente, uma fábrica de componentes em Python criada para auxiliar implementações, automações e até treinamento de modelos de Machine Learning! O objetivo desse repositório é propor uma forma mais fácil de se trabalhar com Python a partir do fornecimento de componentes prontos (funções e classes) para uma série de atividades rotineiras e exploratórias.

## Features
<sub>Componentes marcados com * não foram homologadas
- :file_folder: __fs__: o módulo fs (file system) foi desenvolvido para auxiliar o manuseio de arquivos em sistemas operacionais, como a validação da presença de um arquivo em um diretório, validação de atualização/modificação de um arquivo, cópia de um arquivo de uma origem para um destino, entre outros. Entre os componentes já implementados, é possível listar:
  - _arquivos.py*_
  
- :pencil: __log__: módulo auxiliar responsável por facilitar a geração, configuração e o armazenamento de logs de execução dos demais módulos do pacote.
  - _log_config.py*_
  
- :robot: __ml__: o módulo ml (machine learning) contém os componentes apropriados para a construção e aplicação de Pipelines de pré-processamento de dados, bem como módulos responsáveis por automatizar o treinamento e avaliação de modelos de aprendizado de máquina. Através dos módulos _transformers_ e _trainer_, é possível construir um fluxo inteligente de recebimento, transformação e treinamento de modelos.
  - _transformers.py*_
  - _trainer.py*_
  
- :thought_balloon: __Em andamento...__

A fábrica está a todo vapor! Sua capacidade de produção e seu leque de fornecimento pode ser resumido em:

| Tópico                     | Módulo                   | Funções           | Classes         | Componentes Totais  | Homologados     |
| -------------------------- | :---------------:        | :---------------: | :-------------: | :-----------------: | :-------------: |
| File System                | `pycomp.fs.arquivos`     |         4         |        0        |        4            |        0        |
| Logs                       | `pycomp.log.log_config`  |         1         |        0        |        1            |        0        |
| Machine Learning           | `pycomp.ml.transformers` |         0         |        9        |        9            |        0        |
|                            | `pycomp.ml.trainer`      |         7         |        1        |        8            |        0        |


## Instalação

A última versão do pacote `pycomp` encontra-se publicada no repositório [PyPI](https://pypi.org/project/pycomp/). 

> **Nota**: Como boa prática, recomenda-se a criação de um ambiente virual env Python para alocar as bibliotecas do projeto a ser desenvolvido. Caso não tenha um virtual env criado, o bloco de código abaixo pode ser utilizado para a criação de ambiente virtual em um diretório específico: 
```bash
# Criando diretório para o virtual env
$ cd $HOME  # ou qualquer outro diretório de preferência
$ mkdir dir_name/
$ cd dir_name/

# Criando ambiente virtual
$ python3 -m venv venv_name
```


Utilizando uma ferramenta de desenvolvimento (IDE ou a própria linha de comando), ative o ambiente virtual de trabalho:
```bash
$ source dir_name/venv_name/bin/activate
```

Após a ativação, é possível instalar o pacote `pycomp` via pip:
```bash
# Atualizando pip e instalando pycomp
$ python3 -m pip install --user --upgrade pip
$ pip install --upgrade pycomp
```

> **Nota**: o pacote `pycomp` é construído como uma ferramenta de top level em cima de outros pacotes conhecidos em Python, como sklearn, pandas e numpy. Ao instalar o `pycomp`, as dependências especificadas também serão instaladas automaticamente em seu ambiente virtual de trabalho.

Resumo do output esperado no cmd após a instalação do pacote::
```
Collecting pycomp
[...]
Installing collected packages: pyparsing, kiwisolver, certifi, six, python-dateutil, numpy, pillow, cycler, matplotlib, threadpoolctl, scipy, joblib, scikit-learn, pytz, pandas, seaborn, tqdm, slicer, llvmlite, numba, shap, pycomp
  Running setup.py install for numba ... done
  Running setup.py install for shap ... done
Successfully installed certifi-2020.6.20 cycler-0.10.0 joblib-0.17.0 kiwisolver-1.3.1 llvmlite-0.34.0 matplotlib-3.3.2 numba-0.51.2 numpy-1.19.3 pandas-1.1.3 pillow-8.0.1 pycomp-0.0.13 pyparsing-2.4.7 python-dateutil-2.8.1 pytz-2020.4 scikit-learn-0.23.2 scipy-1.5.4 seaborn-0.11.0 shap-0.37.0 six-1.15.0 slicer-0.0.3 threadpoolctl-2.1.0 tqdm-4.51.0
```

## Utilização
Para demonstrar uma poderosa aplicação do pacote `pycomp`, será exemplificado abaixo um trecho de código responsável por consolidar um Pipeline completo de preparação de dados, treinamento e avaliação de diferentes modelos de classificação frente a uma determinada task. Como base de dados, será utilizado o dataset [Titanic](https://www.kaggle.com/c/titanic) em um formato único (união dos arquivos `train.csv` e `test.csv`). Em termos de aprendizado de máquina, serão treinados e avaliados os modelos `LogisticRegression`, `DecisionTreeClassifier` e `RandomForestClassifiers`.

```python
# Importando bibliotecas
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings
filterwarnings('ignore')

from pycomp.ml.transformers import FormataColunas, FiltraColunas, DefineTarget, EliminaDuplicatas, PreencheDadosNulos, SplitDados
from pycomp.ml.trainer import ClassificadorBinario

# Lendo base de dados (titanic data - train + test)
df = pd.read_csv('titanic.csv')
cols_filter = ['survived', 'pclass', 'age', 'sibsp', 'fare']

# Pipeline da primeira camada (utilizando classes do módulo pycomp.ml.transformers)
first_layer_pipe = Pipeline([
    ('formatter', FormataColunas()),
    ('selector', FiltraColunas(features=cols_filter)),
    ('target_generator', DefineTarget(target_col='survived', pos_class=1.0)),
    ('dup_dropper', EliminaDuplicatas()),
    ('na_filler', PreencheDadosNulos(value_fill=0)),
    ('splitter', SplitDados(target='target'))
])

# Executando pipeline
X_train, X_test, y_train, y_test = first_layer_pipe.fit_transform(df)
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

# Definindo variáveis de execução
OUTPUT_PATH = 'output/'

# Inicializando objeto
trainer = ClassificadorBinario()

# Treinando e avaliando modelos de classificação
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path=OUTPUT_PATH)

# Gerando análises gráficas para os modelos
trainer.visual_analysis(features, output_path=OUTPUT_PATH)
```

Ao utilizar as ferramentas disponibilizadas no módulo `ml` do pacote `pycomp`, o usuário consegue facilmente construir e executar um Pipeline de preparação de dados enxuto e otimizado a partir das classes pré definidas no módulo `transformers`. Em complemento a essa feature, o módulo `trainer` traz consigo a classe `ClassificadorBinario` com o objetivo de facilitar o treinamento e avaliação de classificadores binários. O usuário final necessita apenas fornecer uma base de dados como de input, os _estimators_ (modelos a serem treinados) e seus respectivos hyperparâmetros de busca a serem utilizados no processo. 

No código, após a criação do objeto `trainer` da classe `ClassificadorBinario`, a simples execução de dois métodos abre um vasto leque de possibilidades de treinamento e avaliação de modelos. Tais métodos são:
- `training_flow()`: treinamento e avaliação (treino e teste) de todos os classificadores passados como argumento da classe;
- `visual_analysis()`: construção de plotagens gráficas a serme utilizadas na avalidação e validação dos classificadores treinados

### Outputs
Como feature adicional do `pycomp`, um objeto logger da biblioteca `logging` é instanciado automaticamente e utilizado nas definições do pacote, gerando assim um arquivo `exec_log/execution_log.log` no mesmo diretório de execução do script com os detalhes de cada passo dado nas funções e métodos aplicados. É esperado que o exemplo descrito acima mostre, no cmd ou na IDE utilizada, uma saída próxima a:
```
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;234;Treinando modelo DecisionTree
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;258;Salvando arquivo pkl do modelo DecisionTree treinado
WARNING;2020-11-07 12:52:13;trainer.py;trainer;132;Diretório output/models/ inexistente. Criando diretório no local especificado
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;139;Salvando modelo pkl no diretório especificado
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;234;Treinando modelo LogisticRegression
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;258;Salvando arquivo pkl do modelo LogisticRegression treinado
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;139;Salvando modelo pkl no diretório especificado
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;234;Treinando modelo RandomForest
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;258;Salvando arquivo pkl do modelo RandomForest treinado
DEBUG;2020-11-07 12:52:13;trainer.py;trainer;139;Salvando modelo pkl no diretório especificado
DEBUG;2020-11-07 12:52:14;trainer.py;trainer;293;Computando métricas do modelo DecisionTree utilizando validação cruzada com 5 K-folds
[...]
DEBUG;2020-11-07 12:52:31;trainer.py;trainer;1037;Plotando curvas de aprendizado de treino e validação para o modelo LogisticRegression
DEBUG;2020-11-07 12:52:31;trainer.py;trainer;1019;Retornando parâmetros pro modelo RandomForest e aplicando método learning_curve
DEBUG;2020-11-07 12:52:39;trainer.py;trainer;1037;Plotando curvas de aprendizado de treino e validação para o modelo RandomForest
DEBUG;2020-11-07 12:52:39;trainer.py;trainer;179;Salvando imagem no diretório especificado
INFO;2020-11-07 12:52:41;trainer.py;trainer;183;Imagem salva com sucesso em output/imgs/learning_curve.png
```

Ao definir um diretório de saída, a execução dos métodos irá gerar arquivos úteis para uma definitiva avaliação do melhor classificador para a respectiva tarefa. No exemplo acima, ao definir a variável `OUTPUT_PATH` como `'output/'`, tem-se o seguinte resultado:
```bash
$ tree output/
output
├── imgs
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── learning_curve.png
│   ├── roc_curve.png
│   ├── score_bins_percent.png
│   ├── score_bins.png
│   └── score_distribution.png
├── metrics
│   ├── metrics.csv
│   └── top_features.csv
└── models
    ├── decisiontree.pkl
    ├── logisticregression.pkl
    └── randomforest.pkl
```

## Próximos Passos
- [x] Inserir função para plotagem de matriz de confusão (`trainer.py`)
- [x] Inserir função para plotagem de curva ROC (`trainer.py`)
- [x] Inserir função para plotagem de curva de distribuição de scores (`trainer.py`)
- [x] Inserir função para plotagem de curva de aprendizado (`trainer.py`)
- [ ] Inserir função para análise shap dos modelos treinados (`trainer.py`)
- [x] Consolidar função `visual_analysis()` para gerar todas as análises acima (`trainer.py`)
- [ ] Brainstorming para pipelines automáticos de prep + treino (`transformers.py + trainer.py`)
- [ ] Inserir GIF de demonstração do projeto


## Referências

