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
Installing collected packages: numpy, scipy, pillow, six, cycler, certifi, python-dateutil, pyparsing, kiwisolver, matplotlib, pytz, pandas, seaborn, joblib, threadpoolctl, scikit-learn, pycomp
Successfully installed certifi-2020.6.20 cycler-0.10.0 joblib-0.17.0 kiwisolver-1.3.1 matplotlib-3.3.2 numpy-1.19.3 pandas-1.1.3 pillow-8.0.1 pycomp-0.0.12 pyparsing-2.4.7 python-dateutil-2.8.1 pytz-2020.4 scikit-learn-0.23.2 scipy-1.5.4 seaborn-0.11.0 six-1.15.0 threadpoolctl-2.1.0
```

## Utilização
Para demonstrar uma poderosa aplicação do pacote `pycomp`, será exemplificado abaixo um trecho de código responsável por consolidar um Pipeline completo de preparação de dados, treinamento e avaliação de diferentes modelos de classificação frente a uma determinada task. Como base de dados, será utilizado o dataset [Titanic](https://www.kaggle.com/c/titanic) em um formato único (união dos arquivos `train.csv` e `test.csv`). Em termos de ferramental, serão aplicadas funções e métodos referentes aos módulos `pycomp.ml.transformers` e `pycomp.ml.trainer`.

```python
# Importando bibliotecas
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pycomp.ml.transformers import FormataColunas, FiltraColunas, DefineTarget, EliminaDuplicatas, PreencheDadosNulos, SplitDados
from pycomp.ml.trainer import ClassificadorBinario

# Lendo base de dados (titanic data - train + test)
df = pd.read_csv('titanic.csv')
cols_filter = ['survived', 'pclass', 'age', 'sibsp', 'fare']

# Pipeline da primeira camada
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

# Definindo variáveis de execução
OUTPUT_PATH = 'results/training_results'

# Inicializando objeto
trainer = ClassificadorBinario()

# Treinando e avaliando modelos de classificação
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path=OUTPUT_PATH)

# Gerando gráficos de matriz de confusão para os classificadores
trainer.plot_confusion_matrix(output_path=OUTPUT_PATH)

# Gerando gráficos de curva ROC para os classificadores
trainer.plot_roc_curve(output_path=OUTPUT_PATH)
```

Ao utilizar as ferramentas disponibilizadas no módulo `ml` do pacote `pycomp`, o usuário consegue facilmente construir e executar um Pipeline de preparação de dados enxuto e otimizado a partir das classes pré definidas no módulo `transformers`. Em complemento a essa feature, o módulo `trainer` traz consigo a classe `ClassificadorBinario` com o objetivo de facilitar o treinamento e avaliação de classificadores binários. O usuário final necessita apenas fornecer uma base de dados como de input, os _estimators_ (modelos a serem treinados) e seus respectivos hyperparâmetros de busca a serem utilizados no processo. Detalhando um pouco mais os métodos utilizados no exemplo, tem-se:
- `training_flow()`: treinamento e avaliação (treino e teste) de todos os classificadores passados como argumento da classe;
- `plot_confusion_matrix()`: geração e salvamento de matriz de confusão (treino e teste) para os classificadores analisados;
- `plot_roc_curve()`: geração e salvamento de imagem para curvas ROC (treino e teste) dos classificadores analisados.

### Outputs
Como feature adicional do `pycomp`, um objeto logger da biblioteca `logging` é instanciado automaticamente e utilizado nas definições do pacote, gerando assim um arquivo `exec_log/execution_log.log` no mesmo diretório de execução do script com os detalhes de cada passo dado nas funções e métodos aplicados. É esperado que o exemplo descrito acima gere a seguinte saída no cmd do OS ou a IDE:
```
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;162;Treinando modelo DecisionTree
INFO;2020-11-05 00:07:14;trainer.py;trainer;182;Modelo DecisionTree treinado com sucesso
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;162;Treinando modelo LogisticRegression
INFO;2020-11-05 00:07:14;trainer.py;trainer;182;Modelo LogisticRegression treinado com sucesso
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;217;Computando métricas do modelo DecisionTree utilizando validação cruzada com 5 K-folds
INFO;2020-11-05 00:07:14;trainer.py;trainer;249;Métricas computadas com sucesso nos dados de treino em 0.142 segundos
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;282;Computando métricas do modelo DecisionTree utilizando dados de teste
INFO;2020-11-05 00:07:14;trainer.py;trainer;311;Métricas computadas com sucesso nos dados de teste em 0.008 segundos
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;217;Computando métricas do modelo LogisticRegression utilizando validação cruzada com 5 K-folds
INFO;2020-11-05 00:07:14;trainer.py;trainer;249;Métricas computadas com sucesso nos dados de treino em 0.341 segundos
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;282;Computando métricas do modelo LogisticRegression utilizando dados de teste
INFO;2020-11-05 00:07:14;trainer.py;trainer;311;Métricas computadas com sucesso nos dados de teste em 0.007 segundos
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;97;Salvando arquivo
INFO;2020-11-05 00:07:14;trainer.py;trainer;101;Arquivo salvo em: results/training_results/metrics.csv
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;416;Extraindo importância das features para o modelo DecisionTree
INFO;2020-11-05 00:07:14;trainer.py;trainer;434;Extração da importância das features concluída com sucesso para o modelo DecisionTree
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;416;Extraindo importância das features para o modelo LogisticRegression
WARNING;2020-11-05 00:07:14;trainer.py;trainer;420;Modelo LogisticRegression não possui o método feature_importances_
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;97;Salvando arquivo
INFO;2020-11-05 00:07:14;trainer.py;trainer;101;Arquivo salvo em: results/training_results/top_features.csv
DEBUG;2020-11-05 00:07:14;trainer.py;trainer;562;Inicializando plotagem da matriz de confusão para os modelos
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;570;Retornando dados de treino e teste para o modelo DecisionTree
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;583;Realizando predições para os dados de treino e teste (DecisionTree)
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;591;Gerando matriz de confusão para o modelo DecisionTree
INFO;2020-11-05 00:07:15;trainer.py;trainer;604;Matriz de confusão gerada para o modelo DecisionTree
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;570;Retornando dados de treino e teste para o modelo LogisticRegression
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;583;Realizando predições para os dados de treino e teste (LogisticRegression)
DEBUG;2020-11-05 00:07:15;trainer.py;trainer;591;Gerando matriz de confusão para o modelo LogisticRegression
INFO;2020-11-05 00:07:15;trainer.py;trainer;604;Matriz de confusão gerada para o modelo LogisticRegression
INFO;2020-11-05 00:07:16;trainer.py;trainer;614;Imagem com as matrizes salva com sucesso em results/training_results/confusion_matrix.png
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;641;Inicializando plotagem da curva ROC para os modelos
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;647;Retornando labels e scores de treino e de teste para o modelo DecisionTree
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;660;Calculando FPR, TPR e AUC de treino e teste para o modelo DecisionTree
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;673;Plotando curva ROC de treino e teste para o modelo DecisionTree
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;647;Retornando labels e scores de treino e de teste para o modelo LogisticRegression
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;660;Calculando FPR, TPR e AUC de treino e teste para o modelo LogisticRegression
DEBUG;2020-11-05 00:07:16;trainer.py;trainer;673;Plotando curva ROC de treino e teste para o modelo LogisticRegression
INFO;2020-11-05 00:07:17;trainer.py;trainer;703;Imagem com a curva ROC salva com sucesso em results/training_results/roc_curve.png
```

Ao definir um diretório de saída, a execução dos métodos irá gerar arquivos úteis para uma definitiva avaliação do melhor classificador para a respectiva tarefa. No exemplo acima, ao definir a variável `OUTPUT_PATH` como `'results/training_results'`, tem-se o seguinte resultado:
```bash
$ tree results/training_results/
results/training_results/
├── confusion_matrix.png
├── metrics.csv
├── roc_curve.png
└── top_features.csv
```

## Próximos Passos
- [x] Inserir função para plotagem de matriz de confusão (`trainer.py`)
- [x] Inserir função para plotagem de curva ROC (`trainer.py`)
- [x] Inserir função para plotagem de curva de distribuição de scores (`trainer.py`)
- [ ] Inserir função para plotagem de curva de aprendizado (`trainer.py`)
- [ ] Inserir função para análise shap dos modelos treinados (`trainer.py`)
- [ ] Consolidar função `graphic_evaluation()` para gerar todas as análises acima (`trainer.py`)
- [ ] Brainstorming para pipelines automáticos de prep + treino (`transformers.py + trainer.py`)
- [ ] Inserir GIF de demonstração do projeto


## Referências

