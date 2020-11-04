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
$ pip install pycomp --upgrade
```

> **Nota**: o pacote `pycomp` é construído como uma ferramenta de top level em cima de outros pacotes conhecidos em Python, como sklearn, pandas e numpy. Ao instalar o `pycomp`, as dependências especificadas também serão instaladas automaticamente em seu ambiente virtual de trabalho.

Output esperado:

```
Collecting pycomp
  Downloading https://files.pythonhosted.org/packages/36/a2/84a5c31da4041579285d0b959658db297c05bbcd5e4e5c6a29434b011c41/pycomp-0.0.11-py3-none-any.whl
Collecting numpy==1.19.3 (from pycomp)
  Using cached https://files.pythonhosted.org/packages/65/b3/07864c89acb2a86df6f2e8c9bf091ec5916da58dd3ce3a633a51a02c115e/numpy-1.19.3-cp37-cp37m-manylinux2010_x86_64.whl
Collecting scikit-learn==0.23.2 (from pycomp)
  Using cached https://files.pythonhosted.org/packages/f4/cb/64623369f348e9bfb29ff898a57ac7c91ed4921f228e9726546614d63ccb/scikit_learn-0.23.2-cp37-cp37m-manylinux1_x86_64.whl
Collecting pandas==1.1.3 (from pycomp)
  Using cached https://files.pythonhosted.org/packages/25/47/22fc373440e144e2111363adaa07abb09ec1f03fbc071b6d9fc0bbf65f68/pandas-1.1.3-cp37-cp37m-manylinux1_x86_64.whl
Collecting joblib==0.17.0 (from pycomp)
  Using cached https://files.pythonhosted.org/packages/fc/c9/f58220ac44a1592f79a343caba12f6837f9e0c04c196176a3d66338e1ea8/joblib-0.17.0-py3-none-any.whl
Collecting scipy>=0.19.1 (from scikit-learn==0.23.2->pycomp)
  Using cached https://files.pythonhosted.org/packages/fa/cf/94686c3e2b21cba82904a2bbb014f7529d483021802a0116c3a256b00563/scipy-1.5.3-cp37-cp37m-manylinux1_x86_64.whl
Collecting threadpoolctl>=2.0.0 (from scikit-learn==0.23.2->pycomp)
  Using cached https://files.pythonhosted.org/packages/f7/12/ec3f2e203afa394a149911729357aa48affc59c20e2c1c8297a60f33f133/threadpoolctl-2.1.0-py3-none-any.whl
Collecting python-dateutil>=2.7.3 (from pandas==1.1.3->pycomp)
  Using cached https://files.pythonhosted.org/packages/d4/70/d60450c3dd48ef87586924207ae8907090de0b306af2bce5d134d78615cb/python_dateutil-2.8.1-py2.py3-none-any.whl
Collecting pytz>=2017.2 (from pandas==1.1.3->pycomp)
  Downloading https://files.pythonhosted.org/packages/12/f8/ff09af6ff61a3efaad5f61ba5facdf17e7722c4393f7d8a66674d2dbd29f/pytz-2020.4-py2.py3-none-any.whl (509kB)
     |████████████████████████████████| 512kB 1.7MB/s 
Collecting six>=1.5 (from python-dateutil>=2.7.3->pandas==1.1.3->pycomp)
  Using cached https://files.pythonhosted.org/packages/ee/ff/48bde5c0f013094d729fe4b0316ba2a24774b3ff1c52d924a8a4cb04078a/six-1.15.0-py2.py3-none-any.whl
Installing collected packages: numpy, scipy, joblib, threadpoolctl, scikit-learn, six, python-dateutil, pytz, pandas, pycomp
Successfully installed joblib-0.17.0 numpy-1.19.3 pandas-1.1.3 pycomp-0.0.11 python-dateutil-2.8.1 pytz-2020.4 scikit-learn-0.23.2 scipy-1.5.3 six-1.15.0 threadpoolctl-2.1.0
```

## Utilização
Abaixo, será descrito um exemplo de utilização dos módulos `ml.transformers` e `ml.trainer` do pacote `pycomp` através da leitura, preparação e treinamento de um modelo de classificação utilizando o dataset [Titanic](https://www.kaggle.com/c/titanic)

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

# Treinando e avaliando modelos de classificação
features = list(X_train.columns)
trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features, output_path='results')
```

Output esperado do trecho de código acima no prompt:
```
DEBUG;2020-11-02 02:11:40;trainer.py;trainer;157;Treinando modelo DecisionTree
INFO;2020-11-02 02:11:40;trainer.py;trainer;177;Modelo DecisionTree treinado com sucesso
DEBUG;2020-11-02 02:11:40;trainer.py;trainer;157;Treinando modelo LogisticRegression
INFO;2020-11-02 02:11:41;trainer.py;trainer;177;Modelo LogisticRegression treinado com sucesso
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;212;Computando métricas do modelo DecisionTree utilizando validação cruzada com 5 K-folds
INFO;2020-11-02 02:11:41;trainer.py;trainer;244;Métricas computadas com sucesso nos dados de treino em 0.167 segundos
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;277;Computando métricas do modelo DecisionTree utilizando dados de teste
INFO;2020-11-02 02:11:41;trainer.py;trainer;306;Métricas computadas com sucesso nos dados de teste em 0.005 segundos
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;212;Computando métricas do modelo LogisticRegression utilizando validação cruzada com 5 K-folds
INFO;2020-11-02 02:11:41;trainer.py;trainer;244;Métricas computadas com sucesso nos dados de treino em 0.389 segundos
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;277;Computando métricas do modelo LogisticRegression utilizando dados de teste
INFO;2020-11-02 02:11:41;trainer.py;trainer;306;Métricas computadas com sucesso nos dados de teste em 0.006 segundos
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;92;Salvando arquivo
INFO;2020-11-02 02:11:41;trainer.py;trainer;96;Arquivo salvo em: results/metrics.csv
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;411;Extraindo importância das features para o modelo DecisionTree
INFO;2020-11-02 02:11:41;trainer.py;trainer;429;Extração da importância das features concluída com sucesso para o modelo DecisionTree
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;411;Extraindo importância das features para o modelo LogisticRegression
WARNING;2020-11-02 02:11:41;trainer.py;trainer;415;Modelo LogisticRegression não possui o método feature_importances_
DEBUG;2020-11-02 02:11:41;trainer.py;trainer;92;Salvando arquivo
INFO;2020-11-02 02:11:41;trainer.py;trainer;96;Arquivo salvo em: results/top_features.csv
```

Como feature bônus, o `pycomp` permite aos usuários o armazenamento automático de logs de execução das funções, proporcionando uma análise mais detalhada dos passos tomados. Na prática, o código utilizado no exemplo cria dois novos diretórios no mesmo diretório de localização do script:
  - `exec_log/` contendo o arquivo `execution_log.log` com o log de execução do treinamento dos modelos;
  - `results/` contendo os arquivos `metrics.csv` (resultado das métricas dos modelos) e `top_features.csv` (análise de feature importance dos modelos). Ambos são definidos no código-exemplo.

## Próximos Passos
- [ ] Inserir função para plotagem de curva ROC (`trainer.py`)
- [ ] Inserir função para plotagem de matriz de confusão (`trainer.py`)
- [ ] Inserir função para plotagem de curva de aprendizado (`trainer.py`)
- [ ] Inserir função para plotagem de curva de distribuição de scores (`trainer.py`)
- [ ] Inserir função para análise shap dos modelos treinados (`trainer.py`)
- [ ] Consolidar função `graphic_evaluation()` para gerar todas as análises acima (`trainer.py`)
- [ ] Brainstorming para pipelines automáticos de prep + treino (`transformers.py + trainer.py`)
- [ ] Inserir GIF de demonstração do projeto


## Referências

