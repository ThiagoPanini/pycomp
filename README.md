<h1 align="center">pycomp</h1>

<div align="center">
  :snake::factory::factory::factory::factory::factory::factory::snake:
</div>
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
  - _arquivos.py_
    - _valida_arquivo_origem()*_
    - _valida_dt_mod_arquivo()*_
    - _copia_arquivo()*_
    - _controle_de_diretorio()*_
  
- :pencil: __log__: módulo auxiliar responsável por facilitar a geração, configuração e o armazenamento de logs de execução dos demais módulos do pacote.
  - _log_config.py_
    - _log_config()*_
  
- :robot: __ml__: o módulo ml (machine learning) contém os componentes apropriados para a construção e aplicação de Pipelines de pré-processamento de dados, bem como módulos responsáveis por automatizar o treinamento e avaliação de modelos de aprendizado de máquina. Através dos módulos _transformers_ e _trainer_, é possível construir um fluxo inteligente de recebimento, transformação e treinamento de modelos.
  - _transformers.py_
    - _ColsFormatting()*_
    - _FeatureSelection()*_
    - _TargetDefinition()*_
    - _DropDuplicates()*_
    - _SplitData()*_
    - _DummiesEncoding()*_
    - _FillNullData()*_
    - _DropNullData()*_
    - _TopFeaturesSelector()*_
  
- :thought_balloon: __Em andamento...__

A fábrica está a todo vapor! Sua capacidade de produção e seu leque de fornecimento pode ser resumido em:

| Tópico                     | Módulo                   | Funções           | Classes         | Componentes Totais  | Homologados     |
| -------------------------- | :---------------:        | :---------------: | :-------------: | :-----------------: | :-------------: |
| File System                | `pycomp.fs.arquivos`     |         4         |        0        |        4            |        0        |
| Logs                       | `pycomp.log.log_config`  |         1         |        0        |        1            |        0        |
| Machine Learning           | `pycomp.ml.transformers` |         0         |        9        |        9            |        0        |
|                            | `pycomp.ml.trainer`      |         7         |        1        |        8            |        0        |


## Utilização

A última versão do pacote `pycomp` encontra-se publicada no repositório [PyPI](https://pypi.org/project/pycomp/) e pode ser instalada na linha de comando utilizando:
```bash
$ pip install pycomp --upgrade
```



## Próximos Passos
- [ ] Inserir função para plotagem de curva ROC (`trainer.py`)
- [ ] Inserir função para plotagem de matriz de confusão (`trainer.py`)
- [ ] Inserir função para plotagem de curva de aprendizado (`trainer.py`)
- [ ] Inserir função para plotagem de curva de distribuição de scores (`trainer.py`)
- [ ] Inserir função para análise shap dos modelos treinados (`trainer.py`)
- [ ] Brainstorming para pipelines automáticos de prep + treino `transformers.py + trainer.py`
- [ ] Inserir GIF de demonstração do projeto
