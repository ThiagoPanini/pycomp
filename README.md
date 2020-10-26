<h1 align="center">Python Factory</h1>

<div align="center">
  :snake::factory::factory::factory::factory::factory::factory::snake:
</div>
<div align="center">
  <strong>Fábrica de componentes Python</strong>
</div>

<br />

Literalmente, uma fábrica de códigos em Python criada para auxiliar implementações, automações e até treinamento de modelos de Machine Learning! O objetivo desse repositório é propor uma forma mais fácil de se trabalhar com Python a partir do fornecimento de componentes prontos (funções e classes) para uma série de atividades rotineiras e exploratórias.

## Table of Contents
- [Features](#features)
- [Utilização](#utilização)

## Features
<sub>Componentes marcados com * não foram homologadas
- :file_folder: __File System__: componentes criados para auxiliar o manuseio de arquivos em sistemas operacionais, como a validação da presença de um arquivo em um diretório, validação de atualização de um arquivo, cópia de um arquivo de uma origem para um destino, entre outros. Entre os componentes já implementados, é possível listar:
  - `valida_arquivo_origem()*`
  - `valida_dt_mod_arquivo()*`
  - `copia_arquivo()*`
  - `controle_de_diretorio()*`
  
- :email: __E-mail__: componentes desenvolvidos para facilitar o envio de e-mails customizados (ideia).
  
- :thought_balloon: __Em andamento...__

A fábrica está a todo vapor! Sua capacidade de produção e seu leque de fornecimento pode ser resumido em:

|                            | Funções           | Classes         | Componentes Totais | Homologados  |
| -------------------------- | :---------------: | :-------------: | :-------------: | :-------------: |
| File System                |         4         |        0        |        0        |        0        |
| E-mail                     |         0         |        0        |        0        |        0        |
| Machine Learning           |         0         |        0        |        0        |        0        |


## Utilização

Para consumir os componentes implementados, é necessário baixar o `zip` do repositório ou rodar o comando abaixo em um diretório específico da máquina:
```
git clone https://github.com/ThiagoPanini/python-components.git
```

_Opcional:_ Criação de um ambiente virtual Python para a utilização dos componentes em um projeto específico:
```
mkdir ~/<folder_for_venvs>
cd ~/<folder_for_venvs>
python3 -m venv <venv_name>
source ~/<folder_for_venvs>/<venv_name>/bin/activate
```

Com o ambiente virtual Python do projeto ativo, navegar até o diretório gerado pelo `git clone` e instalando os pacotes necessários para a utilização dos componentes:
```
cd python-components
pip install -r requirements.txt
```
