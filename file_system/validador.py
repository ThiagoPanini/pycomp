"""
---------------------------------------------------
--------------- TÓPICO: File System ---------------
---------------------------------------------------
Arquivo validador das funções implementadas no script
functions.py

Sumário
-------
1. Validação de Arquivos
2. Cópia de Arquivos
"""


# Importando bibliotecas
from fs_functions import *
import os


"""
---------------------------------------------------
------------ 1. VALIDAÇÃO DE ARQUIVOS -------------
---------------------------------------------------
"""

# Parâmetros de validação
#origem = '/home/paninit/workspaces/python-components/file-system/'
origem = 'C:/Users/thipa/Desktop/workspaces/python-components/file_system'
nome_arquivo = 'fs_functions.py'

# 1. valida_arquivos_origem
print('\n1. Testando função: valida_arquivo_origem()')
if valida_arquivo_origem(origem, nome_arquivo=nome_arquivo):
    dum = 0
else:
    dum = 1

# 2. valida_dt_mod_arquivo
print('\n2. Testando função: valida_dt_mod_arquivo()')
dt_valida = {
    'janela': 'anomes',
    'valor': 202009
}
valida_dt_mod_arquivo(origem=origem, nome_arquivo=nome_arquivo, dt_valida=dt_valida)

# 3. copia_arquivo
print('\n3. Testando função: copia_arquivo()')
source = origem + '/' + nome_arquivo
destino = 'C:/Users/thipa/Desktop/workspaces/python-components/testes/copia_fs_functions.py'
copia_arquivo(origem=source, destino=destino)

# 4. controle_de_diretorio
print('\n4. Testando função: controle_de_diretorio()')
root = 'C:/Users/thipa/Desktop/workspaces/python-components'
controle_root = controle_de_diretorio(root=root)

print()