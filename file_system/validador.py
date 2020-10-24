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


"""
---------------------------------------------------
------------ 1. VALIDAÇÃO DE ARQUIVOS -------------
---------------------------------------------------
"""

# Parâmetros de validação
origem = '/home/paninit/workspaces/python-components/file-system/'
nome_arquivo = 'functions.py'

# 1. valida_arquivos_origem
print('\nTestando função: valida_arquivo_origem()')
if valida_arquivo_origem(origem, nome_arquivo=nome_arquivo):
    print(f'Arquivo {nome_arquivo} presente na origem {origem}')
else:
    print(f'Arquivo não presente na origem')

# 2. valida_dt_mod_arquivo
print('\nTestando função: valida_dt_mod_arquivo()')
dt_valida = {
    'janela': 'anomes',
    'valor': 202010
}
valida_dt_mod_arquivo(origem=origem, nome_arquivo=nome_arquivo, dt_valida=dt_valida)