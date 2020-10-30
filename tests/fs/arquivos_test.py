"""
---------------------------------------------------
------ MÓDULO: File System (fs) - arquivos.py -----
---------------------------------------------------
Script de execução de testes relacionados ao móulo
arquivos.py do pacote pycomp.fs (file system).

Sumário
-----------------------------------

-----------------------------------
"""

# Importando módulos
from pycomp.fs.arquivos import *
import os

# Definindo variáveis de teste
PATH = '/home/paninit/workspaces/pycomp/'
FILENAME = 'requirements.txt'


"""
---------------------------------------------------
----------- TESTANDO FUNÇÕES DO MÓDULO ------------
---------------------------------------------------
"""

# 1. Validando arquivos na origem
if valida_arquivo_origem(origem=PATH, nome_arquivo=FILENAME):
    pass
else:
    pass

# 2.1 Validando data de modificação de um arquivo: ano
validador = {
    'janela': 'ano',
    'valor': 2020
}
if valida_dt_mod_arquivo(origem=PATH, nome_arquivo=FILENAME, validador=validador):
    pass
else:
    pass

# 2.2 Validando data de modificação de um arquivo: anomes
validador = {
    'janela': 'anomes',
    'valor': 202010
}
if valida_dt_mod_arquivo(origem=PATH, nome_arquivo=FILENAME, validador=validador):
    pass
else:
    pass

# 2.3 Validando data de modificação de um arquivo: anomesdia
validador = {
    'janela': 'anomesdia',
    'valor': 20201020
}
if valida_dt_mod_arquivo(origem=PATH, nome_arquivo=FILENAME, validador=validador):
    pass
else:
    pass

# 3. Copiando arquivos
DESTINO = '/home/paninit/workspaces/pycomp/tmp/copia_requirements.txt'
copia_arquivo(origem=os.path.join(PATH, FILENAME), destino=DESTINO)

# 4. Gerenciando arquivos em um diretório
ROOT = '/home/paninit/workspaces/pycomp/tmp'
controle_de_diretorio(root=ROOT, output_filepath=os.path.join(ROOT, 'controle_root.csv'))
