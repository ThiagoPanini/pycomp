"""
---------------------------------------------------
------------------- TÓPICO: Log -------------------
---------------------------------------------------
Script python responsável por alocar funções úteis
para auxiliar a geração e o armazenamento de logs
em módulos pares

Sumário
-------
1. Configuração do log
"""

# Importando bibliotecas
import logging
import os
from os.path import isdir


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO DE LOGS --------------
---------------------------------------------------
"""

def log_config(logger, level=logging.DEBUG, 
               log_format='%(levelname)s;%(asctime)s;%(filename)s;%(module)s;%(lineno)d;%(message)s',
               log_filepath='exec_log/execution_log.log', filemode='a'):
    """
    Função que recebe um objeto logging e aplica configurações básicas ao mesmo

    Parâmetros
    ----------
    :param logger: objeto logger criado no escopo do módulo [type: logging.getLogger()]
    :param level: level do objeto logger criado [type: level, default: logging.DEBUG]
    :param log_format: formato do log a ser armazenado [type: string]
    :param log_filepath: caminho onde o arquivo .log será armazenado [type: string, default: 'log/application_log.log']
    :param filemode: tipo de escrita no arquivo de log [type: string, default: 'a' (append)]

    Retorno
    -------
    :return logger: objeto logger pré-configurado
    """

    # Setting level for the logger object
    logger.setLevel(level)

    # Creating a formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Creating handlers
    log_path = ''.join(log_filepath.split('/')[:-1])
    if not isdir(log_path):
        os.mkdir(log_path)

    file_handler = logging.FileHandler(log_filepath, mode=filemode, encoding='utf-8')
    stream_handler = logging.StreamHandler()

    # Setting up formatter on handlers
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Adding handlers on logger object
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger