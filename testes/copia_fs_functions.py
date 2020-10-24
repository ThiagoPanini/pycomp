"""
---------------------------------------------------
--------------- TÓPICO: File System ---------------
---------------------------------------------------
Script python responsável por alocar funções úteis
para auxiliar o tratamento e o manuseio de operações
realizadas no sistema operacional, como validação e
cópia de arquivos.

Sumário
-------
1. Validação de Arquivos
2. Cópia de Arquivos
"""

# Importando bibliotecas
import os
import time
import logging
from os.path import isdir
import shutil


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO INICIAL --------------
        1.1 Configuração padrão de logs
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
    log_path = '/'.join(log_filepath.split('/')[:-1])
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


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO INICIAL --------------
        1.2 Instânciando objetos de logs
---------------------------------------------------
"""

# Definindo objeto de log
logger = logging.getLogger(__file__)
logger = log_config(logger)


"""
---------------------------------------------------
------------ 2. VALIDAÇÃO DE ARQUIVOS -------------
---------------------------------------------------
"""

def valida_arquivo_origem(origem, nome_arquivo):
    """
    Função responsável por validar a presença de umnarquivo em determinado diretório origem

    Parâmetros
    ----------
    :param origem: caminho do diretório origem alvo da validação [type: string]
    :param nome_arquivo: nome do arquivo (com extensão) a ser validado [type: string]

    Retorno
    -------
    :return flag: flag indicativo da presença do arquivo no diretório origem [type: bool]

    Aplicação
    ---------
    # Verificando arquivo em diretório
    nome_arquivo = 'arquivo.txt'
    origem = 'C://Users/user/Desktop'
    if valida_arquivo_origem(origem=origem, nome_arquivo=nome_arquivo):
        doSomething()
    else:
        doNothing()
    """
    
    # Validando presença do arquivo na origem
    try:
        arquivos_na_origem = os.listdir(path=origem)
        if nome_arquivo in arquivos_na_origem:
            logger.info(f'Arquivo {nome_arquivo} presente na origem {origem}')
            return True
        else:
            logger.warning(f'Arquivo {nome_arquivo} não presente na origem {origem}')
            return False
    except NotADirectoryError as e:
        logger.error(f'Parâmetro origem {origem} não é um diretório de rede. Exception lançada: {e}')
        return False
    except FileNotFoundError as e:
        logger.error(f'Arquivo {nome_arquivo} não encontrado na origem. Exception lançada: {e}')
        return False

def valida_dt_mod_arquivo(origem, nome_arquivo, **kwargs):
    """
    Função responsável por validar a presença e a última data de execução
    de um arquivo em determinado diretório origem e em uma determinada janela
    temporal de modificação

    Parâmetros
    ----------
    :param origem: caminho do diretório origem alvo da validação [type: string]
    :param nome_arquivo: nome do arquivo (com extensão) a ser validado [type: string]
    :param **kwargs: dicionário considerando chaves e valores temporais de verificação
        argumentos: {
            'janela': 'ano' ou 'anomes' ou 'anomesdia',
            'valor': 'yyyy' ou 'yyyyMM' ou 'yyyyMMdd'
        }

    Retorno
    -------
    :return flag: flag indicativo da presença e a atualização do arquivo na origem [type: bool]

    Aplicação
    ---------
    # Verificando arquivo em diretório
    nome_arquivo = 'arquivo.txt'
    origem = 'C://Users/user/Desktop'
    janela = {'anomes': 202009}
    if valida_dt_mod_arquivo(origem=origem, nome_arquivo=nome_arquivo, janela=):
        doSomething()
    else:
        doNothing()
    """

    # Extraindo informações dos argumentos dinâmicos
    for _, v in kwargs.items():
        validador = v
    janela = validador['janela']
    dt_valida = validador['valor']

    # Definindo mensagens
    msg_ok = f'A última modificação do arquivo {nome_arquivo} bate com o validador ({janela}: {dt_valida})'
    msg_nok = f'A última modificação do arquivo {nome_arquivo} (placeholder) não bate com o validador ({dt_valida})'

    # Validando presença do arquivo na origem e coletando última data de modificação
    if valida_arquivo_origem(origem=origem, nome_arquivo=nome_arquivo):
        file_mod_date = os.path.getmtime(os.path.join(origem, nome_arquivo))

        # Janela selecionada: ano
        if janela == 'ano':
            ano_mod = int(time.strftime('%Y', time.localtime(file_mod_date)))
            if dt_valida == ano_mod:
                logger.info(msg_ok)
                return True
            else:
                logger.warning(msg_nok.replace('placeholder', str(ano_mod)))
                return False

        # Janela selecionada: anomes
        elif janela == 'anomes':
            anomes_mod = int(time.strftime('%Y%m', time.localtime(file_mod_date)))
            if dt_valida == anomes_mod:
                logger.info(msg_ok)
                return True
            else:
                logger.warning(msg_nok.replace('placeholder', str(anomes_mod)))
                return False

        # Janela selecionada: anomesdida
        elif janela == 'anomesdia':
            anomesdia_mod = int(time.strftime('%Y%m%d', time.localtime(file_mod_date)))
            if dt_valida == anomesdia_mod:
                logger.info(msg_ok)
                return True
            else:
                logger.warning(msg_nok.replace('placeholder', str(anomesdia_mod)))
                return False     
    else:
        return False

def copia_arquivo(origem, destino):
    """
    Função responsável por copiar um arquivo definido em uma origem para um destino

    Parâmetros
    ----------
    :param origem: definição do arquivo origem (caminho + nome do arquivo) [type: string]
    :param destino: definição do destino da cópia (caminho + nome do arquivo) [type: string]

    Retorno
    -------
    None

    Aplicação
    ---------
    # Copiando arquivo
    origem = '/home/user/folder/file.txt'
    destino = '/home/user/new_folder/file.txt'
    copia_arquivo(origem=origem, destino=destino)
    """

    # Verificando se o arquivo está presente na origem
    diretorio = '/'.join(origem.split('/')[:-1])
    nome_arquivo = origem.split('/')[-1]
    if valida_arquivo_origem(origem=diretorio, nome_arquivo=nome_arquivo):
        try:
            shutil.copyfile(src=origem, dst=destino)
            logger.info(f'Cópia realizada com sucesso. Origem: {origem} - Destino: {destino}')
        except Exception as e:
            logger.error(f'Falha durante a cópia. Exception lançada: {e}')
    else:
        logger.warning('Cópia não realizada')
