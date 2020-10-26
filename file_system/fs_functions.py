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
from pandas import DataFrame


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
        os.makedirs(log_path)

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
    validador = {'anomes': 202009}
    if valida_dt_mod_arquivo(origem=origem, nome_arquivo=nome_arquivo, validador=validador):
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
        except FileNotFoundError as e:
            # Erro ao copiar arquivo pro destino
            logger.warning(f'Falha ao copiar arquivo para o destino por inexistência do diretório. Exception lançada: {e}')
            logger.debug(f'Criando novo diretório')
            pasta_destino = '/'.join(destino.split('/')[:-1])
            os.makedirs(pasta_destino)

            # Tentando nova cópia
            try:
                shutil.copyfile(src=origem, dst=destino)
                logger.info(f'Cópia realizada com sucesso. Origem: {origem} - Destino: {destino}')
            except Exception as e:
                logger.error(f'Falha ao copiar arquivo mesmo após criação do diretório destino. Exception lançada: {e}')
    else:
        logger.warning('Cópia não realizada')

def controle_de_diretorio(root, output_filepath=os.path.join(os.getcwd(), 'controle_root.csv')):
    """
    Função responsável por retornar parâmetros de controle de um determinado diretório

    Parâmetros
    ----------
    :param root: caminho do diretório a ser analisado [type: string]
    :param output_file: caminho do output em .csv do arquivo gerado [type: string, default: controle_root.csv]

    Retorno
    -------
    :returns root_manager: arquivo salvo na rede com informações do diretório [type: pd.DataFrame]

    Aplicação
    ---------
    root = '/home/user/folder/'
    controle_root = controle_de_diretorio(root=root)
    """

    # Criando DataFrame e listas para armazenar informações
    root_manager = DataFrame()
    all_files = []
    all_sizes = []
    all_cdt = []
    all_mdt = []
    all_adt = []

    # Iterando sobre todos os arquivos do diretório e subdiretórios
    logger.debug('Iterando sobre os arquivos do diretório root')
    for path, _, files in os.walk(root):
        for name in files:
            # Caminho completo do arquivo
            caminho = os.path.join(path, name)

            # Retornando variáveis
            all_files.append(caminho)
            all_sizes.append(os.path.getsize(caminho))
            all_cdt.append(os.path.getctime(caminho))
            all_mdt.append(os.path.getmtime(caminho))
            all_adt.append(os.path.getatime(caminho))

    # Preenchendo DataFrame
    path_splitter = '\\' if caminho.count('\\') >= caminho.count('/') else '/'
    logger.debug('Preenchendo variáveis de controle')
    root_manager['caminho'] = all_files
    root_manager['arquivo'] = [file.split(path_splitter)[-1] for file in all_files]
    root_manager['tamanho_kb'] = [size / 1024 for size in all_sizes]
    root_manager['dt_criacao'] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cdt)) for cdt in all_cdt] 
    root_manager['dt_ult_modif'] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mdt)) for mdt in all_mdt]
    root_manager['dt_ult_acesso'] = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(adt)) for adt in all_adt]

    # Salvando arquivo gerado
    logger.debug('Salvando arquivo de controle gerado')
    try:
        path_splitter = '\\' if output_filepath.count('\\') >= output_filepath.count('/') else '/'
        output_dir = path_splitter.join(output_filepath.split(path_splitter)[:-1])
        if not isdir(output_dir):
            os.makedirs(output_dir)
        root_manager.to_csv(output_filepath)
        logger.info(f'Arquivo de controle para o diretório {root} salvo com sucesso')
    except Exception as e:
        logger.error(f'Erro ao salvar arquivo de controle. Exception lançada: {e}')

    return root_manager