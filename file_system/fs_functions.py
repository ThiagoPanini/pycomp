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


"""
---------------------------------------------------
------------ 1. VALIDAÇÃO DE ARQUIVOS -------------
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
            return True
        else:
            return False
    except NotADirectoryError as e:
        print(f'Parâmetro origem {origem} não é um diretório de rede. Exception lançada: \n{e}')
        return False
    except FileNotFoundError as e:
        print(f'Arquivo {nome_arquivo} não encontrado na origem. Exception lançada: \n{e}')
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
    for k, v in kwargs.items():
        validador = v
    janela = validador['janela']
    dt_valida = validador['valor']

    # Definindo mensagens
    msg_ok = f'A última modificação do arquivo {nome_arquivo} bate com o validador ({janela}: {dt_valida})'
    msg_nok = f'A última modificação do arquivo {nome_arquivo} não bate com o validador ({dt_valida} contra placeholder)'

    # Validando presença do arquivo na origem e coletando última data de modificação
    if valida_arquivo_origem(origem=origem, nome_arquivo=nome_arquivo):
        file_mod_date = os.path.getmtime(os.path.join(origem, nome_arquivo))

        # Janela selecionada: ano
        if janela == 'ano':
            ano_mod = int(time.strftime('%Y', time.localtime(file_mod_date)))
            if dt_valida == ano_mod:
                print(msg_ok)
                return True
            else:
                print(msg_nok.replace('placeholder', ano_mod))
                return False

        # Janela selecionada: anomes
        elif janela == 'anomes':
            anomes_mod = int(time.strftime('%Y%m', time.localtime(file_mod_date)))
            if dt_valida == anomes_mod:
                print(msg_ok)
                return True
            else:
                print(msg_nok.replace('placeholder', anomes_mod))
                return False

        # Janela selecionada: anomesdida
        elif janela == 'anomesdia':
            anomesdia_mod = int(time.strftime('%Y%m%d', time.localtime(file_mod_date)))
            if dt_valida == anomesdia_mod:
                print(msg_ok)
                return True
            else:
                print(msg_nok.replace('placeholder', anomesdia_mod))
                return False     
    else:
        # Arquivo não existente na origem
        return False