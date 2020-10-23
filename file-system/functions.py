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
    janela = kwargs['janela']
    #valor = kwargs['valor']

    # Validando presença do arquivo na origem
    print(janela)
    print(kwargs['janela'])