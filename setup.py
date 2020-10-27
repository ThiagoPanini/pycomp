"""
---------------------------------------------------------
                    Resumo do Módulo
---------------------------------------------------------
    Arquivo de setup com as principais informações da 
aplicação consolidadas a partir da biblioteca setuptools

---------------------------------------------------------
                          FAQ
---------------------------------------------------------

1. Qual o objetivo do script setup.py?
    R: O arquivo setup.py serve para consolidar algumas 
informações úteis da aplicação e fornecer um informativo 
básico para novos desenvolvedores

---------------------------------------------------------
2. Qual sua usabilidade em aplicações criadas?
    R: Na prática, o arquivo setup.py pode ser utilizado 
para instalação dos pacotes no virtual env de trabalho 

Ref [2.1]: https://stackoverflow.com/questions/1471994/what-is-setup-py
"""

# Bibliotecas
from setuptools import setup, find_packages

# Definindo variáveis de setup
__version__ = '0.0.1'
__description__ = 'Python Factory'
__long_description__ = 'Pacote de funções e classes para auxiliar no desenvolvimento de aplicações e automatização de tarefas'

__author__ = 'Thiago Panini'
__author_email__ = 'thipanini94@gmail.com'

# Criando setup
setup(
    name='bebop',
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    packages=find_packages(),
    description=__description__,
    long_description=__long_description__,
    url='https://github.com/ThiagoPanini/python-components',
    keywords='Python, Factory, Packages, Modules',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrator',
        'Operation System :: OS Independent',
        'Topic :: Software Development',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.0.0"
)

# Hint: publicando Source Archive (tar.gz) e Built Distribution (.whl)
# python3 setup.py sdist bdist_wheel