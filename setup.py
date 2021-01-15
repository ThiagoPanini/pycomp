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
#from pip.req import parse_requirements

# Lendo README.md
with open("README.md", "r", encoding='utf-8') as f:
    __long_description__ = f.read()

# Lendo dependências do pacote
"""install_reqs = parse_requirements('requirements_pkg.txt', session='hack')
reqs = [str(ir.req) for ir in install_reqs]"""

# Criando setup
setup(
    name='pycomp',
    version='0.2.10',
    author='Thiago Panini',
    author_email='thipanini94@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy==1.18.5',
        'pandas==1.1.5',
        'joblib==0.14.1',
        'scikit-learn==0.23.2',
        'matplotlib==3.2.1',
        'seaborn==0.10.0',
        'shap==0.37.0'
    ],
    license='MIT',
    description='Fábrica de componentes Python',
    long_description=__long_description__,
    long_description_content_type="text/markdown",
    url='https://github.com/ThiagoPanini/pycomp',
    keywords='Packages, Components, Machine Learning, AutoML, Viz',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Natural Language :: Portuguese (Brazilian)",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires=">=3.0.0"
)

# Hint: publicando Source Archive (tar.gz) e Built Distribution (.whl)
# python3 setup.py sdist bdist_wheel
# twine check dist/*
# python3 -m twine upload --skip-existing dist/*