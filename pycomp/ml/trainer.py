"""
---------------------------------------------------
------- TÓPICO: Machine Learning - Trainer --------
---------------------------------------------------
Módulo responsável por proporcionar componentes 
inteligentes para o treinamento e avaliação de 
modelos básicos de Machine Learning

Sumário
-----------------------------------

-----------------------------------
"""

# Importando bibliotecas
import logging
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import joblib
from pycomp.log.log_config import log_config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, roc_curve, mean_absolute_error, \
                            mean_squared_error, r2_score, classification_report
from sklearn.exceptions import NotFittedError
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import shap
from pycomp.viz.formatador import format_spines, AnnotateBars


"""
---------------------------------------------------
------------ 1. CONFIGURAÇÃO INICIAL --------------
        1.2 Instanciando Objetos de Log
---------------------------------------------------
"""

# Definindo objeto de log
logger = logging.getLogger(__file__)
logger = log_config(logger)


"""
---------------------------------------------------
--------------- 2. CLASSIFICAÇÃO ------------------
           2.1 Classificador Binário
---------------------------------------------------
"""

class ClassificadorBinario:
    """
    Classe responsável por consolidar métodos úteis para o treinamento
    e avaliação de modelos de classificação binária em um contexto de
    aprendizado supervisionado
    """

    def __init__(self):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.classifiers_info = {}

    def save_data(self, data, output_path, filename):
        """
        Método responsável por salvar objetos DataFrame em formato csv.

        Parâmetros
        ----------
        :param data: arquivo/objeto a ser salvo [type: pd.DataFrame]
        :param output_path: referência de diretório destino [type: string]
        :param filename: referência do nome do arquivo a ser salvo [type: string]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do arquivo devidamente salvo no diretório

        Aplicação
        ---------
        df = file_generator_method()
        self.save_result(df, output_path=OUTPUT_PATH, filename='arquivo.csv')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
                return

        logger.debug(f'Salvando arquivo no diretório especificado')
        try:
            output_file = os.path.join(output_path, filename)
            data.to_csv(output_file, index=False)
        except Exception as e:
            logger.error(f'Erro ao salvar arquivo {filename}. Exception lançada: {e}')

    def save_model(self, model, output_path, filename):
        """
        Método responsável por salvar modelos treinado em fomato pkl.

        Parâmetros
        ----------
        :param model: objeto a ser salvo [type: model]
        :param output_path: referência de diretório destino [type: string]
        :param filename: referência do nome do modelo a ser salvo [type: string]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do objeto devidamente salvo no diretório

        Aplicação
        ---------
        model = classifiers['estimator']
        self.save_model(model, output_path=OUTPUT_PATH, filename='model.pkl')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
                return

        logger.debug(f'Salvando modelo pkl no diretório especificado')
        try:
            output_file = os.path.join(output_path, filename)
            joblib.dump(model, output_file)
        except Exception as e:
            logger.error(f'Erro ao salvar modelo {filename}. Exception lançada: {e}')

    def save_fig(self, fig, output_path, img_name, tight_layout=True, dpi=300):
        """
        Método responsável por salvar imagens geradas pelo matplotlib/seaborn

        Parâmetros
        ----------
        :param fig: figura criada pelo matplotlib para a plotagem gráfica [type: plt.figure]
        :param output_file: caminho final a ser salvo (+ nome do arquivo em formato png) [type: string]
        :param tight_layout: flag que define o acerto da imagem [type: bool, default=True]
        :param dpi: resolução da imagem a ser salva [type: int, default=300]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento da imagem em diretório especificado

        Aplicação
        ---------
        fig, ax = plt.subplots()
        save_fig(fig, output_file='imagem.png')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
        
        # Acertando layout da imagem
        if tight_layout:
            fig.tight_layout()
        
        logger.debug('Salvando imagem no diretório especificado')
        try:
            output_file = os.path.join(output_path, img_name)
            fig.savefig(output_file, dpi=300)
            logger.info(f'Imagem salva com sucesso em {output_file}')
        except Exception as e:
            logger.error(f'Erro ao salvar imagem. Exception lançada: {e}')

    def fit(self, set_classifiers, X_train, y_train, **kwargs):
        """
        Método responsável por treinar cada um dos classificadores contidos no dicionário
        set_classifiers através da aplicação das regras estabelecidas pelos argumentos do método

        Parâmetros
        ----------
        :param set_classifiers: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features do modelo a ser treinado [type: np.array]
        :param y_train: array contendo variável target do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: -1]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento de objetos do modelo [type: string, default=cwd() + 'output/models']
            :arg model_ext: extensão do objeto gerado (pkl ou joblib) - sem o ponto [type: string, default='pkl']

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train_prep, y_train)
        """

        # Referenciando argumentos adicionais
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_classifiers.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                model_key = model_name + approach
                logger.debug(f'Treinando modelo {model_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.classifiers_info[model_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Retornando parâmetros em kwargs
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparando e aplicando busca
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.classifiers_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.classifiers_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Erro ao aplicar RandomizedSearch. Exception lançada: {te}')
                    return

                # Validando salvamento de objetos pkl dos modelos
                if 'save' in kwargs and bool(kwargs['save']):
                    logger.debug(f'Salvando arquivo pkl do modelo {model_name} treinado')
                    model = self.classifiers_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'

                    self.save_model(model, output_path=output_path, filename=model_name.lower() + '.' + model_ext)

        except AttributeError as ae:
            logger.error('Erro ao treinar modelos. Exception lançada: {ae}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')

    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Método responsável por aplicar validação cruzada para retornar a amédia das principais métricas de avaliação
        de um modelo de classificação. Na prática, esse método é chamado por um outro método em uma camada
        superior da classe para medição de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]

        Retorno
        -------
        :return train_performance: DataFrame contendo as métricas calculadas usando validação cruzada [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computando métricas utilizando validação cruzada
        logger.debug(f'Computando métricas do modelo {model_name} utilizando validação cruzada com {cv} K-folds')
        try:
            t0 = time.time()
            accuracy = cross_val_score(estimator, X, y, cv=cv, scoring='accuracy').mean()
            precision = cross_val_score(estimator, X, y, cv=cv, scoring='precision').mean()
            recall = cross_val_score(estimator, X, y, cv=cv, scoring='recall').mean()
            f1 = cross_val_score(estimator, X, y, cv=cv, scoring='f1').mean()

            # Probabilidades para o cálculo da AUC
            try:
                y_scores = cross_val_predict(estimator, X, y, cv=cv, method='decision_function')
            except:
                # Modelos baseados em árvore não possuem o método decision_function() mas sim o predict_proba()
                y_probas = cross_val_predict(estimator, X, y, cv=cv, method='predict_proba')
                y_scores = y_probas[:, 1]
            auc = roc_auc_score(y, y_scores)

            # Salvando métricas no atributo self.classifiers_info
            self.classifiers_info[model_name]['train_scores'] = y_scores

            # Criando DataFrame com o resultado obtido
            t1 = time.time()
            delta_time = t1 - t0
            train_performance = {}
            train_performance['model'] = model_name
            train_performance['approach'] = f'Treino {cv} K-folds'
            train_performance['acc'] = round(accuracy, 4)
            train_performance['precision'] = round(precision, 4)
            train_performance['recall'] = round(recall, 4)
            train_performance['f1'] = round(f1, 4)
            train_performance['auc'] = round(auc, 4)
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de treino em {round(delta_time, 3)} segundos')

            return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def compute_test_performance(self, model_name, estimator, X, y):
        """
        Método responsável por aplicar retornar as principais métricas do model utilizando dados de teste.
        Na prática, esse método é chamado por um outro método em uma camada superior da classe para medição 
        de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y: array contendo a variável resposta dos dados de teste do modelo [type: np.array]

        Retorno
        -------
        :return test_performance: DataFrame contendo as métricas calculadas nos dados de teste [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        test_performance = trainer.compute_test_performance(model_name, estimator, X_test, y_test)
        """

        # Predicting data using the trained model and computing probabilities
        logger.debug(f'Computando métricas do modelo {model_name} utilizando dados de teste')
        try:
            t0 = time.time()
            y_pred = estimator.predict(X)
            y_proba = estimator.predict_proba(X)
            y_scores = y_proba[:, 1]

            # Retrieving metrics using test data
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_scores)

            # Saving probabilities on treined classifiers dictionary
            self.classifiers_info[model_name]['test_scores'] = y_scores

            # Creating a DataFrame with metrics
            t1 = time.time()
            delta_time = t1 - t0
            test_performance = {}
            test_performance['model'] = model_name
            test_performance['approach'] = f'Teste'
            test_performance['acc'] = round(accuracy, 4)
            test_performance['precision'] = round(precision, 4)
            test_performance['recall'] = round(recall, 4)
            test_performance['f1'] = round(f1, 4)
            test_performance['auc'] = round(auc, 4)
            test_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de teste em {round(delta_time, 3)} segundos')

            return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5, **kwargs):
        """
        Método responsável por executar e retornar métricas dos classificadores em treino (média do resultado
        da validação cruzada com cv K-fols) e teste
        
        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
        
        Retorno
        -------
        :return df_performances: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

        Aplicação
        -----------
        # Treinando modelo e avaliando performance em treino e teste
        trainer = ClassificadorBinario()
        trainer.fit(estimator, X_train, X_test)

        # Definindo dicionário de controle do resultado
        df_performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, save=True, output_path=caminho)
        """

        # DataFrame vazio para armazenamento das métrics
        df_performances = pd.DataFrame({})

        # Iterando sobre todos os classificadores da classe
        for model_name, model_info in self.classifiers_info.items():

            # Validando se o modelo já foi treinado (dicionário model_info já terá a chave 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            # Retornando modelo a ser avaliado
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Erro ao retornar a chave "estimator" do dicionário model_info. Modelo {model_name} não treinado')
                continue

            # Computando performance em treino e em teste
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, cv=cv)
            test_performance = self.compute_test_performance(model_name, estimator, X_test, y_test)

            # Adicionando os resultados ao atributo classifiers_info
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['test_performance'] = test_performance

            # Construindo DataFrame com as métricas retornadas
            model_performance = train_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Salvando alguns atributos no dicionário classifiers_info para acessos futuros
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            model_info['model_data'] = model_data

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            self.save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances

    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Método responsável por retornar a importância das features de um modelo treinado
        
        Parâmetros
        ----------
        :param features: lista contendo as features de um modelo [type: list]
        :param top_n: parâmetro para filtragem das top n features [type: int, default=-1]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']

        Retorno
        -------
        :return: all_feat_imp: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]
        """

        # Inicializando DataFrame vazio para armazenamento das feature importance
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})

        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')

            try:
                importances = model_info['estimator'].feature_importances_
            except KeyError as ke:
                logger.error(f'Modelo {model_name} não treinado, sendo impossível extrair o método feature_importances_')
                continue
            except AttributeError as ae:
                logger.error(f'Modelo {model_name} não possui o método feature_importances_')
                continue

            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['model'] = model_name
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp = feat_imp.loc[:, ['model', 'feature', 'importance', 'anomesdia_datetime']]

            # Salvando essa informação no dicionário classifiers_info
            self.classifiers_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            logger.info(f'Extração da importância das features concluída com sucesso para o modelo {model_name}')

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'top_features.csv'
            self.save_data(all_feat_imp, output_path=output_path, filename=output_filename)
        
        return all_feat_imp

    def training_flow(self, set_classifiers, X_train, y_train, X_test, y_test, features, **kwargs):
        """
        Método responsável por consolidar um fluxo completo de treinamento dos classificadores, bem como
        o levantamento de métricas e execução de métodos adicionais para escolha do melhor modelo

        Parâmetros
        ----------
        :param set_classifiers: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param features: lista contendo as features de um modelo [type: list]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'output/')]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: 5]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg models_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/models']
            :arg metrics_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg metrics_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
            :arg featimp_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']
            :arg top_n_featimp: top features a serem analisadas na importância das features [type: int, default=-1]

        Retorno
        -------
        None

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        """

        # Definindo variáveis padrão para retorno dos resultados
        """if not os.path.isdir(output_path):
            os.makedirs(output_path)"""

        # Extraindo parâmetros kwargs
        approach = kwargs['approach'] if 'approach' in kwargs else ''
        random_search = kwargs['random_search'] if 'random_search' in kwargs else False
        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
        cv = kwargs['cv'] if 'cv' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        save = bool(kwargs['save']) if 'save' in kwargs else True
        models_output_path = kwargs['models_output_path'] if 'models_output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
        metrics_output_path = kwargs['metrics_output_path'] if 'metrics_output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
        metrics_output_filename = kwargs['metrics_output_filename'] if 'metrics_output_filename' in kwargs else 'metrics.csv'
        featimp_output_filename = kwargs['featimp_output_filename'] if 'featimp_output_filename' in kwargs else 'top_features.csv'
        top_n_featimp = kwargs['top_n_featimp'] if 'top_n_featimp' in kwargs else -1

        # Treinando classificadores
        self.fit(set_classifiers, X_train, y_train, approach=approach, random_search=random_search, scoring=scoring,
                 cv=cv, verbose=verbose, n_jobs=n_jobs, output_path=models_output_path)

        # Avaliando modelos
        self.evaluate_performance(X_train, y_train, X_test, y_test, save=save, output_path=metrics_output_path, 
                                  output_filename=metrics_output_filename)

        # Analisando Features mais importantes
        self.feature_importance(features, top_n=top_n_featimp, save=save, output_path=metrics_output_path, 
                                output_filename=featimp_output_filename)

    def plot_metrics(self, figsize=(16, 10), palette='rainbow', cv=5, **kwargs):
        """
        Método responsável por plotar os resultados das métricas dos classificadores selecionados

        Parâmetros
        ----------
        :param figsize: dimensões da figura gerada para a plotagem [type: tuple, default=(16, 10)]
        :param palette: paleta de cores do matplotlib [type: string, default='rainbow']
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics_comparison.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da plotagem devidamente salva no diretório destino
        """

        logger.debug(f'Iniciando plotagem gráfica das métricas dos classificadores')
        metrics = pd.DataFrame()
        for model_name, model_info in self.classifiers_info.items():
            
            logger.debug(f'Retornando métricas via validação cruzada para o modelo {model_name}')
            try:
                # Retornando variáveis do classificador
                metrics_tmp = pd.DataFrame()
                estimator = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Métricas não computadas
                accuracy = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='accuracy')
                precision = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='precision')
                recall = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='recall')
                f1 = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='f1')

                # Adicionando ao DataFrame recém criado
                metrics_tmp['accuracy'] = accuracy
                metrics_tmp['precision'] = precision
                metrics_tmp['recall'] = recall
                metrics_tmp['f1'] = f1
                metrics_tmp['model'] = model_name

                # Empilhando métricas
                metrics = metrics.append(metrics_tmp)
            except Exception as e:
                logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')
                continue
        
        logger.debug(f'Modificando DataFrame de métricas para plotagem gráfica')
        try:
            # Pivotando métricas (boxplot)
            index_cols = ['model']
            metrics_cols = ['accuracy', 'precision', 'recall', 'f1']
            df_metrics = pd.melt(metrics, id_vars=index_cols, value_vars=metrics_cols)

            # Agrupando métricas (barras)
            metrics_group = df_metrics.groupby(by=['model', 'variable'], as_index=False).mean()
        except Exception as e:
            logger.error(f'Erro ao pivotar DataFrame. Exception lançada: {e}')
            return

        logger.debug(f'Plotando análise gráfica das métricas para os modelos treinados')
        try:
            # Plotando gráficos
            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=figsize)
            sns.boxplot(x='variable', y='value', data=df_metrics.sort_values(by='model'), hue='model', ax=axs[0], palette=palette)
            sns.barplot(x='variable', y='value', data=metrics_group, hue='model', ax=axs[1], palette=palette, order=metrics_cols)

            # Customizando eixos
            axs[0].set_title(f'Dispersão das métricas utilizando validação cruzada nos dados de treino com {cv} K-folds', size=14, pad=15)
            axs[1].set_title(f'Média para cada métrica obtida na validação cruzada', size=14, pad=15)
            format_spines(axs[0], right_border=False)
            format_spines(axs[1], right_border=False)
            axs[1].get_legend().set_visible(False)
            AnnotateBars(n_dec=3, color='black', font_size=12).vertical(axs[1])
        except Exception as e:
            logger.error(f'Erro ao plotar gráfico das métricas. Exception lançada: {e}')
            return

        # Alinhando figura
        plt.tight_layout()

        # Salvando figura
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics_comparison.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)      

    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Método responsável por realizar uma plotagem gráfica das variáveis mais importantes pro modelo

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param top_n: quantidade de features a ser considerada na plotagem [type: int, default=20]
        :param palette: paleta de cores utilizazda na plotagem [type: string, default='viridis']
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='feature_importances.png']

        Retorno
        -------
        Este método não retorna nada além da imagem devidamente salva no diretório destino
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   

    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        Método utilizada para plotar uma matriz de confusão customizada para um único modelo da classe. Em geral,
        esse método pode ser chamado por um método de camada superior para plotagem de matrizes para todos os
        modelos presentes na classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param y_true: array contendo a variável target do dataset [type: np.array]
        :param y_pred: array com as predições retornadas pelo respectivo modelo [type: np.array]
        :param classes: nomenclatura das classes da matriz [type: list]
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        -----------
        Visualizar o método self.plot_confusion_matrix()
        """

        # Retornando a matriz de confusão usando função do sklearn
        conf_mx = confusion_matrix(y_true, y_pred)

        # Plotando matriz
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizando eixos
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)
    
    def plot_confusion_matrix(self, cmap=plt.cm.Blues, normalize=False, **kwargs):
        """
        Método responsável por plotar gráficos de matriz de confusão usando dados de treino e teste
        para todos os modelos presentes no dicionárion de classificadores self.classifiers_info

        Parâmetros
        ----------
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']
        
        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_confusion_matrix(output_path=OUTPUT_PATH)
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem da matriz de confusão para os modelos')
        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(10, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterando sobre cada classificador da classe
        for model_name, model_info in self.classifiers_info.items():
            logger.debug(f'Retornando dados de treino e teste para o modelo {model_name}')
            try:
                # Retornando dados para cada modelo
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']
                X_test = model_info['model_data']['X_test']
                y_test = model_info['model_data']['y_test']
                classes = np.unique(y_train)
            except Exception as e:
                logger.error(f'Erro ao retornar dados para o modelo {model_name}. Exception lançada: {e}')
                continue

            # Realizando predições em treino (cross validation) e teste
            logger.debug(f'Realizando predições para os dados de treino e teste ({model_name})')
            try:
                train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
                test_pred = model_info['estimator'].predict(X_test)
            except Exception as e:
                logger.error(f'Erro ao realizar predições para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Gerando matriz de confusão para o modelo {model_name}')
            try:
                # Plotando matriz utilizando dados de treino
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, cmap=cmap,
                                            normalize=normalize)
                k += 1

                # Plotando matriz utilizando dados de teste
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Test', y_test, test_pred, classes=classes, cmap=plt.cm.Greens,
                                            normalize=normalize)
                k += 1
                logger.info(f'Matriz de confusão gerada para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar a matriz para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'confusion_matrix.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   

    def plot_roc_curve(self, figsize=(16, 6), **kwargs):
        """
        Método responsável por iterar sobre os classificadores presentes na classe e plotar a curva ROC
        para treino (primeiro eixo) e teste (segundo eixo)

        Parâmetros
        ----------
        :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 6)]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='roc_curve.png']

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da curva ROC especificada

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_roc_curve(output_path=OUTPUT_PATH)
        """

        # Criando figura de plotagem
        logger.debug('Inicializando plotagem da curva ROC para os modelos')
        fig, axs = plt.subplots(ncols=2, figsize=figsize)

        # Iterando sobre os classificadores presentes na classe
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Retornando labels e scores de treino e de teste para o modelo {model_name}')
            try:
                # Retornando label de treino e de teste
                y_train = model_info['model_data']['y_train']
                y_test = model_info['model_data']['y_test']

                # Retornando scores já calculados no método de avaliação de performance
                train_scores = model_info['train_scores']
                test_scores = model_info['test_scores']
            except Exception as e:
                logger.error(f'Erro ao retornar os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Calculando FPR, TPR e AUC de treino e teste para o modelo {model_name}')
            try:
                # Calculando taxas de falsos positivos e verdadeiros positivos
                train_fpr, train_tpr, train_thresholds = roc_curve(y_train, train_scores)
                test_fpr, test_tpr, test_thresholds = roc_curve(y_test, test_scores)

                # Retornando AUC de treino e teste já calculada no método de avaliação de performance
                train_auc = model_info['train_performance']['auc'].values[0]
                test_auc = model_info['test_performance']['auc'].values[0]
            except Exception as e:
                logger.error(f'Erro ao calcular os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando curva ROC de treino e teste para o modelo {model_name}')
            try:
                # Plotando curva ROC (treino)
                plt.subplot(1, 2, 1)
                plt.plot(train_fpr, train_tpr, linewidth=2, label=f'{model_name} auc={train_auc}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([-0.02, 1.02, -0.02, 1.02])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Train Data')
                plt.legend()

                # Plotando curva ROC (teste)
                plt.subplot(1, 2, 2)
                plt.plot(test_fpr, test_tpr, linewidth=2, label=f'{model_name} auc={test_auc}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([-0.02, 1.02, -0.02, 1.02])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Test Data', size=12)
                plt.legend()
            except Exception as e:
                logger.error(f'Erro ao plotar curva ROC para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'roc_curve.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   
    
    def plot_score_distribution(self, shade=True, **kwargs):
        """
        Método responsável por plotar gráficos de distribuição de score (kdeplot) para os
        dados de treino e teste separados pela classe target
        
        Parâmetros
        ----------
        :param shade: flag indicativo de preenchimento da área sob a curva [type: bool, default=True]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='score_distribution.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento do gráfico de distribuição especificado
        
        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_score_distribution(output_path=OUTPUT_PATH)
        """

        # Criando figura de plotagem
        logger.debug('Inicializando plotagem da distribuição de score para os modelos')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterando sobre os classificadores presentes na classe
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Retornando labels e scores de treino e de teste para o modelo {model_name}')
            try:
                # Retornando label de treino e de teste
                y_train = model_info['model_data']['y_train']
                y_test = model_info['model_data']['y_test']

                # Retornando scores já calculados no método de avaliação de performance
                train_scores = model_info['train_scores']
                test_scores = model_info['test_scores']
            except Exception as e:
                logger.error(f'Erro ao retornar os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando distribuição de score de treino e teste para o modelo {model_name}')
            try:
                # Distribuição de score pros dados de treino
                sns.kdeplot(train_scores[y_train == 1], ax=axs[i, 0], label='y=1', shade=shade, color='crimson')
                sns.kdeplot(train_scores[y_train == 0], ax=axs[i, 0], label='y=0', shade=shade, color='darkslateblue')
                axs[i, 0].set_title(f'Distribuição de Score - {model_name} - Treino')
                axs[i, 0].legend()
                axs[i, 0].set_xlabel('Score')
                format_spines(axs[i, 0], right_border=False)

                # Distribuição de score pros dados de teste
                sns.kdeplot(test_scores[y_test == 1], ax=axs[i, 1], label='y=1', shade=shade, color='crimson')
                sns.kdeplot(test_scores[y_test == 0], ax=axs[i, 1], label='y=0', shade=shade, color='darkslateblue')
                axs[i, 1].set_title(f'Distribuição de Score - {model_name} - Teste')
                axs[i, 1].legend()
                axs[i, 1].set_xlabel('Score')
                format_spines(axs[i, 1], right_border=False)
                i += 1
            except Exception as e:
                logger.error(f'Erro ao plotar a curva para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'score_distribution.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   

    def plot_score_bins(self, bin_range=.20, **kwargs):
        """
        Método responsável por realizar a plotagem da distribuição de scores em faixas específicas

        Parâmetros
        ----------
        :param bin_range: intervalo de separação das faixas de score [type: float, default=.25]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='score_bins.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento do gráfico de distribuição especificado
        
        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_score_distribution(output_path=OUTPUT_PATH)
        """

        logger.debug('Inicializando plotagem de distribuição de score em faixas para os modelos')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig1, axs1 = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))
        fig2, axs2 = plt.subplots(nrows=nrows, ncols=2, figsize=(16, nrows * 4))

        # Retornando parâmetros de faixas
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins)) if i > 0]

        # Iterando sobre os classificadores da classe
        for model_name, model_info in self.classifiers_info.items():

            logger.debug(f'Calculando parâmetros de plotagem para o modelo {model_name}')
            try:
                # Retrieving the train scores and creating a DataFrame
                train_scores = model_info['train_scores']
                y_train = model_info['model_data']['y_train']
                df_train_scores = pd.DataFrame({})
                df_train_scores['scores'] = train_scores
                df_train_scores['target'] = y_train
                df_train_scores['faixa'] = pd.cut(train_scores, bins, labels=bins_labels)

                # Computing the distribution for each bin
                df_train_rate = pd.crosstab(df_train_scores['faixa'], df_train_scores['target'])
                df_train_percent = df_train_rate.div(df_train_rate.sum(1).astype(float), axis=0)

                # Retrieving the test scores and creating a DataFrame
                test_scores = model_info['test_scores']
                y_test = model_info['model_data']['y_test']
                df_test_scores = pd.DataFrame({})
                df_test_scores['scores'] = test_scores
                df_test_scores['target'] = y_test
                df_test_scores['faixa'] = pd.cut(test_scores, bins, labels=bins_labels)

                # Computing the distribution for each bin
                df_test_rate = pd.crosstab(df_test_scores['faixa'], df_test_scores['target'])
                df_test_percent = df_test_rate.div(df_test_rate.sum(1).astype(float), axis=0)
            except Exception as e:
                logger.error(f'Erro ao calcular parâmetros para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando distribuição do score em faixas para o modelo {model_name}')
            try:
                sns.countplot(x='faixa', data=df_train_scores, ax=axs1[i, 0], hue='target', palette=['darkslateblue', 'crimson'])
                sns.countplot(x='faixa', data=df_test_scores, ax=axs1[i, 1], hue='target', palette=['darkslateblue', 'crimson'])

                # Formatando legendas e títulos
                axs1[i, 0].legend(loc='upper right')
                axs1[i, 1].legend(loc='upper right')
                axs1[i, 0].set_title(f'Distribuição de Score em Faixas (Volume) - {model_name} - Treino', size=14)
                axs1[i, 1].set_title(f'Distribuição de Score em Faixas (Volume) - {model_name} - Teste', size=14)

                # Adicionando rótulos
                AnnotateBars(n_dec=0, color='black', font_size=12).vertical(axs1[i, 0])
                AnnotateBars(n_dec=0, color='black', font_size=12).vertical(axs1[i, 1])

                # Formatando eixos
                format_spines(axs1[i, 0], right_border=False)
                format_spines(axs1[i, 1], right_border=False)

                logger.debug(f'Plotando percentual de volumetria da faixa para o modelo {model_name}')
                for df_percent, ax in zip([df_train_percent, df_test_percent], [axs2[i, 0], axs2[i, 1]]):
                    df_percent.plot(kind='bar', ax=ax, stacked=True, color=['darkslateblue', 'crimson'], width=0.6)

                    for p in ax.patches:
                        # Coletando parâmetros para inserção de rótulos
                        height = p.get_height()
                        width = p.get_width()
                        x = p.get_x()
                        y = p.get_y()

                        # Formatando parâmetros
                        label_text = f'{round(100 * height, 1)}%'
                        label_x = x + width - 0.30
                        label_y = y + height / 2
                        ax.text(label_x, label_y, label_text, ha='center', va='center', color='white',
                                fontweight='bold', size=10)
                    format_spines(ax, right_border=False)

                    # Formatando legendas e títulos
                    axs2[i, 0].set_title(f'Distribuição do Score em Faixas (Percentual) - {model_name} - Treino')
                    axs2[i, 1].set_title(f'Distribuição do Score em Faixas (Percentual) - {model_name} - Teste')
                i += 1

            except Exception as e:
                logger.error(f'Erro ao plotar gráfico para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        fig1.tight_layout()
        fig2.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            self.save_fig(fig1, output_path, img_name='score_bins.png')
            self.save_fig(fig2, output_path, img_name='score_bins_percent.png')        

    def plot_learning_curve(self, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), **kwargs):
        """
        Método responsável por calcular a curva de aprendizado para um modelo treinado
        
        Parâmetros
        ----------
        :param model_name: chave de referência para análise de um modelo já treinado[type: string]
        :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 6)]
        :param ylim: climite do eixo vertical [type: int, default=None]
        :param cv: k-folds utilizados na validação cruzada para levantamento de informações [type: int, default=5]
        :param n_jobs: número de processadores utilizado no levantamento das informações [type: int, default=1]
        :param train_sizes: array de passos utilizados na curva [type: np.array, default=np.linspace(.1, 1.0, 10)]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento do gráfico de distribuição especificado

        Aplicação
        -----------
        trainer.plot_learning_curve(model_name='LightGBM')
        """

        logger.debug(f'Inicializando plotagem da curvas de aprendizado dos modelos')
        i = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))

        # Iterando sobre os classificadores presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            ax = axs[i]
            logger.debug(f'Retornando parâmetros pro modelo {model_name} e aplicando método learning_curve')
            try:
                model = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Chamando função learning_curve para retornar os scores de treino e validação
                train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

                # Computando médias e desvio padrão (treino e validação)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                val_scores_std = np.std(val_scores, axis=1)
            except Exception as e:
                logger.error(f'Erro ao retornar parâmetros e scores pro modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando curvas de aprendizado de treino e validação para o modelo {model_name}')
            try:
                # Resultados utilizando dados de treino
                ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
                ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                                alpha=0.1, color='blue')

                # Resultados utilizando dados de validação (cross validation)
                ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
                ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                                alpha=0.1, color='crimson')

                # Customizando plotagem
                ax.set_title(f'Model {model_name} - Learning Curve', size=14)
                ax.set_xlabel('Training size (m)')
                ax.set_ylabel('Score')
                ax.grid(True)
                ax.legend(loc='best')
            except Exception as e:
                logger.error(f'Erro ao plotar curva de aprendizado para o modelo {model_name}. Exception lançada: {e}')
                continue
            i += 1
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'learning_curve.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename) 

    def plot_shap_analysis(self, model_name, features, figsize=(16, 10), **kwargs):
        """
        Método responsável por plotar a análise shap pras features em um determinado modelo
        
        Parâmetros
        ----------
        :param model_name: chave de um classificador específico já treinado na classe [type: string]
        :param features: lista de features do dataset [type: list]
        :param figsize: tamanho da figure de plotagem [type: tuple, default=(16, 10)]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da análise shap especificada
        """

        logger.debug(f'Explicando o modelo {model_name} através da análise shap')
        try:
            model_info = self.classifiers_info[model_name]
            model = model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

        logger.debug(f'Retornando parâmetros da classe para o modelo {model_name}')
        try:
            # Retornando parâmetros do modelo
            X_train = model_info['model_data']['X_train']
            X_test = model_info['model_data']['X_test']
            df_train = pd.DataFrame(X_train, columns=features)
            df_test = pd.DataFrame(X_train, columns=features)
        except Exception as e:
            logger.error(f'Erro ao retornar parâmetros para o modelo {model_name}. Exception lançada: {e}')

        logger.debug(f'Criando explainer e gerando valores shap para o modelo {model_name}')
        try:
            explainer = shap.TreeExplainer(model, df_train)
            shap_values = explainer.shap_values(df_test)
        except Exception as e:
            try:
                logger.warning(f'TreeExplainer não se encaixa no modelo {model_name}. Tentando LinearExplainer')
                explainer = shap.LinearExplainer(model, df_train)
                shap_values = explainer.shap_values(df_test, check_additivity=False)
            except Exception as e:
                logger.error(f'Não foi possível retornar os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                return

        logger.debug(f'Plotando análise shap para o modelo {model_name}')
        try:
            fig, ax = plt.subplots(figsize=figsize)
            try:
                shap.summary_plot(shap_values, df_test, plot_type='violin', show=False)
            except Exception as e:
                shap.summary_plot(shap_values[1], df_test, plot_type='violin', show=False)
            plt.title(f'Shap Analysis (violin) para o modelo {model_name}')
            if 'save' in kwargs and bool(kwargs['save']):
                output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
                output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else f'shap_analysis_{model_name}.png'
                self.save_fig(fig, output_path, img_name=output_filename)
        except Exception as e:
            logger.error(f'Erro ao plotar análise shap para o modelo {model_name}. Exception lançada: {e}')
            return 

    def visual_analysis(self, features, metrics=True, feat_imp=True, cfmx=True, roc=True, score_dist=True, score_bins=True, 
                        learn_curve=True, model_shap=None, show=False, save=True, output_path=os.path.join(os.getcwd(), 'output/imgs')):
        """
        Método responsável por consolidar análises gráficas no processo de modelagem

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param metrics: flag inficativo da execução do método plot_metrics() [type: bool, default=True]
        :param feat_imp: flag indicativo da execução da método plot_feature_importance() [type: bool, default=True]
        :param cfmx: flag indicativo da execução da método plot_confusion_matrix() [type: bool, default=True]
        :param roc: flag indicativo da execução da método plot_roc_curve() [type: bool, default=True]
        :param score_dist: flag indicativo da execução da método plot_score_distribution() [type: bool, default=True]
        :param score_bins: flag indicativo da execução da método plot_score_bins() [type: bool, default=True]
        :param learn_curve: flag indicativo da execução da método plot_learning_curve() [type: bool, default=True]
        :param model_shap: chave do modelo a ser utilizado na análise shap [type: string, default=None]
        :param show: flag indicativo para mostragem das figuras em jupyter notebook [type: bool, default=False]
        :param save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
        :param output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']        

        Retorno
        -------
        Este método não retorna nada além das imagens devidamente salvas no diretório destino

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train, y_train, X_test, y_test)        
        """

        # Verificando parâmetro para mostrar
        backend_ = mpl.get_backend()
        if not show:
            mpl.use('Agg')

        logger.debug(f'Inicializando análise gráfica nos modelos treinados')
        try:
            # Verificando plotagem das métricas
            if metrics:
                self.plot_metrics(save=save, output_path=output_path)

            # Verificando plotagem de feature importance
            if feat_imp:
                self.plot_feature_importance(features=features, save=save, output_path=output_path)

            # Verificando plotagem de matriz de confusão
            if cfmx:
                self.plot_confusion_matrix(save=save, output_path=output_path)
            
            # Verificando plotagem de curva ROC
            if roc:
                self.plot_roc_curve(save=save, output_path=output_path)

            # Verificando plotagem de distribuição dos scores
            if score_dist:
                self.plot_score_distribution(save=save, output_path=output_path)

            # Verificando plotagem de distribuição do score em faixa
            if score_bins:
                self.plot_score_bins(save=save, output_path=output_path)

            # Verificando plotagem de curva de aprendizado
            if learn_curve:
                self.plot_learning_curve(save=save, output_path=output_path)

            if model_shap is not None:
                self.plot_shap_analysis(save=save, model_name=model_shap, features=features, output_path=output_path)

        except Exception as e:
            logger.error(f'Erro ao plotar análises gráficas. Exception lançada: {e}')

        # Resetando configurações
        mpl.use(backend_)

    def _get_estimator(self, model_name):
        """
        Método responsável por retornar o estimator de um modelo selecionado

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model: estimator do modelo já treinado
        """

        logger.debug(f'Retornando estimator do modelo {model_name} já treinado')
        try:
            model_info = self.classifiers_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

    def _get_metrics(self, model_name):
        """
        Método responsável por retornar as métricas obtidas no treinamento

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return model_performance: DataFrame contendo as métricas dos modelos treinados [type: pd.DataFrame]
        """

        logger.debug(f'Retornando as métricas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo e métricas já salvas
            model_info = self.classifiers_info[model_name]
            train_performance = model_info['train_performance']
            test_performance = model_info['test_performance']
            model_performance = train_performance.append(test_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')

    def _get_model_info(self, model_name):
        """
        Método responsável por coletar as informações registradas de um determinado modelo da classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model_info: dicionário com informações registradas do modelo [type: dict]
            model_info = {
                'estimator': model,
                'train_scores': np.array,
                'test_scores': np.array,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_test': np.array,
                    'y_test': np.array,
                'feature_importances': pd.DataFrame
                }
            }
        """

        logger.debug(f'Retornando informações registradas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.classifiers_info[model_name]
        except Exception as e:
            logger.error(f'Erro ao retornar informações do modelo {model_name}. Exception lançada: {e}')

    def _get_classifiers_info(self):
        """
        Método responsável por retornar o dicionário classifiers_info contendo todas as informações de todos os modelos

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return classifiers_info: dicionário completo dos modelos presentes na classe
            classifiers_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_scores': np.array,
                                'test_scores': np.array,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_test': np.array,
                                    'y_test': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.classifiers_info


"""
---------------------------------------------------
--------------- 2. CLASSIFICAÇÃO ------------------
         2.1 Classificador Multiclasse
---------------------------------------------------
"""

class ClassificadorMulticlasse:
    """
    Classe responsável por consolidar métodos úteis para o treinamento
    e avaliação de modelos de classificação multiclasse em um contexto de
    aprendizado supervisionado
    """

    def __init__(self, encoded_target=False):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.classifiers_info = {}
        self.encoded_target = encoded_target

    def fit(self, set_classifiers, X_train, y_train, **kwargs):
        """
        Método responsável por treinar cada um dos classificadores contidos no dicionário
        set_classifiers através da aplicação das regras estabelecidas pelos argumentos do método

        Parâmetros
        ----------
        :param set_classifiers: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_classifiers = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features do modelo a ser treinado [type: np.array]
        :param y_train: array contendo variável target do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: -1]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento de objetos do modelo [type: string, default=cwd() + 'output/models']
            :arg model_ext: extensão do objeto gerado (pkl ou joblib) - sem o ponto [type: string, default='pkl']

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train_prep, y_train)
        """

        # Referenciando argumentos adicionais
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_classifiers.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                model_key = model_name + approach
                logger.debug(f'Treinando modelo {model_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.classifiers_info[model_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Retornando parâmetros em kwargs
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'accuracy'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparando e aplicando busca
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.classifiers_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.classifiers_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Erro ao aplicar RandomizedSearch. Exception lançada: {te}')
                    return

                # Validando salvamento de objetos pkl dos modelos
                if 'save' in kwargs and bool(kwargs['save']):
                    logger.debug(f'Salvando arquivo pkl do modelo {model_name} treinado')
                    model = self.classifiers_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'

                    self.save_model(model, output_path=output_path, filename=model_name.lower() + '.' + model_ext)

        except AttributeError as ae:
            logger.error(f'Erro ao treinar modelos. Exception lançada: {ae}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')

    def compute_train_performance(self, model_name, estimator, X, y, cv=5, target_names=None):
        """
        Método responsável por aplicar validação cruzada para retornar a amédia das principais métricas de avaliação
        de um modelo de classificação. Na prática, esse método é chamado por um outro método em uma camada
        superior da classe para medição de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param target_names: lista com referências para as classes [type: list, default=None]

        Retorno
        -------
        :return train_performance: DataFrame contendo as métricas calculadas usando validação cruzada [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computando métricas utilizando validação cruzada
        logger.debug(f'Computando métricas do modelo {model_name} utilizando validação cruzada com {cv} K-folds')
        try:
            # Iniciando cronômetro e computando predições via validação cruzada
            t0 = time.time()
            y_pred = cross_val_predict(estimator, X, y, cv=cv)
            
            # Construindo classification report
            cr = pd.DataFrame(classification_report(y, y_pred, output_dict=True, target_names=target_names)).T
            
            # Definindo fluxo para target codificado ou nao
            if self.encoded_target:
                n_classes = len(cr) - 4
                acc = [accuracy_score(y.T[i], y_pred.T[i]) for i in range(n_classes)]
            else:
                n_classes = len(cr) - 3
                acc = cr.loc['accuracy', :].values
            
            # Customizando classification report
            cr_custom = cr.iloc[:n_classes, :-1]
            cr_custom['model'] = model_name
            cr_custom.reset_index(inplace=True)
            cr_custom.columns = ['class'] + list(cr_custom.columns[1:])
            
            # Calculando acurácia para cada classe
            cr_custom['accuracy'] = acc
            cr_custom['approach'] = f'Treino {cv} K-folds'
            cr_cols = ['model', 'approach', 'class', 'accuracy', 'precision', 'recall', 'f1-score']
            train_performance = cr_custom.loc[:, cr_cols]

            # Adicionando medição de tempo no DataFrame
            t1 = time.time()
            delta_time = t1 - t0
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de treino em {round(delta_time, 3)} segundos')

            return train_performance

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def compute_val_performance(self, model_name, estimator, X, y, target_names=None):
        """
        Método responsável por aplicar retornar as principais métricas do model utilizando dados de validação.
        Na prática, esse método é chamado por um outro método em uma camada superior da classe para medição 
        de performance em treino e em validação

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param target_names: lista com referências para as classes [type: list, default=None]

        Retorno
        -------
        :return val_performance: DataFrame contendo as métricas calculadas nos dados de validação [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        val_performance = trainer.compute_val_performance(model_name, estimator, X_val, y_val)
        """

        # Predicting data using the trained model and computing probabilities
        logger.debug(f'Computando métricas do modelo {model_name} utilizando dados de teste')
        try:
            # Iniciando cronômetro e computando predições puras
            t0 = time.time()
            y_pred = estimator.predict(X)

            # Construindo classification report
            cr = pd.DataFrame(classification_report(y, y_pred, output_dict=True, target_names=target_names)).T
            
            # Definindo fluxo para target codificado ou nao
            if self.encoded_target:
                n_classes = len(cr) - 4
                acc = [accuracy_score(y.T[i], y_pred.T[i]) for i in range(n_classes)]
            else:
                n_classes = len(cr) - 3
                acc = cr.loc['accuracy', :].values
                
            # Customizando classification report
            cr_custom = cr.iloc[:n_classes, :-1]
            cr_custom['model'] = model_name
            cr_custom.reset_index(inplace=True)
            cr_custom.columns = ['class'] + list(cr_custom.columns[1:])

            # Calculando acurácia para cada classe
            cr_custom['accuracy'] = acc
            cr_custom['approach'] = f'Validation set'
            cr_cols = ['model', 'approach', 'class', 'accuracy', 'precision', 'recall', 'f1-score']
            val_performance = cr_custom.loc[:, cr_cols]

            # Adicionando medição de tempo no DataFrame
            t1 = time.time()
            delta_time = t1 - t0
            val_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de validação em {round(delta_time, 3)} segundos')

            return val_performance

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def evaluate_performance(self, X_train, y_train, X_val, y_val, cv=5, target_names=None, **kwargs):
        """
        Método responsável por executar e retornar métricas dos classificadores em treino (média do resultado
        da validação cruzada com cv K-fols) e teste
        
        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param target_names: lista com referências para as classes [type: list, default=None]
        :param encoded_target: define a aplicação do encoding no array target [type: bool, default=True]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
        
        Retorno
        -------
        :return df_performances: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

        Aplicação
        -----------
        # Treinando modelo e avaliando performance em treino e teste
        trainer = ClassificadorBinario()
        trainer.fit(estimator, X_train, X_test)

        # Definindo dicionário de controle do resultado
        df_performance = trainer.evaluate_performance(X_train, y_train, X_val, y_val, save=True, output_path=caminho)
        """

        # DataFrame vazio para armazenamento das métrics
        df_performances = pd.DataFrame({})

        # Iterando sobre todos os classificadores da classe
        for model_name, model_info in self.classifiers_info.items():

            # Validando se o modelo já foi treinado (dicionário model_info já terá a chave 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['val_performance'])
                continue

            # Retornando modelo a ser avaliado
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Erro ao retornar a chave "estimator" do dicionário model_info. Modelo {model_name} não treinado')
                continue

            # Computando performance em treino e em teste
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, 
                                                               cv=cv, target_names=target_names)
            val_performance = self.compute_val_performance(model_name, estimator, X_val, y_val, 
                                                           target_names=target_names)

            # Adicionando os resultados ao atributo classifiers_info
            self.classifiers_info[model_name]['train_performance'] = train_performance
            self.classifiers_info[model_name]['val_performance'] = val_performance

            # Construindo DataFrame com as métricas retornadas
            model_performance = train_performance.append(val_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Salvando alguns atributos no dicionário classifiers_info para acessos futuros
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val
            }
            model_info['model_data'] = model_data

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            self.save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances

    def plot_metrics(self, df_metrics=None, idx_cols=['model', 'class', 'approach'], 
                 value_cols=['accuracy', 'precision', 'recall', 'f1-score'], font_scale=1.2,
                 row='approach', col='model', margin_titles=True, x='class', y='value', hue='variable', 
                 palette='rainbow', legend_loc='center right', x_rotation=30, annot_ndec=2, 
                 annot_fontsize=8, **kwargs):
        """
        Método responsável por plotar os resultados das métricas dos classificadores selecionados

        Parâmetros
        ----------
        :param df_metrics: DataFrame específico contendo as métricas já calculadas pelo objeto [type: pd.DataFrame]
        :param idx_cols: colunas da base de métricas a serem pivotadas 
            [type: list, default=['model', 'class', 'approach']]
        :param value_cols: colunas da base de métricas com os valores das métricas
            [type: list, default=['accuracy', 'precision', 'recall', 'f1-score']
        :param font_scale: escala de fonte do FacetGrid [type: float, default=1.2]
        :param row: referência separadora de linhas do grid [type: string, default='approach']
        :param col: referência separadora de colunas do grid [type: string, default='model']
        :param margin_titles: flag para plotagem dos títulos do grid [type: bool, default=True]
        :param x: eixo x da plotagem [type: string, default='class']
        :param y: eixo y da plotagem [type: string, default='value']
        :param hue: separação das barras em cada plotagem [type: string, default='variable']
        :param palette: paleta de cores utilizada [type: string, default='rainbow']
        :param legend_loc: posicionamento da legenda [type: string, default='center right']
        :param x_rotation: rotação dos labels no eixo x [type: int, default=30]
        :param annot_ndec: casas decimais dos labels nas barras [type: int, default=2]
        :param annot_fontsize: tamanho da fonte dos labels nas barras [type: int, default=8]

        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics_comparison.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da plotagem devidamente salva no diretório destino
        """

        # Verificando a necessidade de calcular as métricas
        if df_metrics is None:
            # Retornando elementos necessários para o cálculo da performance
            try:
                X_train = kwargs['X_train']
                y_train = kwargs['y_train']
                X_val = kwargs['X_val']
                y_val = kwargs['y_val']
                target_names = kwargs['target_names']
                df_metrics = self.evaluate_performance(X_train, y_train, X_val, y_val, 
                                                       target_names=target_names)
            except Exception as e:
                print(f'Erro ao computar as métricas. Exception lançada: {e}')
                return

        # Pivotando DataFrame de métricas de modo a facilitar a plotagem
        metrics_melt = pd.melt(df_metrics, id_vars=idx_cols, value_vars=value_cols)

        # Criando FatecGrid e plotando gráfico
        with sns.plotting_context('notebook', font_scale=1.2):
            g = sns.FacetGrid(metrics_melt, row=row, col=col, margin_titles=margin_titles)
            g.map_dataframe(sns.barplot, x=x, y=y, hue=hue, palette=palette)

        # Customizando figura
        figsize = (17, 5 * len(np.unique(metrics_melt[row])))
        g.fig.set_size_inches(figsize)
        plt.legend(loc=legend_loc)
        g.set_xticklabels(rotation=x_rotation)

        for axs in g.axes:
            for ax in axs:
                AnnotateBars(n_dec=annot_ndec, color='black', font_size=annot_fontsize).vertical(ax)

        plt.tight_layout()
    
    def custom_confusion_matrix(self, model_name, y_true, y_pred, classes, cmap, normalize=False):
        """
        Método utilizada para plotar uma matriz de confusão customizada para um único modelo da classe. Em geral,
        esse método pode ser chamado por um método de camada superior para plotagem de matrizes para todos os
        modelos presentes na classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param y_true: array contendo a variável target do dataset [type: np.array]
        :param y_pred: array com as predições retornadas pelo respectivo modelo [type: np.array]
        :param classes: nomenclatura das classes da matriz [type: list]
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        -----------
        Visualizar o método self.plot_confusion_matrix()
        """

        # Retornando a matriz de confusão usando função do sklearn
        if not self.encoded_target:
            y_true = pd.get_dummies(y_true).values
            y_pred = pd.get_dummies(y_pred).values
            
        conf_mx = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        
        
        # Plotando matriz
        plt.imshow(conf_mx, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))

        # Customizando eixos
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Customizando entradas
        fmt = '.2f' if normalize else 'd'
        thresh = conf_mx.max() / 2.
        for i, j in itertools.product(range(conf_mx.shape[0]), range(conf_mx.shape[1])):
            plt.text(j, i, format(conf_mx[i, j]),
                     horizontalalignment='center',
                     color='white' if conf_mx[i, j] > thresh else 'black')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'{model_name}\nConfusion Matrix', size=12)

    def plot_confusion_matrix(self, classes=None, cmap=plt.cm.Blues, normalize=False, **kwargs):
        """
        Método responsável por plotar gráficos de matriz de confusão usando dados de treino e teste
        para todos os modelos presentes no dicionárion de classificadores self.classifiers_info

        Parâmetros
        ----------
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhuma variável, além da plotagem da matriz especificada

        Aplicação
        ---------
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        trainer.plot_confusion_matrix(output_path=OUTPUT_PATH)
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem da matriz de confusão para os modelos')
        k = 1
        nrows = len(self.classifiers_info.keys())
        fig = plt.figure(figsize=(14, nrows * 5.5))
        sns.set(style='white', palette='muted', color_codes=True)

        # Iterando sobre cada classificador da classe
        for model_name, model_info in self.classifiers_info.items():
            logger.debug(f'Retornando dados de treino e validação para o modelo {model_name}')
            try:
                # Retornando dados para cada modelo
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']
                X_val = model_info['model_data']['X_val']
                y_val = model_info['model_data']['y_val']
                
                # Definindo classes para a plotagem
                if classes is None:
                    classes = list(range(y_train.shape[1]))
                
            except Exception as e:
                logger.error(f'Erro ao retornar dados para o modelo {model_name}. Exception lançada: {e}')
                continue

            # Realizando predições em treino (cross validation) e teste
            logger.debug(f'Realizando predições para os dados de treino e validação ({model_name})')
            try:
                train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
                val_pred = model_info['estimator'].predict(X_val)
            except Exception as e:
                logger.error(f'Erro ao realizar predições para o modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Gerando matriz de confusão para o modelo {model_name}')
            try:
                # Plotando matriz utilizando dados de treino
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Train', y_train, train_pred, classes=classes, 
                                             cmap=cmap, normalize=normalize)
                k += 1

                # Plotando matriz utilizando dados de validação
                plt.subplot(nrows, 2, k)
                self.custom_confusion_matrix(model_name + ' Validation', y_val, val_pred, classes=classes, 
                                             cmap=plt.cm.Greens, normalize=normalize)
                k += 1
                logger.info(f'Matriz de confusão gerada para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar a matriz para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'confusion_matrix.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   
   
    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Método responsável por realizar uma plotagem gráfica das variáveis mais importantes pro modelo

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param top_n: quantidade de features a ser considerada na plotagem [type: int, default=20]
        :param palette: paleta de cores utilizazda na plotagem [type: string, default='viridis']
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='feature_importances.png']

        Retorno
        -------
        Este método não retorna nada além da imagem devidamente salva no diretório destino
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.classifiers_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.classifiers_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], 
                            palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)
    
    def plot_shap_analysis(self, model_name, features, n_classes, figsize=(16, 10), target_names=None,
                           **kwargs):
        """
        Método responsável por plotar a análise shap pras features em um determinado modelo
        
        Parâmetros
        ----------
        :param model_name: chave de um classificador específico já treinado na classe [type: string]
        :param features: lista de features do dataset [type: list]
        :param n_classes: número de classes presentes no modelo [type: int]
        :param figsize: tamanho da figure de plotagem [type: tuple, default=(16, 10)]
        :param target_names: nomes das classes para título da plotagem [type: list, default=None]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da análise shap especificada
        """

        logger.debug(f'Explicando o modelo {model_name} através da análise shap')
        try:
            model_info = self.classifiers_info[model_name]
            model = model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

        logger.debug(f'Retornando parâmetros da classe para o modelo {model_name}')
        try:
            # Retornando parâmetros do modelo
            X_train = model_info['model_data']['X_train']
            X_val = model_info['model_data']['X_val']
            df_train = pd.DataFrame(X_train, columns=features)
            df_val = pd.DataFrame(X_val, columns=features)
        except Exception as e:
            logger.error(f'Erro ao retornar parâmetros para o modelo {model_name}. Exception lançada: {e}')

        logger.debug(f'Criando explainer e gerando valores shap para o modelo {model_name}')
        try:
            explainer = shap.TreeExplainer(model, df_train)
            shap_values = explainer.shap_values(df_val)
        except Exception as e:
            try:
                logger.warning(f'TreeExplainer não se encaixa no modelo {model_name}. Tentando LinearExplainer')
                explainer = shap.LinearExplainer(model, df_train)
                shap_values = explainer.shap_values(df_val, check_additivity=False)
            except Exception as e:
                logger.error(f'Não foi possível retornar os parâmetros para o modelo {model_name}. Exception lançada: {e}')
                return
        
        # Configurando plotagem
        nrows = ceil(n_classes / 2)
        fig = plt.figure()
        if not target_names:
            target_names = [f'Classe {i}' for i in range(1, n_classes + 1)]
        
        # Iterando sobre as classes
        logger.debug(f'Plotando análise shap para o modelo {model_name}')
        for c in range(1, n_classes + 1):
            plt.subplot(nrows, 2, c)
            try:
                shap.summary_plot(shap_values[c - 1], X_val, plot_type='violin', show=False, 
                                  plot_size=(15, nrows * 9))
            except Exception as e:
                logger.error(f'Erro ao plotar análise shap para o modelo {model_name}. Exception lançada: {e}')
                return
            
            # Configurações finais e salvamento da imagem
            plt.title(f'Shap summary plot: {target_names[c - 1]}', size=15)
            plt.tight_layout()
            if 'save' in kwargs and bool(kwargs['save']):
                output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
                output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else f'shap_analysis_{model_name}.png'
                self.save_fig(fig, output_path, img_name=output_filename)
    
    def get_estimator(self, model_name):
        """
        Método responsável por retornar o estimator de um modelo selecionado

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model: estimator do modelo já treinado
        """

        logger.debug(f'Retornando estimator do modelo {model_name} já treinado')
        try:
            model_info = self.classifiers_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

    def get_metrics(self, model_name):
        """
        Método responsável por retornar as métricas obtidas no treinamento

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return model_performance: DataFrame contendo as métricas dos modelos treinados [type: pd.DataFrame]
        """

        logger.debug(f'Retornando as métricas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo e métricas já salvas
            model_info = self.classifiers_info[model_name]
            train_performance = model_info['train_performance']
            test_performance = model_info['test_performance']
            model_performance = train_performance.append(test_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')

    def get_model_info(self, model_name):
        """
        Método responsável por coletar as informações registradas de um determinado modelo da classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário classifiers_info da classe [type: string]

        Retorno
        -------
        :return model_info: dicionário com informações registradas do modelo [type: dict]
            model_info = {
                'estimator': model,
                'train_scores': np.array,
                'test_scores': np.array,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_test': np.array,
                    'y_test': np.array,
                'feature_importances': pd.DataFrame
                }
            }
        """

        logger.debug(f'Retornando informações registradas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.classifiers_info[model_name]
        except Exception as e:
            logger.error(f'Erro ao retornar informações do modelo {model_name}. Exception lançada: {e}')

    def get_classifiers_info(self):
        """
        Método responsável por retornar o dicionário classifiers_info contendo todas as informações de todos os modelos

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return classifiers_info: dicionário completo dos modelos presentes na classe
            classifiers_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_scores': np.array,
                                'test_scores': np.array,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_test': np.array,
                                    'y_test': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.classifiers_info


"""
---------------------------------------------------
--------------- 2. CLASSIFICAÇÃO ------------------
       2.2. Funções de Análise de Modelos
---------------------------------------------------
"""

def save_fig(fig, output_path, img_name, tight_layout=True, dpi=300):
    """
    Função responsável por salvar imagens geradas pelo matplotlib/seaborn

    Parâmetros
    ----------
    :param fig: figura criada pelo matplotlib para a plotagem gráfica [type: plt.figure]
    :param output_file: caminho final a ser salvo (+ nome do arquivo em formato png) [type: string]
    :param tight_layout: flag que define o acerto da imagem [type: bool, default=True]
    :param dpi: resolução da imagem a ser salva [type: int, default=300]

    Retorno
    -------
    Este método não retorna nenhum parâmetro além do salvamento da imagem em diretório especificado

    Aplicação
    ---------
    fig, ax = plt.subplots()
    save_fig(fig, output_path='output/imgs', img_name='imagem.png')
    """

    # Verificando se diretório existe
    if not os.path.isdir(output_path):
        logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
        try:
            os.makedirs(output_path)
        except Exception as e:
            logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')

    # Acertando layout da imagem
    if tight_layout:
        fig.tight_layout()

    logger.debug('Salvando imagem no diretório especificado')
    try:
        output_file = os.path.join(output_path, img_name)
        fig.savefig(output_file, dpi=300)
        logger.info(f'Imagem salva com sucesso em {output_file}')
    except Exception as e:
        logger.error(f'Erro ao salvar imagem. Exception lançada: {e}')

def clf_plot_feature_score_dist(data, feature, model, figsize=(16, 8), kind='boxplot', bin_range=.20, palette='magma',
                            save=True, output_path='output/imgs', img_name='feature_score_dist.png'):
    """
    Função responsável por plotar a distribuição de uma determinada variável do dataset em 
    faixas de score obtidas através das probabilidades de um modelo preditivo
    
    Parâmetros
    ----------
    :param data: base de dados a ser analisada [type: pd.DataFrame]
    :param feature: referência da coluna a ser analisada [type: string]
    :param model: modelo preditivo já treinado [type: estimator]
    :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 10)]
    :param kind: tipo de gráfico a ser plotado [type: string, default='boxplot']
        *opções: 'boxplot', 'boxenplot', 'stripplot', 'kdeplot'
    :param bin_range: separador das faixas de score do modelo [type: float, default=.20]
    :param palette: colormap utilizado na plotagem gráfica [type: string, default='magma']
    :param save: flag para indicar o salvamento da imagem gerada [type: bool, default=True]
    :param output_path: diretório de salvamento da imagem gerada [type: string, default='output/imgs']
    :param img_name: nome do arquivo a ser salvo [type: string, default='feature_model_score_dist.png']
        
    Retorno
    -------
    Essa função não retorna nenhum parâmetro além da figura plotada utilizando o método boxplot do seaborn
    
    Aplicação
    ---------
    # Treinando modelo preditivo
    model = RandomForestClassifier()
    model.fit(X_train_prep, y_train)
    
    # Analisando distribuição de uma variável
    plot_feature_score_proba(X_train_prep, feature='numerical_col', model=model)
    """
    
    model_name = type(model).__name__
    logger.debug(f'Retornando probabilidades do modelo {model_name}')
    df = data.copy()
    try:
        y_scores = model.predict_proba(df)[:, 1]
    except NotFittedError as nfe:
        logger.error(f'Modelo {model_name} não foi treinado, impossibilitando o retorno das probabilidades. Exception: {nfe}')
        return
    
    logger.debug('Criando faixas com as probabilidades obtidas')
    try:
        bins = np.arange(0, 1.01, bin_range)
        bins_labels = [str(round(list(bins)[i - 1], 2)) + ' a ' + str(round(list(bins)[i], 2)) for i in range(len(bins)) if i > 0]
        df['faixa'] = pd.cut(y_scores, bins, labels=bins_labels)
    except Exception as e:
        logger.error(f'Erro ao criar as faixas do modelo. Exception lançada: {e}')
        return
    
    logger.debug(f'Plotando análise de distribuição {kind} para a variável {feature}')
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Validando tipo de plotagem
        if kind == 'boxplot':
            sns.boxplot(x='faixa', y=feature, data=df, palette=palette)
        elif kind == 'boxenplot':
            sns.boxenplot(x='faixa', y=feature, data=df, palette=palette)
        elif kind == 'stripplot':
            sns.stripplot(x='faixa', y=feature, data=df, palette=palette)
        elif kind == 'kdeplot':
            # Plotando um gráfico de distribuição pra cada categoria
            for label in bins_labels:
                sns.kdeplot(df[df['faixa']==label][feature], ax=ax, label=label, shade=True)
                plt.legend()
        else:
            logger.warning(f'Tipo de plotagem {kind} não disponível. Selecione entre ["boxplot", "boxenplot", "stripplot"]')
            return
        
        ax.set_title(f'Gráfico ${kind}$ de ${feature}$ nas faixas de score do modelo ${model_name}$', size=14)
        format_spines(ax, right_border=False)
    except Exception as e:
        logger.error(f'Erro ao plotar o gráfico para a variável {feature}. Exception: {e}')
        return
    
    # Validando salvamento da figura
    if save:
        save_fig(fig=fig, output_path=output_path, img_name=img_name)

def clf_cv_performance(model, X, y, cv=5, model_name=None):
    """
    Função responsável por calcular as principais métricas de um modelo de classificação
    utilizando validação cruzada
    
    Parâmetros
    ----------
    :param model: estimator do modelo preditivo [type: estimator]
    :param X: dados de entrada do modelo [type: np.array]
    :param y: array de target do modelo [type: np.array]
    :param cv: número de k-folds utilizado na validação cruzada [type: int, default=5]
        
    Retorno
    -------
    :return df_performance: DataFrame contendo as principais métricas de classificação [type: pd.DataFrame]
    
    Aplicação
    ---------
    results = clf_cv_performance(model=model, X=X, y=y)
    """

    # Computing metrics using cross validation
    t0 = time.time()
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision').mean()
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall').mean()
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1').mean()

    # Probas for calculating AUC
    try:
        y_scores = cross_val_predict(model, X, y, cv=cv, method='decision_function')
    except:
        # Tree based models don't have 'decision_function()' method, but 'predict_proba()'
        y_probas = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        y_scores = y_probas[:, 1]
    auc = roc_auc_score(y, y_scores)

    # Creating a DataFrame with metrics
    t1 = time.time()
    delta_time = t1 - t0
    train_performance = {}
    if model_name is None:
        train_performance['model'] = model.__class__.__name__
    else:
        train_performance['model'] = model_name
    train_performance['approach'] = 'Final Model'
    train_performance['acc'] = round(accuracy, 4)
    train_performance['precision'] = round(precision, 4)
    train_performance['recall'] = round(recall, 4)
    train_performance['f1'] = round(f1, 4)
    train_performance['auc'] = round(auc, 4)
    train_performance['total_time'] = round(delta_time, 3)
    df_performance = pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

    # Adding information of measuring and execution time
    cols_performance = list(df_performance.columns)
    df_performance['anomesdia'] = datetime.now().strftime('%Y%m%d')
    df_performance['anomesdia_datetime'] = datetime.now()
    df_performance = df_performance.loc[:, ['anomesdia', 'anomesdia_datetime'] + cols_performance]

    return df_performance



"""
---------------------------------------------------
------------- 3. REGRESSÃO LINEAR -----------------
          3.1 Treinamento e Avaliação
---------------------------------------------------
"""

class RegressorLinear:
    """
    Classe responsável por consolidar métodos úteis para o treinamento
    e avaliação de modelos de regressão linear em um contexto de
    aprendizado supervisionado
    """
    
    def __init__(self):
        """
        Método construtor inicializa dicionário de informações dos modelos treinados
        """
        self.regressors_info = {}
        
    def save_data(self, data, output_path, filename):
        """
        Método responsável por salvar objetos DataFrame em formato csv.

        Parâmetros
        ----------
        :param data: arquivo/objeto a ser salvo [type: pd.DataFrame]
        :param output_path: referência de diretório destino [type: string]
        :param filename: referência do nome do arquivo a ser salvo [type: string]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do arquivo devidamente salvo no diretório

        Aplicação
        ---------
        df = file_generator_method()
        self.save_result(df, output_path=OUTPUT_PATH, filename='arquivo.csv')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
                return

        logger.debug(f'Salvando arquivo no diretório especificado')
        try:
            output_file = os.path.join(output_path, filename)
            data.to_csv(output_file, index=False)
        except Exception as e:
            logger.error(f'Erro ao salvar arquivo {filename}. Exception lançada: {e}')
            
    def save_model(self, model, output_path, filename):
        """
        Método responsável por salvar modelos treinado em fomato pkl.

        Parâmetros
        ----------
        :param model: objeto a ser salvo [type: model]
        :param output_path: referência de diretório destino [type: string]
        :param filename: referência do nome do modelo a ser salvo [type: string]

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do objeto devidamente salvo no diretório

        Aplicação
        ---------
        model = classifiers['estimator']
        self.save_model(model, output_path=OUTPUT_PATH, filename='model.pkl')
        """

        # Verificando se diretório existe
        if not os.path.isdir(output_path):
            logger.warning(f'Diretório {output_path} inexistente. Criando diretório no local especificado')
            try:
                os.makedirs(output_path)
            except Exception as e:
                logger.error(f'Erro ao tentar criar o diretório {output_path}. Exception lançada: {e}')
                return

        logger.debug(f'Salvando modelo pkl no diretório especificado')
        try:
            output_file = os.path.join(output_path, filename)
            joblib.dump(model, output_file)
        except Exception as e:
            logger.error(f'Erro ao salvar modelo {filename}. Exception lançada: {e}')
        
    def fit(self, set_regressors, X_train, y_train, **kwargs):
        """
        Método responsável por treinar cada um dos regressores contidos no dicionário
        set_regressors através da aplicação das regras estabelecidas pelos argumentos do método

        Parâmetros
        ----------
        :param set_regressors: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_regressors = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: features do modelo a ser treinado [type: np.array]
        :param y_train: array contendo variável target do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: -1]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento de objetos do modelo [type: string, default=cwd() + 'output/models']
            :arg model_ext: extensão do objeto gerado (pkl ou joblib) - sem o ponto [type: string, default='pkl']

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = RegressorLinear()
        trainer.fit(set_regressors, X_train_prep, y_train)
        """

        # Referenciando argumentos adicionais
        approach = kwargs['approach'] if 'approach' in kwargs else ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_regressors.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                model_key = model_name + approach
                logger.debug(f'Treinando modelo {model_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.regressors_info[model_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        
                        # Retornando parâmetros em kwargs
                        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'neg_mean_squared_error'
                        cv = kwargs['cv'] if 'cv' in kwargs else 5
                        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
                        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
                        
                        # Preparando e aplicando busca
                        rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv,
                                                        verbose=verbose, random_state=42, n_jobs=n_jobs)
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X_train, y_train)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.regressors_info[model_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.regressors_info[model_key]['estimator'] = model.fit(X_train, y_train)
                except TypeError as te:
                    logger.error(f'Erro ao aplicar RandomizedSearch. Exception lançada: {te}')
                    return

                # Validando salvamento de objetos pkl dos modelos
                if 'save' in kwargs and bool(kwargs['save']):
                    logger.debug(f'Salvando arquivo pkl do modelo {model_name} treinado')
                    model = self.regressors_info[model_key]['estimator']
                    output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
                    model_ext = kwargs['model_ext'] if 'model_ext' in kwargs else 'pkl'

                    self.save_model(model, output_path=output_path, filename=model_name.lower() + '.' + model_ext)

        except AttributeError as ae:
            logger.error('Erro ao treinar modelos. Exception lançada: {ae}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')
            
    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Método responsável por aplicar validação cruzada para retornar a amédia das principais métricas de avaliação
        de um modelo de regressão. Na prática, esse método é chamado por um outro método em uma camada
        superior da classe para medição de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]

        Retorno
        -------
        :return train_performance: DataFrame contendo as métricas calculadas usando validação cruzada [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        train_performance = trainer.compute_train_performance(model_name, estimator, X_train, y_train)
        """

        # Computando métricas utilizando validação cruzada
        logger.debug(f'Computando métricas do modelo {model_name} utilizando validação cruzada com {cv} K-folds')
        try:
            t0 = time.time()
            mae = -(cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_absolute_error')).mean()
            mse_scores = cross_val_score(estimator, X, y, cv=5, scoring='neg_mean_squared_error')
            mse = (-mse_scores).mean()
            rmse = np.sqrt(-mse_scores).mean()
            r2 = cross_val_score(estimator, X, y, cv=5, scoring='r2').mean()

            # Criando DataFrame com o resultado obtido
            t1 = time.time()
            delta_time = t1 - t0
            train_performance = {}
            train_performance['model'] = model_name
            train_performance['approach'] = f'Treino {cv} K-folds'
            train_performance['mae'] = round(mae, 3)
            train_performance['mse'] = round(mse, 3)
            train_performance['rmse'] = round(rmse, 3)
            train_performance['r2'] = round(r2, 3)
            train_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de treino em {round(delta_time, 3)} segundos')

            return pd.DataFrame(train_performance, index=train_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')    

    def compute_test_performance(self, model_name, estimator, X, y):
        """
        Método responsável por aplicar retornar as principais métricas do modelo utilizando dados de teste.
        Na prática, esse método é chamado por um outro método em uma camada superior da classe para medição 
        de performance em treino e em teste

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo contida no atributo self.classifiers_info [type: string]
        :param estimator: estimator do modelo a ser avaliado [type: object]
        :param X: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y: array contendo a variável resposta dos dados de teste do modelo [type: np.array]

        Retorno
        -------
        :return test_performance: DataFrame contendo as métricas calculadas nos dados de teste [type: pd.DataFrame]

        Aplicação
        ---------
        # Instanciando e treinando modelo
        trainer = ClassificadorBinario()
        trainer.fit(model, X_train, y_train)
        test_performance = trainer.compute_test_performance(model_name, estimator, X_test, y_test)
        """

        # Predicting data using the trained model and computing probabilities
        logger.debug(f'Computando métricas do modelo {model_name} utilizando dados de teste')
        try:
            t0 = time.time()
            y_pred = estimator.predict(X)

            # Retrieving metrics using test data
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            # Creating a DataFrame with metrics
            t1 = time.time()
            delta_time = t1 - t0
            test_performance = {}
            test_performance['model'] = model_name
            test_performance['approach'] = 'Teste'
            test_performance['mae'] = round(mae, 3)
            test_performance['mse'] = round(mse, 3)
            test_performance['rmse'] = round(rmse, 3)
            test_performance['r2'] = round(r2, 3)
            test_performance['total_time'] = round(delta_time, 3)
            logger.info(f'Métricas computadas com sucesso nos dados de teste em {round(delta_time, 3)} segundos')

            return pd.DataFrame(test_performance, index=test_performance.keys()).reset_index(drop=True).loc[:0, :]

        except Exception as e:
            logger.error(f'Erro ao computar as métricas. Exception lançada: {e}')

    def evaluate_performance(self, X_train, y_train, X_test, y_test, cv=5, **kwargs):
        """
        Método responsável por executar e retornar métricas dos regressores em treino (média do resultado
        da validação cruzada com cv K-fols) e teste

        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']

        Retorno
        -------
        :return df_performances: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

        Aplicação
        -----------
        # Treinando modelo e avaliando performance em treino e teste
        trainer = ClassificadorBinario()
        trainer.fit(estimator, X_train, X_test)

        # Definindo dicionário de controle do resultado
        df_performance = trainer.evaluate_performance(X_train, y_train, X_test, y_test, save=True, output_path=caminho)
        """

        # DataFrame vazio para armazenamento das métrics
        df_performances = pd.DataFrame({})

        # Iterando sobre todos os classificadores da classe
        for model_name, model_info in self.regressors_info.items():

            # Validando se o modelo já foi treinado (dicionário model_info já terá a chave 'train_performance')
            if 'train_performance' in model_info.keys():
                df_performances = df_performances.append(model_info['train_performance'])
                df_performances = df_performances.append(model_info['test_performance'])
                continue

            # Retornando modelo a ser avaliado
            try:
                estimator = model_info['estimator']
            except KeyError as e:
                logger.error(f'Erro ao retornar a chave "estimator" do dicionário model_info. Modelo {model_name} não treinado')
                continue

            # Computando performance em treino e em teste
            train_performance = self.compute_train_performance(model_name, estimator, X_train, y_train, cv=cv)
            test_performance = self.compute_test_performance(model_name, estimator, X_test, y_test)

            # Adicionando os resultados ao atributo classifiers_info
            self.regressors_info[model_name]['train_performance'] = train_performance
            self.regressors_info[model_name]['test_performance'] = test_performance

            # Construindo DataFrame com as métricas retornadas
            model_performance = train_performance.append(test_performance)
            df_performances = df_performances.append(model_performance)
            df_performances['anomesdia_datetime'] = datetime.now()

            # Salvando alguns atributos no dicionário classifiers_info para acessos futuros
            model_data = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            model_info['model_data'] = model_data

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics.csv'
            self.save_data(df_performances, output_path=output_path, filename=output_filename)

        return df_performances
    
    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Método responsável por retornar a importância das features de um modelo treinado
        
        Parâmetros
        ----------
        :param features: lista contendo as features de um modelo [type: list]
        :param top_n: parâmetro para filtragem das top n features [type: int, default=-1]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']

        Retorno
        -------
        :return: all_feat_imp: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]
        """

        # Inicializando DataFrame vazio para armazenamento das feature importance
        feat_imp = pd.DataFrame({})
        all_feat_imp = pd.DataFrame({})

        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.regressors_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')

            try:
                importances = model_info['estimator'].feature_importances_
            except KeyError as ke:
                logger.error(f'Modelo {model_name} não treinado, sendo impossível extrair o método feature_importances_')
                continue
            except AttributeError as ae:
                logger.error(f'Modelo {model_name} não possui o método feature_importances_')
                continue

            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp['model'] = model_name
            feat_imp['anomesdia_datetime'] = datetime.now()
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)
            feat_imp = feat_imp.loc[:, ['model', 'feature', 'importance', 'anomesdia_datetime']]

            # Salvando essa informação no dicionário classifiers_info
            self.regressors_info[model_name]['feature_importances'] = feat_imp
            all_feat_imp = all_feat_imp.append(feat_imp)
            logger.info(f'Extração da importância das features concluída com sucesso para o modelo {model_name}')

        # Validando salvamento dos resultados
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'top_features.csv'
            self.save_data(all_feat_imp, output_path=output_path, filename=output_filename)
        
        return all_feat_imp
    
    def training_flow(self, set_regressors, X_train, y_train, X_test, y_test, features, **kwargs):
        """
        Método responsável por consolidar um fluxo completo de treinamento dos regressores, bem como
        o levantamento de métricas e execução de métodos adicionais para escolha do melhor modelo

        Parâmetros
        ----------
        :param set_regressors: dicionário contendo informações dos modelos a serem treinados [type: dict]
            set_regressors = {
                'model_name': {
                    'model': __estimator__,
                    'params': __estimator_params__
                }
            }
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param features: lista contendo as features de um modelo [type: list]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'output/')]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: 5]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg models_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/models']
            :arg metrics_output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/metrics']
            :arg metrics_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics.csv']
            :arg featimp_output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='top_features.csv']
            :arg top_n_featimp: top features a serem analisadas na importância das features [type: int, default=-1]

        Retorno
        -------
        None

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.training_flow(set_classifiers, X_train, y_train, X_test, y_test, features)
        """

        # Definindo variáveis padrão para retorno dos resultados
        """if not os.path.isdir(output_path):
            os.makedirs(output_path)"""

        # Extraindo parâmetros kwargs
        approach = kwargs['approach'] if 'approach' in kwargs else ''
        random_search = kwargs['random_search'] if 'random_search' in kwargs else False
        scoring = kwargs['scoring'] if 'scoring' in kwargs else 'neg_mean_squared_error'
        cv = kwargs['cv'] if 'cv' in kwargs else 5
        verbose = kwargs['verbose'] if 'verbose' in kwargs else -1
        n_jobs = kwargs['n_jobs'] if 'n_jobs' in kwargs else -1
        save = bool(kwargs['save']) if 'save' in kwargs else True
        models_output_path = kwargs['models_output_path'] if 'models_output_path' in kwargs else os.path.join(os.getcwd(), 'output/models')
        metrics_output_path = kwargs['metrics_output_path'] if 'metrics_output_path' in kwargs else os.path.join(os.getcwd(), 'output/metrics')
        metrics_output_filename = kwargs['metrics_output_filename'] if 'metrics_output_filename' in kwargs else 'metrics.csv'
        featimp_output_filename = kwargs['featimp_output_filename'] if 'featimp_output_filename' in kwargs else 'top_features.csv'
        top_n_featimp = kwargs['top_n_featimp'] if 'top_n_featimp' in kwargs else -1

        # Treinando classificadores
        self.fit(set_regressors, X_train, y_train, approach=approach, random_search=random_search, scoring=scoring,
                 cv=cv, verbose=verbose, n_jobs=n_jobs, output_path=models_output_path)

        # Avaliando modelos
        self.evaluate_performance(X_train, y_train, X_test, y_test, save=save, output_path=metrics_output_path, 
                                  output_filename=metrics_output_filename)

        # Analisando Features mais importantes
        self.feature_importance(features, top_n=top_n_featimp, save=save, output_path=metrics_output_path, 
                                output_filename=featimp_output_filename)
        
    def plot_feature_importance(self, features, top_n=20, palette='viridis', **kwargs):
        """
        Método responsável por realizar uma plotagem gráfica das variáveis mais importantes pro modelo

        Parâmetros
        ----------
        :param features: lista de features do conjunto de dados [type: list]
        :param top_n: quantidade de features a ser considerada na plotagem [type: int, default=20]
        :param palette: paleta de cores utilizazda na plotagem [type: string, default='viridis']
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='feature_importances.png']

        Retorno
        -------
        Este método não retorna nada além da imagem devidamente salva no diretório destino
        """

        # Definindo parâmetros de plotagem
        logger.debug('Inicializando plotagem das features mais importantes para os modelos')
        feat_imp = pd.DataFrame({})
        i = 0
        ax_del = 0
        nrows = len(self.regressors_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))
        sns.set(style='white', palette='muted', color_codes=True)
        
        # Iterando sobre os modelos presentes na classe
        for model_name, model_info in self.regressors_info.items():
            # Validando possibilidade de extrair a importância das features do modelo
            logger.debug(f'Extraindo importância das features para o modelo {model_name}')
            try:
                importances = model_info['estimator'].feature_importances_
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
                ax_del += 1
                continue
            
            # Preparando o dataset para armazenamento das informações
            feat_imp['feature'] = features
            feat_imp['importance'] = importances
            feat_imp.sort_values(by='importance', ascending=False, inplace=True)

            logger.debug(f'Plotando gráfico de importância das features para o modelo {model_name}')
            try:
                # Plotando feature importance
                sns.barplot(x='importance', y='feature', data=feat_imp.iloc[:top_n, :], ax=axs[i], 
                            palette=palette)

                # Customizando gráfico
                axs[i].set_title(f'Features Mais Importantes: {model_name}', size=14)
                format_spines(axs[i], right_border=False)
                i += 1
  
                logger.info(f'Gráfico de importância das features plotado com sucesso para o modelo {model_name}')
            except Exception as e:
                logger.error(f'Erro ao gerar gráfico de importância das features para o modelo {model_name}. Exception lançada: {e}')
                continue

        # Deletando eixos sobressalentes (se aplicável)
        if ax_del > 0:
            logger.debug('Deletando eixos referentes a análises não realizadas')
            try:
                for i in range(-1, -(ax_del+1), -1):
                    fig.delaxes(axs[i])
            except Exception as e:
                logger.error(f'Erro ao deletar eixo. Exception lançada: {e}')
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'feature_importances.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)   
    
    def plot_metrics(self, figsize=(16, 10), palette='rainbow', cv=5, **kwargs):
        """
        Método responsável por plotar os resultados das métricas dos regressores selecionados

        Parâmetros
        ----------
        :param figsize: dimensões da figura gerada para a plotagem [type: tuple, default=(16, 10)]
        :param palette: paleta de cores do matplotlib [type: string, default='rainbow']
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='metrics_comparison.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além da plotagem devidamente salva no diretório destino
        """

        logger.debug(f'Iniciando plotagem gráfica das métricas dos classificadores')
        metrics = pd.DataFrame()
        for model_name, model_info in self.regressors_info.items():

            logger.debug(f'Retornando métricas via validação cruzada para o modelo {model_name}')
            try:
                # Retornando variáveis do classificador
                metrics_tmp = pd.DataFrame()
                estimator = model_info['estimator']
                X = model_info['model_data']['X_train']
                y = model_info['model_data']['y_train']

                # Métricas não computadas
                mae = -(cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_absolute_error'))
                mse = -(cross_val_score(estimator, X, y, cv=cv, scoring='neg_mean_squared_error'))
                rmse = np.sqrt(mse)
                r2 = cross_val_score(estimator, X, y, cv=cv, scoring='r2')

                # Adicionando ao DataFrame recém criado
                metrics_tmp['mae'] = mae
                metrics_tmp['mse'] = mse
                metrics_tmp['rmse'] = rmse
                metrics_tmp['r2'] = r2
                metrics_tmp['model'] = model_name

                # Empilhando métricas
                metrics = metrics.append(metrics_tmp)
            except Exception as e:
                logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')
                continue

        logger.debug(f'Modificando DataFrame de métricas para plotagem gráfica')
        try:
            # Pivotando métricas (boxplot)
            index_cols = ['model']
            metrics_cols = ['mae', 'mse', 'rmse', 'r2']
            df_metrics = pd.melt(metrics, id_vars=index_cols, value_vars=metrics_cols)

            # Agrupando métricas (barras)
            metrics_group = df_metrics.groupby(by=['model', 'variable'], as_index=False).mean()
        except Exception as e:
            logger.error(f'Erro ao pivotar DataFrame. Exception lançada: {e}')
            return

        logger.debug(f'Plotando análise gráfica das métricas para os modelos treinados')
        x_rot = kwargs['x_rot'] if 'x_rot' in kwargs else 90
        bar_label = bool(kwargs['bar_label']) if 'bar_label' in kwargs else False
        try:
            # Criando figura de plotagem
            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=figsize)
            
            i = 0
            for metric in metrics_cols:                
                # Definindo eixos
                ax0 = axs[0, i]
                ax1 = axs[1, i]
                
                # Plotando gráficos
                sns.boxplot(x='variable', y='value', data=df_metrics.query('variable == @metric'), 
                            hue='model', ax=ax0, palette=palette)
                sns.barplot(x='model', y='value', data=metrics_group.query('variable == @metric'), 
                            ax=ax1, palette=palette, order=list(df_metrics['model'].unique()))
            
                # Personalizando plotagem
                ax0.set_title(f'Boxplot {cv}K-folds: {metric}')
                ax1.set_title(f'Média {cv} K-folds: {metric}')
                
                format_spines(ax0, right_border=False)
                format_spines(ax1, right_border=False)
                
                if bar_label:
                    AnnotateBars(n_dec=3, color='black', font_size=12).vertical(ax1)
                
                if i < 3:
                    ax0.get_legend().set_visible(False)
                    
                for tick in ax1.get_xticklabels():
                    tick.set_rotation(x_rot)
            
                i += 1
        except Exception as e:
            logger.error(f'Erro ao plotar gráfico das métricas. Exception lançada: {e}')
            return

        # Alinhando figura
        plt.tight_layout()

        # Salvando figura
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'metrics_comparison.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename)      
            
    def plot_learning_curve(self, ylim=None, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10), **kwargs):
        """
        Método responsável por calcular a curva de aprendizado para um modelo treinado
        
        Parâmetros
        ----------
        :param model_name: chave de referência para análise de um modelo já treinado[type: string]
        :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 6)]
        :param ylim: climite do eixo vertical [type: int, default=None]
        :param cv: k-folds utilizados na validação cruzada para levantamento de informações [type: int, default=5]
        :param n_jobs: número de processadores utilizado no levantamento das informações [type: int, default=1]
        :param train_sizes: array de passos utilizados na curva [type: np.array, default=np.linspace(.1, 1.0, 10)]
        :param **kwargs: argumentos adicionais do método
            :arg save: flag booleano para indicar o salvamento dos arquivos em disco [type: bool, default=True]
            :arg output_path: diretório para salvamento dos arquivos [type: string, default=cwd() + 'output/imgs']
            :arg output_filename: referência do nome do arquivo csv a ser salvo [type: string, default='confusion_matrix.png']

        Retorno
        -------
        Este método não retorna nenhum parâmetro além do salvamento do gráfico de distribuição especificado

        Aplicação
        -----------
        trainer.plot_learning_curve(model_name='LightGBM')
        """

        logger.debug(f'Inicializando plotagem da curvas de aprendizado dos modelos')
        i = 0
        nrows = len(self.regressors_info.keys())
        fig, axs = plt.subplots(nrows=nrows, figsize=(16, nrows * 6))

        # Iterando sobre os classificadores presentes na classe
        for model_name, model_info in self.regressors_info.items():
            ax = axs[i]
            logger.debug(f'Retornando parâmetros pro modelo {model_name} e aplicando método learning_curve')
            try:
                model = model_info['estimator']
                X_train = model_info['model_data']['X_train']
                y_train = model_info['model_data']['y_train']

                # Chamando função learning_curve para retornar os scores de treino e validação
                train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

                # Computando médias e desvio padrão (treino e validação)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                val_scores_mean = np.mean(val_scores, axis=1)
                val_scores_std = np.std(val_scores, axis=1)
            except Exception as e:
                logger.error(f'Erro ao retornar parâmetros e scores pro modelo {model_name}. Exception lançada: {e}')
                continue

            logger.debug(f'Plotando curvas de aprendizado de treino e validação para o modelo {model_name}')
            try:
                # Resultados utilizando dados de treino
                ax.plot(train_sizes, train_scores_mean, 'o-', color='navy', label='Training Score')
                ax.fill_between(train_sizes, (train_scores_mean - train_scores_std), (train_scores_mean + train_scores_std),
                                alpha=0.1, color='blue')

                # Resultados utilizando dados de validação (cross validation)
                ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross Val Score')
                ax.fill_between(train_sizes, (val_scores_mean - val_scores_std), (val_scores_mean + val_scores_std),
                                alpha=0.1, color='crimson')

                # Customizando plotagem
                ax.set_title(f'Model {model_name} - Learning Curve', size=14)
                ax.set_xlabel('Training size (m)')
                ax.set_ylabel('Score')
                ax.grid(True)
                ax.legend(loc='best')
            except Exception as e:
                logger.error(f'Erro ao plotar curva de aprendizado para o modelo {model_name}. Exception lançada: {e}')
                continue
            i += 1
        
        # Alinhando figura
        plt.tight_layout()

        # Salvando imagem
        if 'save' in kwargs and bool(kwargs['save']):
            output_path = kwargs['output_path'] if 'output_path' in kwargs else os.path.join(os.getcwd(), 'output/imgs')
            output_filename = kwargs['output_filename'] if 'output_filename' in kwargs else 'learning_curve.png'
            self.save_fig(fig, output_path=output_path, img_name=output_filename) 

    def _get_estimator(self, model_name):
        """
        Método responsável por retornar o estimator de um modelo selecionado

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário regressors_info da classe [type: string]

        Retorno
        -------
        :return model: estimator do modelo já treinado
        """

        logger.debug(f'Retornando estimator do modelo {model_name} já treinado')
        try:
            model_info = self.regressors_info[model_name]
            return model_info['estimator']
        except Exception as e:
            logger.error(f'Classificador {model_name} não existente ou não treinado. Opções possíveis: {list(self.classifiers_info.keys())}')
            return

    def _get_metrics(self, model_name):
        """
        Método responsável por retornar as métricas obtidas no treinamento

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return model_performance: DataFrame contendo as métricas dos modelos treinados [type: pd.DataFrame]
        """

        logger.debug(f'Retornando as métricas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo e métricas já salvas
            model_info = self.regressors_info[model_name]
            train_performance = model_info['train_performance']
            test_performance = model_info['test_performance']
            model_performance = train_performance.append(test_performance)
            model_performance.reset_index(drop=True, inplace=True)

            return model_performance
        except Exception as e:
            logger.error(f'Erro ao retornar as métricas para o modelo {model_name}. Exception lançada: {e}')

    def _get_model_info(self, model_name):
        """
        Método responsável por coletar as informações registradas de um determinado modelo da classe

        Parâmetros
        ----------
        :param model_name: chave identificadora do modelo no dicionário regressors_info da classe [type: string]

        Retorno
        -------
        :return model_info: dicionário com informações registradas do modelo [type: dict]
            model_info = {
                'estimator': model,
                'train_performance': pd.DataFrame,
                'test_performance': pd.DataFrame,
                'model_data': {
                    'X_train': np.array,
                    'y_train': np.array,
                    'X_test': np.array,
                    'y_test': np.array,
                'feature_importances': pd.DataFrame
                }
            }
        """

        logger.debug(f'Retornando informações registradas do modelo {model_name}')
        try:
            # Retornando dicionário do modelo
            return self.regressors_info[model_name]
        except Exception as e:
            logger.error(f'Erro ao retornar informações do modelo {model_name}. Exception lançada: {e}')

    def _get_regressors_info(self):
        """
        Método responsável por retornar o dicionário regressors_info contendo todas as informações de todos os modelos

        Parâmetros
        ----------
        None

        Retorno
        -------
        :return regressors_info: dicionário completo dos modelos presentes na classe
            regressors_info ={
                'model_name': model_info = {
                                'estimator': model,
                                'train_performance': pd.DataFrame,
                                'test_performance': pd.DataFrame,
                                'model_data': {
                                    'X_train': np.array,
                                    'y_train': np.array,
                                    'X_test': np.array,
                                    'y_test': np.array,
                                'feature_importances': pd.DataFrame
                                }
                            }
        """

        return self.regressors_info
