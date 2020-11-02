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
import time
from datetime import datetime
import os
import pandas as pd
from pycomp.log.log_config import log_config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


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
          2.1 Treinamento e Avaliação
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

    def save_data(self, df, kwargs):
        """
        Método responsável por salvar arquivos oriundos de execuções e validações de resultados
        presentes em outros métodos da classe. O método save_result() recebe um dicionário de argumentos
        necessários para validar o salvamento de um objeto em um determinado diretório

        Parâmetros
        ----------
        :param df: arquivo/objeto a ser salvo [type: pd.DataFrame]
        :param **kwargs: outros parâmetros da função:
            :arg save: flag booleano para indicar o salvamento do resultado em arquivo csv [type: bool]
            :arg overwrite: flag para indicar a sobrescrita ou append dos resultados [type: bool]
            :arg output_path: caminho onde o arquivo de resultados será salvo: [type: string]

        Retorno
        -------
        None

        Aplicação
        ---------
        df = file_generator_method()
        self.save_result(df, kwargs)
        """

        if 'save' in kwargs and bool(kwargs['save']):
            # Validando argumento path
            if 'output_path' not in kwargs:
                logger.warning('Argumento "output_path" não definido. Especifique o caminho para salvar o arquivo')
                return 
            else:
                output_path = kwargs['output_path']

            # Validando overwrite dos resultados
            logger.debug('Salvando arquivo')
            if 'overwrite' in kwargs and bool(kwargs['overwrite']):
                try:
                    df.to_csv(output_path, index=False)
                    logger.info(f'Arquivo salvo em: {output_path}')
                except Exception as e:
                    logger.error(f'Falha ao salvar o arquivo em {output_path}. Exception lançada: {e}')
            elif 'overwrite' in kwargs and not bool(kwargs['overwrite']):
                try:
                    df_log = pd.read_csv(output_path)
                    df_full = df_log.append(df)
                    df_full.to_csv(output_path, index=False)
                    logger.info(f'Arquivo salvo em: {output_path}')
                except FileNotFoundError:
                    logger.error(f'Erro ao salvar arquivo. Exception lançada: {e}')
                    df.to_csv(output_path, index=False)
            elif 'overwrite' not in kwargs:
                logger.warning('Parâmetro overwrite não contido no dicionário **kwargs. Especifique esse flag booleano para salvar o arquivo')

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
        :param y_train: array contendo variável do modelo [type: np.array]
        :param **kwargs: argumentos adicionais do método
            :arg approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
            :arg random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
            :arg scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
            :arg cv: K-folds utiliados na validação cruzada [type: int, default: 5]
            :arg verbose: nível de verbosity da busca aleatória [type: int, default: 5]
            :arg n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]

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
        if 'approach' in kwargs:
            approach = kwargs['approach']
        else:
            approach = ''

        # Iterando sobre os modelos presentes no dicionário de classificadores
        try:
            for model_name, model_info in set_classifiers.items():
                # Definindo chave do classificador para o dicionário classifiers_info
                clf_key = model_name + approach
                logger.debug(f'Treinando modelo {clf_key}')
                model = model_info['model']

                # Criando dicionário vazio para armazenar dados do modelo
                self.classifiers_info[clf_key] = {}

                # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
                try:
                    if 'random_search' in kwargs and bool(kwargs['random_search']):
                        params = model_info['params']
                        rnd_search = RandomizedSearchCV(model, params, scoring=kwargs['scoring'], cv=kwargs['cv'], 
                                                        verbose=kwargs['verbose'], random_state=42, n_jobs=kwargs['n_jobs'])
                        logger.debug('Aplicando RandomizedSearchCV')
                        rnd_search.fit(X, y)

                        # Salvando melhor modelo no atributo classifiers_info
                        self.classifiers_info[clf_key]['estimator'] = rnd_search.best_estimator_
                    else:
                        # Treinando modelo sem busca e salvando no atirbuto
                        self.classifiers_info[clf_key]['estimator'] = model.fit(X_train, y_train)
                    logger.info(f'Modelo {clf_key} treinado com sucesso')
                except Exception as e:
                    logger.error(f'Erro ao treinar o modelo {clf_key}. Exception lançada: {e}')
                    continue
        except AttributeError as e:
            logger.error('Dicionário de classificador(es) preparado de forma incorreta. Utilize {model_name: {estimator: estimator, params: params}}')
            logger.warning(f'Treinamento do(s) modelo(s) não realizado')

    def compute_train_performance(self, model_name, estimator, X, y, cv=5):
        """
        Método responsável por aplicar validação cruzada para retornar as principais métricas de avaliação
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
        Método responsável por centralizar a avaliação de métricas de um modelo a partir da chamada
        das funções que calculam performance nos dados de treino e de teste.
        
        Parâmetros
        ----------
        :param X_train: conjunto de features do modelo contido nos dados de treino [type: np.array]
        :param y_train: array contendo a variável resposta dos dados de treino do modelo [type: np.array]
        :param X_test: conjunto de features do modelo contido nos dados de teste [type: np.array]
        :param y_test: array contendo a variável resposta dos dados de teste do modelo [type: np.array]
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param **kwargs: outros parâmetros da função:
            :arg save: flag booleano para indicar o salvamento do resultado em arquivo csv [type: bool]
            :arg overwrite: flag para indicar a sobrescrita ou append dos resultados [type: bool]
            :arg output_path: caminho onde o arquivo de resultados será salvo: [type: string]
        
        Retorno
        ------
        :return df_performance: DataFrame contendo as métricas calculadas em treino e teste [type: pd.DataFrame]

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
            estimator = model_info['estimator']

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
        self.save_data(df_performances, kwargs=kwargs)

        return df_performances

    def feature_importance(self, features, top_n=-1, **kwargs):
        """
        Método responsável por retornar a importância das features de um modelo treinado
        
        Parâmetros
        ----------
        :param features: lista contendo as features de um modelo [type: list]
        :param top_n: parâmetro para filtragem das top n features [type: int, default: -1]
        :param **kwargs: outros parâmetros da função:
            :arg save: flag booleano para indicar o salvamento do resultado em arquivo csv [type: bool]
            :arg overwrite: flag para indicar a sobrescrita ou append dos resultados [type: bool]
            :arg path: caminho onde o arquivo de resultados será salvo: [type: string]

        Retorno
        -------
        :return: feature_importance: pandas DataFrame com a análise de feature importance dos modelos [type: pd.DataFrame]
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
            except:
                logger.warning(f'Modelo {model_name} não possui o método feature_importances_')
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
        self.save_data(all_feat_imp, kwargs=kwargs)

        return all_feat_imp

    def training_flow(self, set_classifiers, X_train, y_train, X_test, y_test, features,
                      save=True, overwrite=True, output_path=os.path.join(os.getcwd(), 'results/')):
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
        :param save: flag booleano para indicar o salvamento do resultado em arquivo csv [type: bool, default=True]
        :param overwrite: flag para indicar a sobrescrita ou append dos resultados [type: bool, default=True]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'results'/)]

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
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # Treinando classificadores
        self.fit(set_classifiers, X_train, y_train)

        # Avaliando modelos
        self.evaluate_performance(X_train, y_train, X_test, y_test, save=save, overwrite=overwrite, 
                                  output_path=os.path.join(output_path, 'metrics.csv'))

        # Analisando Features mais importantes
        self.feature_importance(features, save=save, overwrite=overwrite, 
                                output_path=os.path.join(output_path, 'top_features.csv'))