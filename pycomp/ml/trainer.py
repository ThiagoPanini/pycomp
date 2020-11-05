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
from pycomp.log.log_config import log_config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, \
                            f1_score, confusion_matrix, roc_curve
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


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
            if 'output_file' not in kwargs:
                logger.warning('Argumento "output_file" não definido. Especifique o caminho para salvar o arquivo')
                return 
            else:
                output_file = kwargs['output_file']

            # Validando overwrite dos resultados
            logger.debug('Salvando arquivo')
            if 'overwrite' in kwargs and bool(kwargs['overwrite']):
                try:
                    df.to_csv(output_file, index=False)
                    logger.info(f'Arquivo salvo em: {output_file}')
                except Exception as e:
                    logger.error(f'Falha ao salvar o arquivo em {output_file}. Exception lançada: {e}')
            elif 'overwrite' in kwargs and not bool(kwargs['overwrite']):
                try:
                    df_log = pd.read_csv(output_file)
                    df_full = df_log.append(df)
                    df_full.to_csv(output_file, index=False)
                    logger.info(f'Arquivo salvo em: {output_file}')
                except FileNotFoundError:
                    logger.error(f'Erro ao salvar arquivo. Exception lançada: {e}')
                    df.to_csv(output_file, index=False)
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
            :arg output_file: caminho onde o arquivo de resultados será salvo: [type: string]
        
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
                                  output_file=os.path.join(output_path, 'metrics.csv'))

        # Analisando Features mais importantes
        self.feature_importance(features, save=save, overwrite=overwrite, 
                                output_file=os.path.join(output_path, 'top_features.csv'))

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
    
    def plot_confusion_matrix(self, cmap=plt.cm.Blues, normalize=False, 
                              output_path=os.path.join(os.getcwd(), 'results/')):
        """
        Método responsável por plotar gráficos de matriz de confusão usando dados de treino e teste
        para todos os modelos presentes no dicionárion de classificadores self.classifiers_info

        Parâmetros
        ----------
        :param cmap: colormap para a matriz gerada [type: matplotlib.colormap]
        :param normalize: flag para normalizar as entradas da matriz [type: bool, default=False]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'results'/)]
        
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
                return

            # Realizando predições em treino (cross validation) e teste
            logger.debug(f'Realizando predições para os dados de treino e teste ({model_name})')
            try:
                train_pred = cross_val_predict(model_info['estimator'], X_train, y_train, cv=5)
                test_pred = model_info['estimator'].predict(X_test)
            except Exception as e:
                logger.error(f'Erro ao realizar predições para o modelo {model_name}. Exception lançada: {e}')
                return

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
                return

        # Salvando imagem
        plt.tight_layout()
        try:
            output_file = os.path.join(output_path, 'confusion_matrix.png')
            plt.savefig(output_file, dpi=300)
            logger.info(f'Imagem com as matrizes salva com sucesso em {output_file}')
        except Exception as e:
            logger.error(f'Erro ao salvar a imagem. Exception lançada: {e}')
            return

    def plot_roc_curve(self, figsize=(16, 6), output_path=os.path.join(os.getcwd(), 'results/')):
        """
        Método responsável por iterar sobre os classificadores presentes na classe e plotar a curva ROC
        para treino (primeiro eixo) e teste (segundo eixo)

        Parâmetros
        ----------
        :param figsize: dimensões da figura de plotagem [type: tuple, default=(16, 6)]
        :param output_path: caminho onde o arquivo de resultados será salvo: [type: string, default=os.path.join(os.path.getcwd(), 'results'/)]

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
                return

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
                return

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
                return

        # Salvando imagem
        plt.tight_layout()
        try:
            output_file = os.path.join(output_path, 'roc_curve.png')
            plt.savefig(output_file, dpi=300)
            logger.info(f'Imagem com a curva ROC salva com sucesso em {output_file}')
        except Exception as e:
            logger.error(f'Erro ao salvar a imagem. Exception lançada: {e}')
            return
    