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
from sklearn.model_selection import RandomizedSearchCV


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

    def fit(self, set_classifiers, X, y, approach='', random_search=False, scoring='accuracy', 
            cv=5, verbose=5, n_jobs=-1):
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
        :param X: features do modelo a ser treinado [type: np.array]
        :param y: array contendo variável do modelo [type: np.array]
        :param approach: indicativo de sufixo para armazenamento no atributo classifiers_info [type: string, default: '']
        :param random_search: flag para aplicação do RandomizedSearchCV [type: bool, default: False]
        :param scoring: métrica a ser otimizada pelo RandomizedSearchCV [type: string, default: 'accuracy']
        :param cv: K-folds utiliados na validação cruzada [type: int, default: 5]
        :param verbose: nível de verbosity da busca aleatória [type: int, default: 5]
        :param n_jobs: quantidade de jobs aplicados durante a busca dos hiperparâmetros [type: int, default: -1]

        Retorno
        -------
        Este método não retorna nada além do preenchimento de informações do treinamento no atributo self.classifiers_info

        Aplicação
        ---------
        # Instanciando objeto
        trainer = ClassificadorBinario()
        trainer.fit(set_classifiers, X_train_prep, y_train)
        """

        # Iterando sobre os modelos presentes no dicionário de classificadores
        for model_name, model_info in set_classifiers.items():
            # Definindo chave do classificador para o dicionário classifiers_info
            clf_key = model_name + approach
            logger.debug(f'Treinando modelo {clf_key}')
            model = model_info['model']

            # Criando dicionário vazio para armazenar dados do modelo
            self.classifiers_info[clf_key] = {}

            # Validando aplicação da busca aleatória pelos melhores hiperparâmetros
            if random_search:
                params = model_info['params']
                rnd_search = RandomizedSearchCV(model, params, scoring=scoring, cv=cv, verbose=verbose,
                                                random_state=42, n_jobs=n_jobs)
                rnd_search.fit(X, y)

                # Salvando melhor modelo no atributo classifiers_info
                self.classifiers_info[clf_key]['estimator'] = rnd_search.best_estimator_
            else:
                # Treinando modelo sem busca e salvando no atirbuto
                self.classifiers_info[clf_key]['estimator'] = model.fit(X, y)

