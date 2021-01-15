"""
---------------------------------------------------
----- TÓPICO: Machine Learning - Transformers -----
---------------------------------------------------
Módulo responsável por alocar classes customizadas
para a construção de Pipelines de Data Prep em
projetos de Machine Learning. As classes seguem os
requisitos necessários para serem inclusas em
Pipelines, importando os módulos BaseEstimator e
TransformerMixing na biblioteca sklearn, herdando
assim, por default, os métodos fit() e transform()

Sumário
-----------------------------------
1. Configuração Inicial
    1.1 Instanciando Objeto de Log
2. Transformadores
    2.1 Pipelines Iniciais
    2.2 Pipelines de Data Prep
    2.3 Pipelines de Consumo de Modelo
-----------------------------------
"""

# Importando bibliotecas
import logging
from pycomp.log.log_config import log_config
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


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
------------ 2. CUSTOM TRANSFORMERS ---------------
        2.1 Pipelines de Pré Processamento
---------------------------------------------------
"""

class FormataColunas(BaseEstimator, TransformerMixin):
    """
    Classe responsável por aplicar formatação customizada nas colunas de um DataFrame
    a partir das funções lower(), strip() e replace().
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    None

    Retorno
    ------
    :return: df: pandas DataFrame após a formatação das colunas [type: pd.DataFrame]

    Aplicação
    -----------
    cols_formatter = FormataColunas()
    df_custom = cols_formatter.fit_transform(df_old)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df


class FiltraColunas(BaseEstimator, TransformerMixin):
    """
    Classe responsável por filtrar as colunas de um DataFrame
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parêmetros
    ----------
    :param features: lista de colunas a serem filtradas do DataFrame [type: list]

    Retorno
    -------
    :return: df: pandas DataFrame após a filtragem dos atributos [type: pd.DataFrame]

    Aplicação
    ---------
    selector = FiltraColunas(features=model_features)
    df_filtered = selector.fit_transform(df)
    """

    def __init__(self, features):
        self.features = features

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df[self.features]


class DefineTarget(BaseEstimator, TransformerMixin):
    """
    Classe responsável por transformar uma coluna target em uma coluna numérica baseada
    em uma entrada da classe positiva.
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    :param target_col: referência para a coluna original de target do DataFrame [type: string]
    :param pos_class: entrada da classe positiva na coluna original de target [type: string]
    :param new_target_name: nome da nova coluna criada após o mapeamento [type: string, default: 'target]

    Retorno
    -------
    :return: df: pandas DataFrame após o mapeamento [pd.DataFrame]

    Aplicação
    ---------
    target_prep = DefineTarget(target_col='original_target', pos_class='DETRATOR', new_target_name='target')
    df = target_prep.fit_transform(df)
    """

    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name deve ser diferente da original target_col
        if self.target_col == self.new_target_name:
            self.flag_equal = 1
        else:
            self.flag_equal = 0

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Aplicando o mapeamento baseado na entrada da classe positiva
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Validando drop da coluna antiga de target
        if self.flag_equal:
            return df
        else:
            return df.drop(self.target_col, axis=1)


class EliminaDuplicatas(BaseEstimator, TransformerMixin):
    """
    Classe responsável por dropar duplicatas em um DataFrame.
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    None

    Retorno
    -------
    :return: df: pandas DataFrame sem duplicatas [type: pd.DataFrame]

    Aplicação
    ---------
    dup_dropper = EliminaDuplicatas()
    df_nodup = dup_dropper.fit_transform(df)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.drop_duplicates()


class SplitDados(BaseEstimator, TransformerMixin):
    """
    Classe responsável por aplicar a separação de uma base de dados em treino e teste
    a partir da aplicação da função train_test_split() do módulo sklearn.model_selection.
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    :param target: referência da variável target no dataset [type: string]
    :param test_size: percentual a ser direcionado para o conjunto de teste [type: float, default: .20]
    :param random_state: semente randômica [type: int, default: 42]

    Dicas Adicionais
    ----------------
    X_: atributo associado às features do dataset antes do split [1]
    y_: atributo associado ao target do dataset antes do split [1]
        [1] Os atributos X_ e y_ são inicializados no momento antes do split e podem ser retornados no decorrer do script

    Retorno
    ------
    :return: X_train: DataFrame referente aos dados de treino [type: pd.DataFrame]
             X_test: DataFrame referente aos dados de teste [type: pd.DataFrame]
             y_train: array com o target de treino [type: np.array]
             y_test: array com o target de teste [type: np.array]

    Aplicação
    ---------
    splitter = SplitDados(target='target')
    X_train, X_test, y_train, y_test = splitter.fit_transform(df)
    """

    def __init__(self, target, test_size=.20, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Retornando os atributos X_ e y_
        self.X_ = df.drop(self.target, axis=1)
        self.y_ = df[self.target].values

        return train_test_split(self.X_, self.y_, test_size=self.test_size, random_state=self.random_state)


class AgrupamentoCategoricoInicial(BaseEstimator, TransformerMixin):
    """
    Classe responsável por realizar um agrupamento categórico inicial em variáveis de muitas entradas
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    :param features: lista de colunas alvo da aplicação [type: list]
    :param n_cat: quantidade limite de entradas categóricas [type: int, default=5]
    :param other_tag: categoria referente às demais entradas [type: string, default='Other']

    Retorno
    ------
    :return: df: DataFrame após o agrupamento categórico [type: pandas.DataFrame]

    Aplicação
    ---------
    cat_agrup = AgrupamentoCategoricoInicial(features=lista, n_cat=3)
    df_prep = cat_agrup.fit_transform(df)
    """
    
    def __init__(self, features, n_cat=5, other_tag='Other'):
        self.features = features
        self.n_cat = n_cat
        self.other_tag = other_tag
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        # Iterando sobre o set de features
        for feature in self.features:
            other_list = list(df[feature].value_counts().index[self.n_cat:])
            df[feature] = df[feature].apply(lambda x: x if x not in other_list else self.other_tag)
            
        return df


class AgrupamentoCategoricoFinal(BaseEstimator, TransformerMixin):
    """
    Classe responsável por realizar um agrupamento categórico definitivo nas variáveis
    O método fit_transform() é herdado dos objetos BaseEstimator e TransformerMixin

    Parâmetros
    ----------
    :param cat_dict: dicionário com a relação de colunas e entradas [type: dict]
    :param other_tag: categoria referente às demais entradas [type: string, default='Other']

    Retorno
    ------
    :return: df: DataFrame após o agrupamento categórico [type: pandas.DataFrame]

    Aplicação
    ---------
    cat_agrup = AgrupamentoCategoricoFinal(cat_dict=cat_dict, other_tag=OTHER_TAG)
    df_prep = cat_agrup.fit_transform(df)
    """
    
    def __init__(self, cat_dict, other_tag='Other'):
        self.cat_dict = cat_dict
        self.other_tag = other_tag
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df, y=None):
        # Iterando sobre dicionário e transformando entradas das colunas
        for col, cats in self.cat_dict.items():
            df[col] = df[col].apply(lambda x: x if x in cats else self.other_tag)
            
        return df


"""
---------------------------------------------------
------------ 2. CUSTOM TRANSFORMERS ---------------
           2.2 Pipelines de DataPrep
---------------------------------------------------
"""


class DummiesEncoding(BaseEstimator, TransformerMixin):
    """
    Classe responsável por aplicar o processo de encoding em dados categóricos utilizando o método
    get_dummies() do pandas. Além disso, essa classe reserva as features após o processo de encoding,
    permitindo seu retorno e utilização ao longo do script.
    Esta classe deve ser utilizada em um pipeline de dados categóricos.

    Parâmetros
    ----------
    :param dummy_na: flag responsável por guitar o encoding dos valores nulos [type: bool, default: True]

    Retorno
    -------
    :return: X_dum: Dataframe (com dados categóricos) após o encoding [type: pd.DataFrame]

    Aplicação
    ---------
    encoder = DummiesEncoding(dummy_na=True)
    X_encoded = encoder.fit_transform(df[cat_features])
    """

    def __init__(self, dummy_na=True, cat_features_ori=None):
        self.dummy_na = dummy_na
        self.cat_features_ori = cat_features_ori

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Salvando features em um atributo da classe
        if self.cat_features_ori is None:
            try:
                self.cat_features_ori = list(X.columns)
            except AttributeError as ae:
                print(f'Impossível retornar colunas de um objeto do tipo numpy array. Utilize o atributo "cat_features_ori" da classe')
                print(f'Exception lançada: {e}')
                return
        else:
            X = pd.DataFrame(X, columns=self.cat_features_ori)

        # Aplicando encoding
        X_cat_dum = pd.get_dummies(X, dummy_na=self.dummy_na)

        # Unindo datasets e dropando colunas originals
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)

        # Salvando features após o encoding
        self.features_after_encoding = list(X_dum.columns)

        return X_dum


class PreencheDadosNulos(BaseEstimator, TransformerMixin):
    """
    Classe responsável por preencher dados nulos.
    Esta classe deve ser utilizada em um pipeline de dados numéricos.

    Parâmetros
    ----------
    :param cols_to_fill: colunas a serem preenchidas - set None para preencher todas as colunas [type: list, default: None]
    :param value_fill: valor a ser preenchido nas colunas [type: int, default: 0]

    Retorno
    -------
    :return: X: DataFrame com dados nulos preenchidos [type: pd.DataFrame]

    Aplicação
    ---------
    filler = PreencheDadosNulos(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X_filled = filler.fit_transform(X)
    """

    def __init__(self, cols_to_fill=None, value_fill=0):
        self.cols_to_fill = cols_to_fill
        self.value_fill = value_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Preenchendo dados nulos
        if self.cols_to_fill is not None:
            X[self.cols_to_fill] = X[self.cols_to_fill].fillna(value=self.value_fill)
            return X
        else:
            return X.fillna(value=self.value_fill)


class EliminaDadosNulos(BaseEstimator, TransformerMixin):
    """
    Classe responsável por droppar dados nulos a partir do método dropna()

    Parâmetros
    ----------
    :param cols_dropna: colunas cujos nulos serão dropados - set None para considerar todas [type: list, default: None]

    Retorno
    -------
    :return: X: DataFrame sem dados nulos [type: pd.DataFrame]

    Application
    -----------
    null_dropper = EliminaDadosNulos(cols_dropna=['colA', 'colB', 'colC'])
    X = null_dropper.fit_transform(X)
    """

    def __init__(self, cols_dropna=None):
        self.cols_dropna = cols_dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Dropando nulos
        if self.cols_dropna is not None:
            X[self.cols_dropna] = X[self.cols_dropna].dropna()
            return X
        else:
            return X.dropna()


class SeletorTopFeatures(BaseEstimator, TransformerMixin):
    """
    Classe responsável por selecionar as top k features mais importantes de um modelo treinado

    Parâmetros
    ----------
    :param feature_importance: array com a importâncias das features retornado de um modelo treinado [np.array]
    :param k: define as top k features a serem filtradas do array [type: int]

    Retorno
    -------
    :return: DataFrame filtrado pelas top k features mais importantes [pd.DataFrame]

    Aplicação
    ---------
    feature_selector = SeletorTopFeatures(feature_importance, k=10)
    X_selected = feature_selector.fit_transform(X)
    """

    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        indices = np.sort(np.argpartition(np.array(self.feature_importance), -self.k)[-self.k:])
        return X[:, indices]


"""
---------------------------------------------------
------------ 2. CUSTOM TRANSFORMERS ---------------
        2.3 Pipelines de Consumo de Modelo
---------------------------------------------------
"""

class ConsumoModelo(BaseEstimator, TransformerMixin):
    """
    Classe responsável por realizar o consumo de um modelo e gerar uma base final com as predições (classe e score)

    Parâmetros
    ----------
    :param model: estimator treinado do modelo a ser utilizado nas predições [type: estimator]
    :param features: features finais do modelo após o step de preparação [type: list]

    Retorno
    -------
    :return df_pred: DataFrame contendo as features e os resultados das preidções [type: pd.DataFrame]

    Aplicação
    ---------
    model = trainer._get_estimator(model_name='RandomForest')
    model_exec = ConsumoModelo(model=model, features=MODEL_FEATURES)
    df_pred = model_exec.fit_transform()
    """
    
    def __init__(self, model, features):
        self.model = model
        self.features = features
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        # Criando DataFrame com features do modelo
        df_final = pd.DataFrame(X, columns=self.features)
        
        # Realizando predições
        df_final['y_pred'] = self.model.predict(X)
        df_final['y_scores'] = self.model.predict_proba(X)[:, 1]
        
        return df_final        