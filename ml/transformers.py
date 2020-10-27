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
2. Validação e Manuseio de Arquivos
    2.1 Validação na Origem
    2.1 Cópia de Arquivos
3. Controle de Diretório
-----------------------------------
"""

# Importando bibliotecas
import logging
from logs import log_config


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
------------ 2. CUSTOM TRANSFORMERS --------------
        2.1 Pipelines de Pré Processamento
---------------------------------------------------
"""

class ColsFormatting(BaseEstimator, TransformerMixin):
    """
    This class applies lower(), strip() and replace() method on a pandas DataFrame object.
    It's not necessary to pass anything as args.

    Return
    ------
    :return: df: pandas DataFrame after cols formatting [type: pd.DataFrame]

    Application
    -----------
    cols_formatter = ColsFormatting()
    df_custom = cols_formatter.fit_transform(df_old)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        return df


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    This class filters a dataset based on a set of features passed as argument.

    Parameters
    ----------
    :param features: set of features to be selected on a DataFrame [type: list]

    Return
    ------
    :return: df: pandas DataFrame after filtering attributes [type: pd.DataFrame]

    Application
    -----------
    selector = FeatureSelection(features=model_features)
    df_filtered = selector.fit_transform(df)
    """

    def __init__(self, features):
        self.features = features

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df[self.features]


class TargetDefinition(BaseEstimator, TransformerMixin):
    """
    This class transform a categorical target column into a numerical one base on a positive_class

    Parameters
    ----------
    :param target_col: reference for the target column on the dataset [type: string]
    :param pos_class: entry reference for positive class in the new target [type: string]
    :param new_target_name: name of the new column created after the target mapping [type: string, default: 'target]

    Return
    ------
    :return: df: pandas DataFrame after target mapping [pd.DataFrame]

    Application
    -----------
    target_prep = TargetDefinition(target_col='class_target', pos_class='Some Category', new_target_name='target')
    df = target_prep.fit_transform(df)
    """

    def __init__(self, target_col, pos_class, new_target_name='target'):
        self.target_col = target_col
        self.pos_class = pos_class
        self.new_target_name = new_target_name

        # Sanity check: new_target_name may differ from target_col
        if self.target_col == self.new_target_name:
            print('[WARNING]')
            print(f'New target column named {self.new_target_name} must differ from raw one named {self.target_col}')

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Applying the new target rule based on positive class
        df[self.new_target_name] = df[self.target_col].apply(lambda x: 1 if x == self.pos_class else 0)

        # Dropping the old target column
        return df.drop(self.target_col, axis=1)


class DropDuplicates(BaseEstimator, TransformerMixin):
    """
    This class filters a dataset based on a set of features passed as argument.
    It's not necessary to pass anything as args.

    Return
    ------
    :return: df: pandas DataFrame dropping duplicates [type: pd.DataFrame]

    Application
    -----------
    dup_dropper = DropDuplicates()
    df_nodup = dup_dropper.fit_transform(df)
    """

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df.drop_duplicates()


class SplitData(BaseEstimator, TransformerMixin):
    """
    This class helps splitting data into training and testing and it can be used at the end of a pre_processing pipe.
    In practice, the class applies the train_test_split() function from sklearn.model_selection module.

    Parameters
    ----------
    :param target: reference of the target feature on the dataset [type: string]
    :param test_size: test_size param of train_test_split() function [type: float, default: .20]
    :param random_state: random_state param of train_test_split() function [type: int, default: 42]

    X_: attribute associated with the features dataset before splitting [1]
    y_: attribute associated with the target array before splitting [1]
        [1] The X_ and y_ attributes are initialized right before splitting and can be retrieved later in the script.

    Return
    ------
    :return: X_train: DataFrame for training data [type: pd.DataFrame]
             X_test: DataFrame for testing data [type: pd.DataFrame]
             y_train: array for training target data [type: np.array]
             y_test: array for testing target data [type: np.array]

    Application
    -----------
    splitter = SplitData(target='target')
    X_train, X_test, y_train, y_test = splitter.fit_transform(df)
    """

    def __init__(self, target, test_size=.20, random_state=42):
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        # Returning X and y attributes (those can be retrieved in the future)
        self.X_ = df.drop(self.target, axis=1)
        self.y_ = df[self.target].values

        return train_test_split(self.X_, self.y_, test_size=self.test_size, random_state=self.random_state)


"""
-----------------------------------
----- 2. CUSTOM TRANSFORMERS ------
    2.2 Preparation Pipelines
-----------------------------------
"""


class DummiesEncoding(BaseEstimator, TransformerMixin):
    """
    This class applies the encoding on categorical data using pandas get_dummies() method. It also retrieves the
    features after the encoding so it can be used further on the script

    Parameters
    ----------
    :param dummy_na: flag that guides the encoding of NaN values on categorical features [type: bool, default: True]

    Return
    ------
    :return: X_dum: Dataframe object (with categorical features) after encoding [type: pd.DataFrame]

    Application
    -----------
    encoder = DummiesEncoding(dummy_na=True)
    X_encoded = encoder.fit_transform(df[cat_features])
    """

    def __init__(self, dummy_na=True):
        self.dummy_na = dummy_na

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Saving features into class attribute
        self.cat_features_ori = list(X.columns)

        # Applying encoding with pandas get_dummies()
        X_cat_dum = pd.get_dummies(X, dummy_na=self.dummy_na)

        # Joining datasets and dropping original columns before encoding
        X_dum = X.join(X_cat_dum)
        X_dum = X_dum.drop(self.cat_features_ori, axis=1)

        # Retrieving features after encoding
        self.features_after_encoding = list(X_dum.columns)

        return X_dum


class FillNullData(BaseEstimator, TransformerMixin):
    """
    This class fills null data. It's possible to select just some attributes to be filled with different values

    Parameters
    ----------
    :param cols_to_fill: columns to be filled. Leave None if all the columns will be filled [type: list, default: None]
    :param value_fill: value to be filled on the columns [type: int, default: 0]

    Return
    ------
    :return: X: DataFrame object with NaN data filled [type: pd.DataFrame]

    Application
    -----------
    filler = FillNullData(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X_filled = filler.fit_transform(X)
    """

    def __init__(self, cols_to_fill=None, value_fill=0):
        self.cols_to_fill = cols_to_fill
        self.value_fill = value_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Filling null data according to passed args
        if self.cols_to_fill is not None:
            X[self.cols_to_fill] = X[self.cols_to_fill].fillna(value=self.value_fill)
            return X
        else:
            return X.fillna(value=self.value_fill)


class DropNullData(BaseEstimator, TransformerMixin):
    """
    This class drops null data. It's possible to select just some attributes to be filled with different values

    Parameters
    ----------
    :param cols_dropna: columns to be filled. Leave None if all the columns will be filled [type: list, default: None]

    Return
    ------
    :return: X: DataFrame object with NaN data filled [type: pd.DataFrame]

    Application
    -----------
    null_dropper = DropNulldata(cols_to_fill=['colA', 'colB', 'colC'], value_fill=-999)
    X = null_dropper.fit_transform(X)
    """

    def __init__(self, cols_dropna=None):
        self.cols_dropna = cols_dropna

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Filling null data according to passed args
        if self.cols_dropna is not None:
            X[self.cols_dropna] = X[self.cols_dropna].dropna()
            return X
        else:
            return X.dropna()


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    """
    This class selects the top k most important features from a trained model

    Parameters
    ----------
    :param feature_importance: array with feature importance given by a trained model [np.array]
    :param k: integer that defines the top features to be filtered from the array [type: int]

    Return
    ------
    :return: pandas DataFrame object filtered by the k important features [pd.DataFrame]

    Application
    -----------
    feature_selector = TopFeatureSelector(feature_importance, k=10)
    X_selected = feature_selector.fit_transform(X)
    """

    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, indices_of_top_k(self.feature_importance, self.k)]
