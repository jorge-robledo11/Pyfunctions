import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Normalizer


class CustomColumnsNormalizer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for normalizing selected columns of a DataFrame.

    Parameters:
    -----------
    columns_to_normalize: list
        A list of column names to be normalized.

    norm: str, default='l2'
        The norm to use for normalization. Options: 'l1', 'l2', 'max'.

    Attributes:
    -----------
    columns_to_normalize: list
        The list of column names to be normalized.

    norm: str
        The norm used for normalization.

    normalizer: Normalizer
        The scikit-learn Normalizer instance used for normalization.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. It initializes the Normalizer instance with the specified norm.

    transform(X):
        Normalize the selected columns of the input DataFrame X.

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
    >>> columns_to_normalize = ['A', 'B']
    >>> transformer = CustomColumnNormalizer(columns_to_normalize=columns_to_normalize, norm='l2')
    >>> df_transformed = transformer.transform(df)
    >>> df_transformed
              A         B  C
    0  0.267261  0.801784  5
    1  0.447214  0.894427  6
    """
    
    def __init__(self, variables, norm='l2'):
        self.variables = variables
        self.norm = norm
        self.normalizer = Normalizer(norm=self.norm)

    def fit(self, X:pd.DataFrame, y=None):
        self.normalizer.fit(X[self.variables])
        return self

    def transform(self, X):
        X_normalized = X.copy()
        X_normalized[self.variables] = self.normalizer.transform(X[self.variables])
        return X_normalized
