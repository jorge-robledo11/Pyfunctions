import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnNameTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for renaming columns of a DataFrame using a custom transformation function.

    Parameters:
    -----------
    transformation: function
        A function that takes a column name (string) as input and returns the new column name.

    Attributes:
    -----------
    transformation: function
        The transformation function used for renaming column names.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Rename columns of the input DataFrame X using the provided transformation function.

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> def custom_transform(col_name):
    ...     return col_name.lower()
    >>> transformer = ColumnNameTransformer(transformation=custom_transform)
    >>> df_transformed = transformer.transform(df)
    >>> df_transformed
       a  b
    0  1  3
    1  2  4
    """

    def __init__(self, transformation):
        
        """
        Initialize the transformer with a custom column name transformation function.

        Parameters:
        -----------
        transformation : function
            A function that takes a column name (string) as input and returns the new column name.
        """
        self.transformation = transformation

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self : ColumnNameTransformer
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame):
        
        """
        Rename columns of the input DataFrame X using the provided transformation function.

        Parameters:
        ----------
        X: pandas.DataFrame
            The input DataFrame with columns to be renamed.

        Returns:
        -------
        X_transformed : pandas.DataFrame
            The DataFrame with column names transformed according to the provided function.
        """
        
        X = X.copy()
        X_transformed = X.rename(columns=self.transformation)
        return X_transformed

    
class DropDuplicatedRowsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer to remove duplicate rows from a DataFrame.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Remove duplicate rows from the input DataFrame X.

    Examples:
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris(as_frame=True)
    >>> df = iris.data
    >>> transformer = DropDuplicatedRowsTransformer()
    >>> df_no_duplicates = transformer.transform(df)
    """

    def __init__(self):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        None
        """
        pass

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self : DropDuplicatedTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Remove duplicate rows from the input DataFrame X.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame from which duplicate rows will be removed.

        Returns:
        --------
        X_no_duplicates: pandas.DataFrame
            The DataFrame with duplicate rows removed.
        """
        
        X = X.copy()
        X_no_duplicates = X.drop_duplicates()
        return X_no_duplicates


class FillMissingValuesTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer to fill missing values in a DataFrame with np.nan.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    None

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Fill missing values in the input DataFrame X with NaN.

    Examples:
    --------
    >>> data = pd.DataFrame({'col1': ['A', 'B', '', 'C'], 'col2': [1, np.nan, 'None', 'N/A']})
    >>> transformer = FillMissingValuesTransformer()
    >>> data_no_missing = transformer.transform(data)
    """

    def __init__(self):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        None
        """
        
        pass

    def fit(self, X:pd.DataFrame, y=None):
        
        """
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame.

        y: None
            Ignored. This parameter is included for compatibility with scikit-learn's transformers.

        Returns:
        --------
        self: FillMissingValuesTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Fill missing values in the input DataFrame X with np.nan.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame in which missing values will be replaced with np.nan.

        Returns:
        --------
        X_no_missing: pandas.DataFrame
            The DataFrame with missing values replaced by np.nan.
        """
        
        X = X.copy()
        X_no_missing = X.fillna(np.nan)
        X_no_missing = X_no_missing.replace({'ERROR': np.nan,
                                             '': np.nan,
                                             'None': np.nan,
                                             'n/a': np.nan,
                                             'N/A': np.nan,
                                             'NULL': np.nan, 
                                             'NA': np.nan,
                                             'NAN': np.nan})
        return X_no_missing
