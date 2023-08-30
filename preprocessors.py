import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
    
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


class DateColumnsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for transforming date columns in a DataFrame.

    Parameters:
    -----------
    date_columns: list of str, optional (default=None)
        A list of column names to be transformed. If None, all columns containing
        'fecha', 'date', 'tiempo', or 'time' in their names will be transformed.

    format: str, optional (default='%Y-%m-%d')
        The date format to which the columns will be converted.

    Attributes:
    -----------
    date_columns: list of str
        The list of column names to be transformed.

    format: str
        The date format used for conversion.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Transform date columns of the input DataFrame X to the specified date format.

    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'fecha_inicio': ['2023-01-01', '2023-02-01'],
    ...                    'date_ending': ['2023-03-01', '2023-04-01'],
    ...                    'other_column': [1, 2]})
    >>> transformer = DateColumnsTransformer()
    >>> df_transformed = transformer.transform(df)
    >>> df_transformed
      fecha_inicio date_ending  other_column
    0   2023-01-01  2023-03-01             1
    1   2023-02-01  2023-04-01             2
    """

    def __init__(self, date_columns=None, format='%Y-%m-%d'):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        date_columns : list of str, optional (default=None)
            A list of column names to be transformed. If None, all columns containing
            'fecha', 'date', 'tiempo', or 'time' in their names will be transformed.

        format : str, optional (default='%Y-%m-%d')
            The date format to which the columns will be converted.
        """
        
        self.date_columns = date_columns
        self.format = format

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
        self: DateColumnsTransformer
            The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame):
        
        """
        Transform date columns of the input DataFrame X to the specified date format.

        Parameters:
        ----------
        X: pandas.DataFrame
            The input DataFrame with date columns to be transformed.

        Returns:
        -------
        X_transformed : pandas.DataFrame
            The DataFrame with date columns transformed to the specified format.
        """
        X_transformed = X.copy()

        if self.date_columns is None:
            # If date_columns is not specified, select columns with date-related names
            date_columns = X.select_dtypes(include=['datetime64']).columns
        else:
            date_columns = self.date_columns

        for col in date_columns:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_datetime(X_transformed[col], format=self.format)

        return X_transformed
