import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnsRenameTransformer(BaseEstimator, TransformerMixin):
    
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
    >>> transformer = ColumnsRenameTransformer(transformation=custom_transform)
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
        ----------
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

    def transform(self, X:pd.DataFrame):
        
        """
        Rename columns of the input DataFrame X using the provided transformation function.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame with columns to be renamed.

        Returns:
        --------
        X_transformed: pandas.DataFrame
            The DataFrame with column names transformed according to the provided function.
        """
        
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

    def transform(self, X:pd.DataFrame):
        
        """
        Transform date columns of the input DataFrame X to the specified date format.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame with date columns to be transformed.

        Returns:
        --------
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

class CategoricalColumnsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for preprocessing categorical columns in a DataFrame.

    Parameters:
    -----------
    strip_and_lower: bool, optional (default=True)
        If True, strip leading and trailing whitespaces and convert to lowercase for string columns.

    Attributes:
    -----------
    strip_and_lower: bool
        Whether to apply strip and lowercase transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training,
        it returns itself unchanged.

    transform(X):
        Preprocess categorical columns in the input DataFrame X.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [' Foo', 'Bar  ', 'Baz'], 'B': ['True', ' False ', True]})
    >>> transformer = CategoricalColumnsTransformer()
    >>> data_transformed = transformer.transform(data)
    >>> data_transformed
         A      B
    0   foo   true
    1   bar  false
    2   baz   true
    """

    def __init__(self, strip_and_lower=True):
        
        """
        Initialize the transformer.

        Parameters:
        -----------
        strip_and_lower : bool, optional (default=True)
            If True, strip leading and trailing whitespaces and convert to lowercase for string columns.
        """
        
        self.strip_and_lower = strip_and_lower

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
        self: CategoricalColumnsTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Preprocess categorical columns in the input DataFrame X.

        Parameters:
        -----------
        X: pandas.DataFrame
            The input DataFrame with categorical columns to be preprocessed.

        Returns:
        --------
        X_transformed: pandas.DataFrame
            The DataFrame with categorical columns preprocessed according to the specified options.
        """
        
        X = X.copy()

        if self.strip_and_lower:
            categoricals = X.select_dtypes(include=['object', 'bool']).columns
            X[categoricals] = X[categoricals].applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

        return X

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    
    """
    A transformer for dropping columns in a DataFrame based on a missing value threshold.

    Parameters:
    -----------
    threshold: float
        The threshold for column removal. Columns with missing values exceeding this threshold will be dropped.

    Attributes:
    -----------
    features_to_drop_: list
        A list to store the names of columns that were dropped during transformation.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. Since this transformer doesn't require any training, it returns itself unchanged.

    transform(X):
        Remove columns from the input DataFrame X based on the missing value threshold.

    Examples:
    ---------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [None, 4], 'C': [5, None]})
    >>> transformer = DropColumnsTransformer(threshold=1/3)
    >>> transformed_data = transformer.transform(data)
    >>> transformer.dropped_columns
    ['B', 'C']
    """

    def __init__(self, threshold=1/3):
        
        """
        Initialize the transformer with a missing value threshold.

        Parameters:
        -----------
        threshold: float, default 1/3
            The threshold for column removal. Columns with missing values exceeding this threshold will be dropped.
        """
        
        self.threshold = threshold
        self.features_to_drop_ = list()

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
        self: DropColumnsTransformer
            The fitted transformer instance.
        """
        
        return self

    def transform(self, X:pd.DataFrame):
        
        """
        Remove columns from the input DataFrame X based on the missing value threshold.

        Parameters:
        ----------
        X: pandas.DataFrame
            The input DataFrame with columns to be potentially removed.

        Returns:
        -------
        X_transformed: pandas.DataFrame
            The DataFrame with columns removed based on the provided threshold.
        """
        
        X_copy = X.copy()
        columns_to_drop = X.columns[X.isnull().mean() > self.threshold].tolist()
        self.features_to_drop_ = columns_to_drop  # Store the names of dropped columns
        X_transformed = X_copy.drop(columns=columns_to_drop)
        
        return X_transformed
