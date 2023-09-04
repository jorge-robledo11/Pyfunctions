# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Pyfunctions ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Librerías y/o depedencias
from IPython.display import display, Latex
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif
from sklearn.calibration import calibration_curve
from scipy import stats
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import empiricaldist
sns.set_theme(context='notebook', style=plt.style.use('dark_background'))

# X = data.loc[:, data.columns != «Target»] -> Dataframe
# y = data.loc[:, data.columns == «Target»].squeeze() -> Serie

# Función para reducir el peso en memoria de un DataFrame
def downcast_dtypes(data:pd.DataFrame) -> pd.DataFrame:

    """
    Function to downcast any type variable

    Args:
        data: DataFrame
    
    Return:
        DataFrame
    """

    start = data.memory_usage(deep=True).sum() / 1024 ** 2
    float_cols = [col for col in data if data[col].dtype == 'float64']
    int_cols = [col for col in data if data[col].dtype in ['int64', 'int32']]
    object_cols = [col for col in data if data[col].dtype in ['object', 'bool']]

    data[float_cols] = data[float_cols].astype(np.float64)
    data[int_cols] = data[int_cols].astype(np.int64)
    data[object_cols] = data[object_cols].astype('category')

    end = data.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (start - end) / start * 100
    print(f'Memory Saved: {saved:0.2f}%', '\n')

    return data.info()


# Capturar variables
# Función para capturar los tipos de variables
def capture_variables(data:pd.DataFrame) -> tuple:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: A tuple of lists
    
    The order to unpack variables:
    1. numericals
    2. continous
    3. categoricals
    4. discretes
    5. temporaries
    """

    numericals = list(data.select_dtypes(include = [np.int32, np.int64, np.float32, np.float64]).columns)
    categoricals = list(data.select_dtypes(include = ['category', 'bool']).columns)
    temporaries = list(data.select_dtypes(include = ['datetime', 'timedelta']).columns)
    discretes = [col for col in data[numericals] if len(data[numericals][col].unique()) < 10]
    continuous = [col for col in data[numericals] if col not in discretes]

    # Variables
    print('\t\tTipos de variables')
    print(f'Hay {len(continuous)} variables continuas')
    print(f'Hay {len(discretes)} variables discretas')
    print(f'Hay {len(temporaries)} variables temporales')
    print(f'Hay {len(categoricals)} variables categóricas')
    
    variables = tuple((continuous, categoricals, discretes, temporaries))

    # Retornamos una tupla de listas
    return variables


# Valores faltantes
# Función para observar variables con datos nulos y su porcentaje
def nan_values(data:pd.DataFrame, variables:list, variable_type:str) -> any:
    
    """
    Function to observe variables with nan values and their percentages

    Args:
        data: DataFrame
        variables: list
        variable_type: str
    
    Return:
        print: Variables that meet this condition
    """
    
    print(f'Variables {variable_type}')
    
    for var in variables:    
        if data[var].isnull().sum() > 0:
            print(f'{var}: {data[var].isnull().mean()*100:0.2f}%')
            

# Función para graficar los datos con valores nulos
def plotting_nan_values(data:pd.DataFrame) -> any:

    """
    Function to plot nan values

    Args:
        data: DataFrame
    
    Return:
        Dataviz
    """

    vars_with_nan = [var for var in data.columns if data[var].isnull().sum() > 0]
    
    if len(vars_with_nan) == 0:
        print('No se encontraron variables con nulos')
    
    else:
        # Plotting
        plt.figure(figsize=(14, 6))
        data[vars_with_nan].isnull().mean().sort_values(ascending=False).plot.bar(color='crimson', width=0.4, 
                                                                                  edgecolor='skyblue', lw=0.75)
        plt.axhline(1/3, color='#E51A4C', ls='dashed', lw=1.5, label='⅓ Missing Values')
        plt.ylim(0, 1)
        plt.xlabel('Predictors', fontsize=12)
        plt.ylabel('Percentage of missing data', fontsize=12)
        plt.xticks(fontsize=10, rotation=25)
        plt.yticks(fontsize=10)
        plt.legend()
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()


# Variables estratificadas por clases
# Función para obtener la estratificación de clases/target
def class_distribution(data:pd.DataFrame, target:str) -> any:
    
    """
    Function to get balance by classes

    Args:
        data: DataFrame
        target: str
    
    Return:
        Dataviz
    """

    # Distribución de clases
    distribucion = data[target].value_counts(normalize=True)

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 4))

    # Ajustar el margen izquierdo de los ejes para separar las barras del eje Y
    ax.margins(y=0.2)

    # Ajustar la posición de las etiquetas de las barras
    ax.invert_yaxis()

    # Crear gráfico de barras horizontales con la paleta de colores personalizada
    ax.barh(distribucion.index, distribucion.values, align='center', color='darkblue',
            edgecolor='white', height=0.5, linewidth=0.5)

    # Definir título y etiquetas de los ejes
    ax.set_title('Distribución de clases\n', fontsize=14)
    ax.set_xlabel('Porcentajes', fontsize=12)
    ax.set_ylabel(f'{target}'.capitalize(), fontsize=12)

    # Mostrar el gráfico
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()
    plt.show()


# Función para obtener la matriz de correlaciones entre los predictores
def correlation_matrix(data:pd.DataFrame, continuous:list) -> any:
    
    """
    Function to plot correlation_matrix

    Args:
        data: DataFrame
        continuous: list
    
    Return:
        Dataviz
    """
    
    correlations = data[continuous].corr(method='pearson', numeric_only=True)
    plt.figure(figsize=(17, 10))
    sns.heatmap(correlations, vmax=1, annot=True, cmap='gist_yarg', linewidths=1, square=True)
    plt.title('Matriz de Correlaciones\n', fontsize=14)
    plt.xticks(fontsize=10, rotation=25)
    plt.yticks(fontsize=10, rotation=25)
    plt.tight_layout()


# Covarianza entre los predictores
# Función para obtener una matriz de covarianza con los predictores
def covariance_matrix(data:pd.DataFrame):
    
    """
    Function to get mapped covariance matrix

    Args:
        data: DataFrame
        map_function: function
    
    Return:
        DataFrame
    """
    
    cov_matrix = data.cov()
    
    # Crear una matriz de ceros con el mismo tamaño que la matriz de covarianza
    zeros_matrix = np.zeros(cov_matrix.shape)
    
    # Crear una matriz diagonal de ceros reemplazando los valores de la diagonal de la matriz con ceros
    diagonal_zeros_matrix = np.diag(zeros_matrix)
    
    # Reemplazar la diagonal de la matriz de covarianza con la matriz diagonal de ceros
    np.fill_diagonal(cov_matrix.to_numpy(), diagonal_zeros_matrix)
    
    # Mapear los valores con etiquetas para saber cómo covarian los predictores
    cov_matrix = cov_matrix.applymap(lambda x: 'Positivo' if x > 0 else 'Negativo' if x < 0 else '')
    
    return cov_matrix


# Función para graficar la covarianza entre los predictores
def plotting_covariance(X:pd.DataFrame, y:pd.Series, continuous:list, n_iter:int) -> any:
  
    """
    Function to plot covariance matrix choosing some random predictors

    Args:
        X: DataFrame
        y: Series
        continuous: list
        n_iter: int
    
    Return:
        DataViz
    """
    
    # Semilla para efectos de reproducibilidad
    np.random.seed(42)
  
    for _ in range(n_iter):
        # Creamos una figura con tres subfiguras
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle('Covariance Plots\n', fontsize=14)

        # Seleccionamos dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la primera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax1, hue=y, palette='viridis', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Seleccionamos dos nuevas variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la segunda subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax2, hue=y, palette='flare', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax2.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Seleccionamos otras dos variables aleatorias del Dataframe
        var1 = np.random.choice(X[continuous].columns)
        var2 = np.random.choice(X[continuous].columns)
        while var1 == var2:
            var2 = np.random.choice(X[continuous].columns)

        # Graficamos la covarianza en la tercera subfigura
        sns.scatterplot(x=var1, y=var2, data=X[continuous], ax=ax3, hue=y, palette='Set1', alpha=0.6)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.xticks()
        plt.yticks()
        ax3.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Mostramos la figura
        fig.tight_layout()


# Función para observar el comportamiento de variables temporales
def temporaries_plots(data:pd.DataFrame, variables:list, target:str):
    
    """
    Function to get mean temporaries vs target

    Args:
        data: DataFrame
        variables: list
        target: str
    
    Return:
        Dataviz
    """
    
    for var in data[variables]:
        plt.figure(figsize=(18, 5))
        data.groupby(var)[target].mean().plot(color='gold')
        plt.title(f'Las medias de {target}\n', fontsize=14)
        plt.ylabel('Porcentajes')
        plt.xticks(fontsize=10, rotation=25)
        plt.yticks(fontsize=10)
        plt.grid()
        plt.legend()
        plt.tight_layout()


# Diagnóstico de variables
# Función para observar el comportamiento de variables continuas
def diagnostic_plots(data:pd.DataFrame, variables:list) -> any:

    """
    Function to get diagnostic graphics into 
    numerical (continous and discretes) predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
        
    for var in data[variables]:
        fig, (ax, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 5))
        fig.suptitle('Diagnostic Plots', fontsize=16)
        plt.rcParams.update({'figure.max_open_warning': 0}) # Evitar un warning

        # Histogram Plot
        plt.subplot(1, 4, 1)
        plt.title('Histogram Plot')
        sns.histplot(data[var], bins=25, color='midnightblue', edgecolor='white', lw=0.5)
        plt.axvline(data[var].mean(), color='#E51A4C', ls='dashed', lw=1.5, label='Mean')
        plt.axvline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.legend(fontsize=10)
        
        # CDF Plot
        plt.subplot(1, 4, 2)
        plt.title('CDF Plot')
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).cdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        empiricaldist.Cdf.from_seq(data[var], normalize=True).plot(color='chartreuse')
        plt.xlabel(var)
        plt.xticks(rotation=25)
        plt.ylabel('Probabilidad')
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper left')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # PDF Plot
        plt.subplot(1, 4, 3)
        plt.title('PDF Plot')
        kurtosis = stats.kurtosis(data[var], nan_policy='omit') # Kurtosis
        skew = stats.skew(data[var], nan_policy='omit') # Sesgo
        xs = np.linspace(data[var].min(), data[var].max())
        ys = stats.norm(data[var].mean(), data[var].std()).pdf(xs) # Distribución normal a partir de unos datos
        plt.plot(xs, ys, color='cornflowerblue', ls='dashed')
        sns.kdeplot(data=data, x=data[var], fill=True, lw=0.75, color='crimson', alpha=0.5, edgecolor='white')
        plt.text(s=f'Skew: {skew:0.2f}\nKurtosis: {kurtosis:0.2f}',
                 x=0.25, y=0.65, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='center', horizontalalignment='center')
        plt.ylabel('Densidad')
        plt.xticks(rotation=25)
        plt.xlabel(var)
        plt.xlim()
        plt.legend(['Distribución normal', var], fontsize=8, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)

        # Boxplot & Stripplot
        plt.subplot(1, 4, 4)
        plt.title('Boxplot')
        sns.boxplot(data=data[var], width=0.4, color='silver', sym='*',
                    boxprops=dict(lw=1, edgecolor='white'),
                    whiskerprops=dict(color='white', lw=1),
                    capprops=dict(color='white', lw=1),
                    medianprops=dict(),
                    flierprops=dict(color='red', lw=1, marker='o', markerfacecolor='red'))
        plt.axhline(data[var].quantile(0.75), color='magenta', ls='dotted', lw=1.5, label='IQR 75%')
        plt.axhline(data[var].median(), color='gold', ls='dashed', lw=1.5, label='Median')
        plt.axhline(data[var].quantile(0.25), color='cyan', ls='dotted', lw=1.5, label='IQR 25%')
        plt.xlabel(var)
        plt.tick_params(labelbottom=False)
        plt.ylabel('Unidades')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        
        fig.tight_layout()


# Revisar la cardinalidad de variables categóricas y discretas
# Función para graficar variables categóricas
def categoricals_plot(data:pd.DataFrame, variables:list) -> any:
    
    """
    Function to get distributions graphics into 
    categoricals and discretes predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    # Definir el número de filas y columnas para organizar los subplots
    num_rows = (len(variables) + 1) // 2  # Dividir el número de variables por 2 y redondear hacia arriba
    num_cols = 2  # Dos columnas de gráficos por fila

    # Crear una figura y ejes para organizar los subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 30))
    
    plt.suptitle('Categoricals Plots', fontsize=24, y=0.95)
    
    # Asegurarse de que 'axes' sea una matriz 2D incluso si solo hay una variable
    if len(variables) == 1:
        axes = axes.reshape(1, -1)

    # Iterar sobre las variables y crear gráficos para cada una
    for i, var in enumerate(variables):
        row, col = i // 2, i % 2  # Calcular la fila y columna actual

        # Crear un gráfico de barras en los ejes correspondientes
        temp_dataframe = pd.Series(data[var].value_counts(normalize=True))
        temp_dataframe.sort_values(ascending=False).plot.bar(color='royalblue', edgecolor='skyblue', ax=axes[row, col])
        
        # Añadir una línea horizontal a 5% para resaltar las categorías poco comunes
        axes[row, col].axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5)
        axes[row, col].set_ylabel('Porcentajes')
        axes[row, col].set_xlabel(var)
        axes[row, col].set_xticklabels(temp_dataframe.index, rotation=25)
        axes[row, col].grid(color='white', linestyle='-', linewidth=0.25)
    
    # Ajustar automáticamente el espaciado entre subplots
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # El argumento rect controla el espacio para el título superior
    plt.show()


# # Puede quedar obsoleta
# def categoricals_plot2(data:pd.DataFrame, variables:list) -> any:

#     """
#     Function to get distributions graphics into 
#     categoricals predictors

#     Args:
#         data: DataFrame
#         variables: list
    
#     Return:
#         Dataviz
#     """
    
#     plt.suptitle('Categoricals Plot', fontsize=16)
#     for var in variables:
#         temp_dataframe = pd.Series(data[var].value_counts() / len(data))

#         # Graficar con los porcentajes
#         temp_dataframe.sort_values(ascending=False).plot.bar(color='lavender', edgecolor='skyblue')

#         # Añadir una línea horizontal a 5% para resaltar categorías poco comunes
#         plt.axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5)
#         plt.ylabel('Porcentajes')
#         plt.xlabel(var)
#         plt.xticks(rotation=25)
#         plt.grid(color='white', linestyle='-', linewidth=0.25)
#         plt.show()


# Función para graficar las categóricas segmentadas por el target
def categoricals_hue_target(data:pd.DataFrame, variables:list, target:str) -> any:
    
    # Graficos de cómo covarian algunas variables con respecto al target
    paletas = ['rocket', 'mako', 'crest', 'magma', 'viridis', 'flare']
    np.random.seed(11)

    for var in data[variables]:
        plt.figure(figsize=(12, 6))
        plt.title(f'{var} segmentado por {target}\n', fontsize=12)
        sns.countplot(x=var, hue=target, data=data, edgecolor='white', lw=0.5, palette=np.random.choice(paletas))
        plt.ylabel('Cantidades')
        plt.xticks(fontsize=12, rotation=25)
        plt.yticks(fontsize=12)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.tight_layout()


# Test de Normalidad de D’Agostino y Pearson
# Función para observar el comportamiento de las variables continuas en una prueba de normalidad
# Y realizar un contraste de hipótesis para saber si se asemeja a una distribución normal
def normality_test(data:pd.DataFrame, variables:list) -> any:
    
    """
    Function to get Normality Test into continuous predictors

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    print("\x1b[0;36m" + "\t\t\tPrueba de Normalidad")
    
    # Q-Q Plot
    for var in data[variables]:
        plt.figure(figsize=(5.5, 3.5))
        stats.probplot(data[var], dist='norm', plot=plt)
        plt.xlabel(var)
        plt.xticks(rotation=25)
        plt.grid(color='white', linestyle='-', linewidth=0.25)
        plt.show()
        
        # Contraste de Hipótesis
        print(var)
        p_value = stats.normaltest(data[var], nan_policy='omit')[1]
        print(f'P-value: {p_value:0.3f}')
        
        if p_value < 5e-2:
            display(Latex(f'Se rechaza la $H_0$ sugiere que los datos no se ajustan de manera significativa a una distribución normal'))
        else:
            display(Latex(f'Se acepta la $H_1$ sugiere que los datos se ajustan de manera significativa a una distribución normal'))
        print()


# def normality_test_v2(data:pd.DataFrame, variables:list):

#     """
#     Esta función falta optimizarla pero es mejor opción
#     """
#     display(Latex('Si el $pvalue$ < 0.05; se rechaza la $H_0$ sugiere que los datos no se ajustan de manera significativa a una distribución normal'))
    
#     # Configurar figura
#     fig = plt.figure(figsize=(24, 20))
#     plt.suptitle('Prueba de Normalidad', fontsize=18)
#     gs = gridspec.GridSpec(nrows=len(variables) // 3+1, ncols=3, figure=fig)
    
#     for i, var in enumerate(variables):

#         ax = fig.add_subplot(gs[i//3, i % 3])

#         # Gráfico Q-Q
#         stats.probplot(data[var], dist='norm', plot=ax)
#         ax.set_xlabel(var)
#         ax.set_xticks(ax.get_xticks())
#         ax.set_xticklabels(ax.get_xticklabels())
#         ax.grid(color='white', linestyle='-', linewidth=0.25)

#         # P-value
#         p_value = stats.normaltest(data[var])[1]
#         ax.text(0.8, 0.9, f"p-value={p_value:0.3f}", transform=ax.transAxes, fontsize=13) 

#     plt.tight_layout(pad=3)
#     plt.show()


# Definir la transformación de Yeo-Johnson
def yeo_johnson_transform(x):
    y = np.where(x >= 0, x + 1, 1 / (1 - x))
    y = np.sign(y) * (np.abs(y) ** 0.5)
    return y

def gaussian_transformation(data:pd.DataFrame, variables:list) -> dict:
    
    """
    Function to get Gaussian transformations of the variables

    Args:
        data: DataFrame
        variables: list
    
    Return:
        results: dict
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': np.log,
        'Sqrt': np.sqrt, 
        'Reciprocal': lambda x: 1/x, 
        'Exp': lambda x: x**2, 
        'Yeo-Johnson': yeo_johnson_transform
        }
    
    # Crear un diccionario para almacenar los resultados de las pruebas de normalidad
    results = dict()

    # Iterar a través de las variables y las transformaciones
    for var in data[variables].columns:
        mejores_p_value = 0
        mejor_transformacion = None
        
        for nombre_transformacion, transformacion in transformaciones_gaussianas.items():
            # Aplicar la transformación a la columna
            variable_transformada = transformacion(data[var])
            
            # Calcular el p-value de la prueba de normalidad
            p_value = stats.normaltest(variable_transformada)[1]
            
            # Actualizar el mejor p-value y transformación si es necesario
            if p_value > mejores_p_value:
                mejores_p_value = p_value
                mejor_transformacion = nombre_transformacion
        
        # Almacenar el resultado en el diccionario
        results[var] = mejor_transformacion
        
    return results


# Graficar la comparativa entre las variables originales y su respectiva transformación
def graficar_transformaciones(data:pd.DataFrame, continuous:list, transformacion:dict) -> any:
    
    """
    Function to plot compare Gaussian transformations of the variables and their original state

    Args:
        data: DataFrame
        variables: list
    
    Return:
        Dataviz
    """
    
    # Definir las transformaciones gaussianas a utilizar
    transformaciones_gaussianas = {
        'Log': np.log,
        'Sqrt': np.sqrt, 
        'Reciprocal': lambda x: 1/x, 
        'Exp': lambda x: x**2, 
        'Yeo-Johnson': yeo_johnson_transform
        }
    
    data = data.copy()
    data = data[continuous]
    
    for variable, transformacion_name in transformacion.items():
        # Obtener datos originales
        data_original = data[variable]
        
        # Obtener la transformación correspondiente
        transformacion_func = transformaciones_gaussianas.get(transformacion_name)
        
        # Aplicar transformación
        data_transformada = transformacion_func(data_original)

        # Crear figura con 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Graficar histograma datos originales 
        hist_kws = {'color': 'royalblue', 'lw': 0.5}
        sns.histplot(data_original, ax=ax1, kde=True, bins=50, **hist_kws)
        ax1.set_title('Original')
        ax1.grid(color='white', linestyle='-', linewidth=0.25)

        # Graficar histograma datos transformados
        sns.histplot(data_transformada, ax=ax2, kde=True, bins=50, **hist_kws)
        ax2.set_title(f'{transformacion_name}')
        ax2.grid(color='white', linestyle='-', linewidth=0.25)
        
        # Cambiar color del KDE en ambos gráficos
        for ax in [ax1, ax2]:
            for line in ax.lines:
                line.set_color('crimson')

        # Mostrar figura
        plt.tight_layout()
        plt.show()


# Función para observar todos P-values de los predictores
def every_pvalue(X_train:pd.DataFrame, y_train):

    """
    Function to observe every P-value on features

    Args:
        X_train: Dataframe
        y_train: Dataframe or Series
    
    Return:
        Dataviz
    """
    
    # Statistic test
    stat_test = GenericUnivariateSelect(mutual_info_classif, mode='fwe')
    stat_test.fit(X_train, y_train)

    # Get obtain the p-values from the test
    pvalues = pd.DataFrame(stat_test.pvalues_, columns=['P-value'])
    pvalues.index = X_train.columns
    pvalues.sort_values(ascending=True, by='P-value', inplace=True)
    
    # Plotting
    pvalues.columns = ['']
    pvalues.plot.bar(rot=25, color='dodgerblue', alpha=0.7, edgecolor='coral', lw=0.75)
    plt.title('Feature importance based on P-values')
    plt.axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5, label='P-value')
    plt.ylabel('P-value')
    plt.legend()
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()
    

# Función para observar aquellos predictores que son estadísticamente significativos
def pvalues(X_train:pd.DataFrame, y_train):
    
    """
    Function to observe the features that are statistically significant
    with respect to the target (below 5%)

    Args:
        X_train: Dataframe
        y_train: Series
    
    Return:
        Dataviz & Dataframe.index
    """
    
    stat_test = GenericUnivariateSelect(mutual_info_classif, mode='fwe')
    stat_test.fit(X_train, y_train)

    # Get obtain the p-values from the test
    pvalues = pd.DataFrame(stat_test.pvalues_, columns=['P-value'])
    pvalues.index = X_train.columns
    pvalues.sort_values(ascending=True, by='P-value', inplace=True)
    
    # Plotting
    pvalues_filtered = pvalues[pvalues['P-value'] <= 0.05]
    p_values = pvalues_filtered.copy()
    pvalues_filtered.columns = ['']
    pvalues_filtered.plot.bar(rot=25, color='dodgerblue', alpha=0.7, edgecolor='coral', lw=0.75)
    plt.title('Feature importance based on P-values below 5%')
    plt.axhline(y=0.05, color='#E51A4C', ls='dashed', lw=1.5, label='P-value')
    plt.ylabel('P-value')
    plt.legend(loc='upper left')
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()
    
    # P-values with index
    p_values = p_values.index
    return p_values


# Regla de Freedman y Diaconis
def freedman_and_diaconis_rule(data:pd.DataFrame, variables:list) -> dict:
    
    """
    Function to get optimal number of bins using 
    the Freedman and Diaconis rule
    
    Args:
        data: DataFrame
        variables: list
    
    Return:
        dict
    """
    
    bins = list()
    
    for var in data[variables]:
        iqr = np.percentile(data[var], 75) - np.percentile(data[var], 25)
        bin_width = 2 * iqr / np.power(len(data), 1/3)
        num_bins = np.int8((np.max(data[var]) -  np.min(data[var])) / bin_width)  + 1
        bins.append(num_bins)
    
    optimal_bins = dict(zip(variables, bins))
    return optimal_bins


# Función para observar la calibración de los modelos posterior a las técnicas de remuestreo
def plot_calibration_curve(y_true:np.array, y_pred:np.array, bins:float, model_name) -> any:
    
    """
    Function to observe calibration between fraction of positives and the mean of
    predicteds values

    Args:
        y_true: np.array (y_test)
        y_pred: np.array (model.predict_proba(X_test)[:, 1])
        bins: float
        model_name: Classifier (mainly from Scikit-learn)
    
    Return:
        Dataviz
    """

    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred, 
                                                                    n_bins=bins, strategy='uniform')
    max_val = max(mean_predicted_value)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title('Calibration Curve')
    plt.plot(mean_predicted_value, fraction_of_positives, label=type(model_name).__name__, 
             c='xkcd:pale cyan')
    plt.plot(np.linspace(0, max_val, bins), np.linspace(0, max_val, bins), c='red',
             linestyle='--', label='Perfect calibration')
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of positive examples')
    plt.legend(loc='upper left')
    plt.grid(color='white', linestyle='-', linewidth=0.25)

    plt.subplot(1, 2, 2)
    plt.title('Density')
    plt.hist(y_pred, range=(0, 1), bins=bins, density=False, stacked=True, alpha=0.3, 
             color='xkcd:dark cyan', edgecolor='white', lw=0.5)
    plt.xlabel('Probability Predictions')
    plt.ylabel('Fraction of examples')
    plt.grid(color='white', linestyle='-', linewidth=0.25)
    plt.tight_layout()


# Función de matriz de confusión
def cnf_matrix(y_true:pd.DataFrame, y_pred:pd.Series, threshold=0.5):
    
    sns.set_theme(context='notebook', style=plt.style.use('dark_background'))
    y_true = y_true.to_numpy().reshape(-1)
    y_pred = np.where(y_pred > threshold, 1, 0)
    
    cm = confusion_matrix(y_true, y_pred)

    # Calculamos TP, FN, FP, TN a partir de la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    
    # Creamos la figura y los ejes del heatmap con un tamaño adecuado
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(font_scale=1.2)

    # Creamos la matriz de confusión
    sns.heatmap([[tp, fp], [fn, tn]], annot=True, cmap='magma', fmt='g', square=True, linewidths=1,
                yticklabels=['Verdaderos Positivos', ''], 
                xticklabels=['Falsos Negativos', 'Verdaderos Negativos'], ax=ax)
                    
    # Configura los ejes del heatmap
    ax.set_title(f'Confusion Matrix\n', fontsize=14)
    plt.tight_layout()
   