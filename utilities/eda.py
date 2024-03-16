# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Pyfunctions ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Librerías y/o depedencias
from IPython.display import display, Latex
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import empiricaldist
sns.set_theme(context='notebook', style=plt.style.use('dark_background'))


# Función para capturar los tipos de variables
def capture_variables(data: pd.DataFrame) -> tuple:
    
    """
    Function to capture the types of Dataframe variables

    Args:
        dataframe: DataFrame
    
    Return:
        variables: A tuple of lists
    
    The order to unpack variables:
    1. continous
    2. categoricals
    3. discretes
    4. temporaries
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


# Función para observar variables con datos nulos y su porcentaje
def nan_values(data: pd.DataFrame, variables: list, variable_type: str) -> any:
    
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
def plotting_nan_values(data: pd.DataFrame) -> any:

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


# Función para obtener la estratificación de clases/target
def class_distribution(data: pd.DataFrame, target: str) -> any:
    
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
def correlation_matrix(data: pd.DataFrame, continuous: list) -> any:
    
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


# Función para obtener una matriz de covarianza con los predictores
def covariance_matrix(data: pd.DataFrame):
    
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
def plotting_covariance(X: pd.DataFrame, y: pd.Series, continuous: list, n_iter: int) -> any:
  
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
def temporaries_plots(data: pd.DataFrame, variables: list, target: str):
    
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


# Función para observar el comportamiento de variables continuas
def diagnostic_plots(data: pd.DataFrame, variables: list) -> any:

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


# Función para graficar variables categóricas y discretas
def categoricals_plot(data: pd.DataFrame, variables: list) -> any:
    
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


# Función para graficar las categóricas segmentadas por el target
def categoricals_hue_target(data: pd.DataFrame, variables: list, target: str) -> any:
    
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
def normality_test(data: pd.DataFrame, variables: list) -> any:
    
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
  