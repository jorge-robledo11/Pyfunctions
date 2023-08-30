import pandas as pd
import numpy as np
import re
import logging
import warnings
warnings.simplefilter('ignore')

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler
file_handler = logging.FileHandler('module.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Función para preprocesar datos
def preprocessing(path:str) -> pd.DataFrame:

    try:
        # Cargar y hacer lectura de los datos
        data:pd.DataFrame = pd.read_pickle(path)
        logger.info('1. Carga y lectura realizada exitosamente')
        
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
            data[int_cols] = data[int_cols].astype(np.int32)
            data[object_cols] = data[object_cols].astype('category')

            end = data.memory_usage(deep=True).sum() / 1024 ** 2
            saved = (start - end) / start * 100

            return data

        # Reducir el espacio en memoria
        downcast_dtypes(data=data)
        logger.info('2. Reducción de espacio en memoria realizado exitosamente')

        # Manejar el nombre de los predictores
        data = data.rename(columns=lambda col: str(col).lower().strip())
        logger.info('3. Renombramiento de los predictores realizado exitosamente')

        # Eliminar duplicados
        data.drop_duplicates(inplace=True, ignore_index=True)
        logger.info('4. Remoción de duplicados realizado exitosamente')

        # Reemplazar valores faltantes de distintas fuentes a np.nan
        data = data.fillna(np.nan)
        data = data.replace({
            'ERROR': np.nan,
            '': np.nan,
            'None': np.nan,
            'n/a': np.nan,
            'N/A': np.nan,
            'NULL': np.nan, 
            'NA': np.nan,
            'NAN': np.nan})
        logger.info('5. Reemplazo de valores faltantes de distintas fuentes realizado exitosamente')

        # Transformar los predictores temporales y cambiar su formato
        t = data.filter(regex='fecha|date|tiempo|time').columns
        data[t] = data[t].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d'))
        logger.info('6. Transformación de predictores temporales realizado exitosamente')

        # Uniformizar los predictores categóricas
        categoricals = data.select_dtypes(include=['object', 'bool']).columns
        data[categoricals] = data[categoricals].applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
        logger.info('7. Transformación de predictores categóricos realizado exitosamente')

        # Remover predictores que su distribución supera el 33.33% como datos faltantes
        data.drop(data.columns[data.isnull().mean() > 1/3].to_list(), inplace=True, axis=1)
        logger.info('8. Remoción de predictores con valores faltantes en su distribución superior a 1/3 realizado exitosamente')

        # Exportar los datos preprocesados
        data.to_parquet('../data/silver-zone/preprocesado_vista360_historico.parquet')
        logger.info('10. Exportación de los datos pre-procesados realizado exitosamente')
        
    except Exception as e:
        print(type(e).__name__)
        
    finally:
        logger.info('¡Pre-procesamiento realizado exitosamente!')
