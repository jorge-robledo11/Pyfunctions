# • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ Utilidades ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ •
# Utilidades
import pandas as pd
import numpy as np

# X = data.loc[:, data.columns != «Target»] Dataframe
# y = data.iloc[:, data.columns == «Target»] Serie

def map_function(x:float) -> str:
    
    if x > 0:
        return 'Positively'
    
    elif x < 0:
        return 'Negatively'
    
    else:
        return ''