# %%

# Lendo o arquivo CSV que contém nosso dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% 
df = pd.read_csv(r"C:\research_multivariate_obesity\data\ObesityDataSet_raw_and_data_sinthetic.csv")
df.head()
# %%

# Renomeando as colunas principais
def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.rename(columns={
                                    'Gender': 'sexo',
                                    'Age': 'idade',
                                    'Height': 'altura',
                                    'Weight': 'peso',
                                    'family_history_with_overweight': 'historico_familia_obesidade',
                                    'FAVC':'eh_consumidor_altas_calorias_freq',
                                    'FCVC': 'come_vegetais_freq',
                                    'NCP': 'refs_principais_dia',
                                    'CAEC': 'come_entre_refs',
                                    'SMOKE': 'eh_fumante',
                                    'CH20': 'agua_dia_freq',
                                    'SCC': 'faz_controle_calorias_dia',
                                    'FAF': 'atividades_fisicas_freq',
                                    'TUE': 'eletronicos_freq',
                                    'CALC': 'bebe_alcool_freq',
                                    'MTRANS': 'transporte_usado',
                                    'NObeyesdad': 'nivel_obesidade'
                                    }
                                )
                                        

df_alterado = df.pipe(rename_columns)
df_alterado.columns 
# %%

# Função para pegar variáveis binárias objects e transforma-lás em binarias
def object_to_binary(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.replace({'yes': 1, 'no': 0})
    

df_final = df_alterado.pipe(
    object_to_binary
)
df_final
# %% 

df_final['bebe_alcool_freq'] = df_final['bebe_alcool_freq'].replace(0, 'no')
df_final