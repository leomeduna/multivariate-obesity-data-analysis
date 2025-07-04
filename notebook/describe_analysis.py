# %%

# Lendo o arquivo CSV que contém nosso dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% 
df = pd.read_csv("C:/research_multivariate_obesity/data/ObesityDataSet_raw_and_data_sinthetic.csv")
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
                                    'CH2O': 'agua_dia_freq',
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
# %%

# (EDA): Aplicando uma árvore de decisão para entender melhor os dados

# Passo 1: Definindo as variaveis que vao para o EDA e a nossa 'target' 
features = ['idade',
            'altura',	
            'peso',	
            'historico_familia_obesidade',	
            'eh_consumidor_altas_calorias_freq',	
            'come_vegetais_freq',	
            'refs_principais_dia',
            'eh_fumante',
            'agua_dia_freq',
            'faz_controle_calorias_dia',
            'atividades_fisicas_freq',
            'eletronicos_freq']
target = ['nivel_obesidade']

X = df_final[features]
y = df_final[target]

# %%
# Análise multivariada sobre os dados
groupby_mean_numerical = df_final.groupby('nivel_obesidade')[features].mean()
groupby_mean_numerical

# %%
categorical_cols = [
    'come_entre_refs',
    'bebe_alcool_freq'
                    ]

groupby_freq_categorical = df_final.groupby('nivel_obesidade')[categorical_cols].count()
groupby_freq_categorical 

# %%
#To-do: Visualização das distribuições dos dados numéricos com histogramas e boxplots
plt.scatter(
    y=df_final['atividades_fisicas_freq'],
    x=df_final['peso'],
    marker= 'o'
    )
plt.xlabel(xlabel='Peso'),
plt.ylabel(ylabel='Atividades_fisicas_freq'),
plt.grid(True),
plt.show()

plt.scatter(
    y=df_final['altura'],
    x=df_final['peso'],
    marker= 'o'
)
plt.xlabel(xlabel='Altura'),
plt.ylabel(ylabel='Peso'),
plt.grid(True)
plt.show()
# %%

# Analisando a distribuição.
plt.hist(
    bins=30,
    x=(df_final['peso']) 
)
plt.title('Peso')
plt.grid(True),
plt.show()

plt.hist(
    bins=30,
    x=df_final['altura'] 
)
plt.title('Altura')
plt.grid(True),
plt.show()
# %%
df_final['nivel_obesidade'].unique()

# %%
obesos = ['Obesity_Type_I',
          'Obesity_Type_II',
          'Obesity_Type_III']
df_obesos = df_final[df_final['nivel_obesidade'].str.contains(
    'Obesity'
    )] 

prop_obesos_vs_geral = (len(df_obesos) / len(df_final)) * 100
print(prop_obesos_vs_geral)
# %%
print(len(df_final))
print(len(df_obesos))
print(prop_obesos_vs_geral)

# %%
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X=X, y=y) 


# %%

plt.figure(dpi=400, figsize=(12,4))

tree.plot_tree(
    model,
    max_depth=4,
    feature_names=features,
    class_names=model.classes_,
    filled=True
               )