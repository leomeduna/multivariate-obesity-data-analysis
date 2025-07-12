# %%
# Lendo o arquivo CSV que contém nosso dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importações adicionais que incluí para os passos seguintes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency, f_oneway # Para ANOVA

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
df_alterado 
# %%
# Função para mapear variáveis binárias "Yes"/"No" para 1/0
def map_yes_no_to_binary(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    for col in columns:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].map({'yes': 1, 'no': 0})
    return dataframe

binary_cols = [
    'historico_familia_obesidade', 
    'eh_consumidor_altas_calorias_freq', 
    'eh_fumante', 
    'faz_controle_calorias_dia'
]

df_final = df_alterado.copy()
df_final = map_yes_no_to_binary(df_final, binary_cols)

print(df_final.head())
print(df_final.info())

# %% 
# Tratamento de variáveis categóricas ordinais e nominais
print("\nValores únicos para 'come_entre_refs':", df_final['come_entre_refs'].unique())
print("Valores únicos para 'bebe_alcool_freq':", df_final['bebe_alcool_freq'].unique())
# ... (outras impressões de unique)

ordinal_mapping_caec = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
ordinal_mapping_calc = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

df_final['come_entre_refs_encoded'] = df_final['come_entre_refs'].map(ordinal_mapping_caec)
df_final['bebe_alcool_freq_encoded'] = df_final['bebe_alcool_freq'].map(ordinal_mapping_calc)

print("\nVerificando as colunas codificadas:")
print(df_final[['come_entre_refs', 'come_entre_refs_encoded', 'bebe_alcool_freq', 'bebe_alcool_freq_encoded']].head())
# %%
# Identificando as variáveis numéricas ...
numerical_cols = ['idade', 'altura', 'peso', 'come_vegetais_freq', 'refs_principais_dia', 
                  'agua_dia_freq', 'atividades_fisicas_freq', 'eletronicos_freq']
# ...

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df_final[col], kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()


# %%
# Análise multivariada sobre os dados
groupby_mean_numerical = df_final.groupby('nivel_obesidade')[numerical_cols].mean()
groupby_mean_numerical

# %%
categorical_cols = [
    'come_entre_refs',
    'bebe_alcool_freq'
                    ]

groupby_freq_categorical = df_final.groupby('nivel_obesidade')[categorical_cols].count()
groupby_freq_categorical 

# %%
# 2.1. Análise Univariada 
# %%
target_col = 'nivel_obesidade'

print("\n--- Análise Numéricas vs. Nível de Obesidade (Box Plots e ANOVA) ---")
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_col, y=col, data=df_final, order=df_final[target_col].value_counts().index)
    plt.title(f'{col} por {target_col}')
    plt.xlabel('Nível de Obesidade')
    plt.ylabel(col)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Teste ANOVA: ...
    groups = [df_final[df_final[target_col] == level][col].dropna() for level in df_final[target_col].unique()]
    f_stat, p_val_anova = f_oneway(*groups)
    print(f"ANOVA para {col} vs. {target_col}: F-statistic={f_stat:.2f}, p-value={p_val_anova:.4f}")
    if p_val_anova < 0.05:
        print(f"  => Há uma diferença significativa nas médias de {col} entre os níveis de obesidade.")
    else:
        print(f"  => Não há diferença significativa nas médias de {col} entre os níveis de obesidade.")
# %%
# Para variáveis categóricas binárias (0/1) vs. 'nivel_obesidade':
print("\n--- Análise Binárias (0/1) vs. Nível de Obesidade (Qui-Quadrado) ---")
for col in binary_cols:
    contingency_table = pd.crosstab(df_final[col], df_final[target_col])
    print(f"\nTabela de Contingência para {col} vs. {target_col}:\n{contingency_table}")
    
    chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    print(f"Teste Qui-Quadrado para {col} vs. {target_col}: p-value = {p_value_chi2:.4f}")
    if p_value_chi2 < 0.05:
        print(f"  => Há uma associação significativa entre {col} e {target_col}.")
    else:
        print(f"  => Não há associação significativa entre {col} e {target_col}.")

    # Plotagem de proporções (countplot)

# Para variáveis categóricas nominais e ordinais vs. 'nivel_obesidade':
print("\n--- Análise Categóricas (Nominais/Ordinais) vs. Nível de Obesidade (Qui-Quadrado) ---")
for col in categorical_cols + ['come_entre_refs', 'bebe_alcool_freq']:
    contingency_table = pd.crosstab(df_final[col], df_final[target_col])
    print(f"\nTabela de Contingência para {col} vs. {target_col}:\n{contingency_table}")
    
    chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    print(f"Teste Qui-Quadrado para {col} vs. {target_col}: p-value = {p_value_chi2:.4f}")
    if p_value_chi2 < 0.05:
        print(f"  => Há uma associação significativa entre {col} e {target_col}.")
        n_obs = contingency_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n_obs * (min(contingency_table.shape) - 1)))
        print(f"  => V de Cramer: {cramer_v:.4f}")
    else:
        print(f"  => Não há associação significativa entre {col} e {target_col}.")
    
    # Plotagem de proporções (countplot)
# %%
obesos = ['Obesity_Type_I',
          'Obesity_Type_II',
          'Obesity_Type_III']
df_obesos = df_final[df_final['nivel_obesidade'].str.contains(
    'Obesity'
    )] 

prop_obesos_vs_geral = (len(df_obesos) / len(df_final)) * 100
print(len(df_final))
print(len(df_obesos))
print(prop_obesos_vs_geral)

# %%
#from sklearn import tree
#model = tree.DecisionTreeClassifier()
#model.fit(X=X, y=y) 


# %%

#plt.figure(dpi=400, figsize=(12,4))

#tree.plot_tree(
#    model,
#    max_depth=4,
#    feature_names=features,
#    class_names=model.classes_,
#    filled=True
#               )