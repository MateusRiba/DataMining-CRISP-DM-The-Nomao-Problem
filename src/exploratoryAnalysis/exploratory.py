from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import seaborn as sns
import matplotlib.pyplot as plt

processed_path = "C:\\Users\\mateu\\Arquivos de Programas Faculdade\\repositorios\\DataMining-CRISP-DM-The-Nomao-Problem\\data\\processed"
arff_path = "C:\\Users\\mateu\\Arquivos de Programas Faculdade\\repositorios\\DataMining-CRISP-DM-The-Nomao-Problem\\data\\raw\\phpDYCOet.arff"

#Carregando arquivo
data, meta = arff.loadarff(arff_path)

#Conversão para DataFrame do pandas
df = pd.DataFrame(data)

#Mapeamento visual da variavel alvo
df['Class'] = df['Class'].map({b'1': 'Não Duplicado', b'2': 'Duplicado'})

#Exibindo as primeiras linhas e os nomes das colunas do DataFrame
print("----------------------------------------------------")
print(df.columns)
print(df.head())
print("----------------------------------------------------")

#Coluna alvo é a coluna 'Class'
X = df.drop('Class', axis=1) #Dados de entrada
y = df['Class'] #Variavel alvo

#Divisção dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Visualização das dimensões dos conjuntos
print("----------------------------------------------------")
print(f"Dimensões do conjunto de treino: X: {X_train.shape}, y: {y_train.shape}")
print(f"Dimensões do conjunto de teste: X: {X_test.shape}, y: {y_test.shape}")
print("----------------------------------------------------")

#--------------------------------------
#Verificação de balanceamento das classes
train_class_distribution = y_train.value_counts(normalize=True)
test_class_distribution = y_test.value_counts(normalize=True)

print("----------------------------------------------------")
print("Distribuição das classes no conjunto de treino:")
print(train_class_distribution)

print("\nDistribuição das classes no conjunto de teste:")
print(test_class_distribution)
print("----------------------------------------------------")
#--------------------------------------

'''Sprint 2 - Verificação de valores ausentes'''

#QUantificação
n_nulos = df.isnull().sum()
n_nulos_percentage = (n_nulos / len(df)) * 100

#Verificação
for i in n_nulos.index:
    if n_nulos[i] > 0:
        print(f"A coluna '{i}' possui {n_nulos[i]} valores nulos, correspondendo a {n_nulos_percentage[i]:.2f}% do total.")

#Impressoes e visualização
print("----------------------------------------------------")
print("Quantidade de valores nulos por coluna:")
print(n_nulos)
print("\nPorcentagem de valores nulos por coluna:")
print(n_nulos_percentage)
print("----------------------------------------------------")

plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', xticklabels=False, yticklabels=False)
plt.title('Mapa de Calor de Valores Nulos')
plt.show()

#--------------------------------------
'''Sprint 3 - Entendimentos de variaveis categóricas

Identificação de variáveis categóricas
Observando o dataset, muitas variaveis chamam a atenção, porem será escolhido:
clean_name_including(V7), City_Including: (V15), Zip_Including: (V23), Street_Including: (V31)
GeocoderPostalcodenumber_Including: (V79), Phone_Equality: (V92), Coordinates_Long_Equality: (V112), Coordinates_Lat_Equality: (V116)'''

#Visualizando a tabela apenas com: as colunas categóricas
categorical_columns = ['V7', 'V15', 'V23', 'V31', 'V79', 'V92', 'V112', 'V116']

#Dicionario para mapeamento de variaveis e Dicionario para visualização do mapa
var_names = {
    
    #Categoricas
    'V7': 'Clean_Name_Including',
    'V15': 'City_Including',
    'V23': 'Zip_Including',
    'V31': 'Street_Including',
    'V79': 'GeocoderPostalcodenumber_Including',
    'V92': 'Phone_Equality',
    'V112': 'Coordinates_Long_Equality',
    'V116': 'Coordinates_Lat_Equality',
    
    #Numericas 
    'V3': 'clean_name_levenshtein_sim',
    'V11': 'City_levenshtein_sim',
    'V19': 'Zip_levenshtein_sim',
    'V27': 'Street_levenshtein_sim',
    'V89': 'phone_diff',
    'V109': 'Coordinates_Long_diff',
}

category_map = {1: 'Não Inclui (n)', 2: 'Inclui (s)', 3: 'Inclusão Maxima (m)'}

#Conversão das variaveis categóricas de Bytes (B1, B2, B3) para inteiros (1, 2, 3)
for col in categorical_columns:
    if df[col].dtype == object:  # só mexe se for do tipo bytes
        df[col] = df[col].apply(lambda x: int(x.decode('utf-8')) if isinstance(x, bytes) else x)

#Visualização das primeiras linhas das colunas categóricas
print("----------------------------------------------------")
print("Primeiras linhas das colunas categóricas selecionadas:")
print(df[categorical_columns].head())
print("----------------------------------------------------")

#Geração de gráfivos para cada variável categórica

for col in var_names.keys():
    if col == 'V3' or col == 'V11' or col == 'V19' or col == 'V27' or col == 'V89' or col == 'V109':
        continue  # Pula as variáveis numéricas
    
    plt.figure(figsize=(10, 8))
    ax = sns.countplot(data=df, x=col, hue='Class')  # agora ax existe
    plt.title(f'Distribuição da variável {var_names[col]} ({col}) por Classe')
    
    # Ajusta os labels se for variável com 1,2,3
    if df[col].dropna().nunique() <= 3:
        new_labels = [f"{val} ({category_map.get(val,'?')})" for val in sorted(df[col].dropna().unique())]
        ax.set_xticks(range(len(new_labels)))
        ax.set_xticklabels(new_labels)

    plt.xlabel(f'{var_names[col]} ({col})')    
    plt.xticks(rotation=45)
    plt.legend(title='Class', loc='upper right')
    plt.tight_layout()
    plt.show()

#--------------------------------------
'''Sprint 4 - Conectando Categorias e Números

#Relação entre variaveis categóricas e numéricas. Pares usados:
#Clean Name Including (V7) e clean_name_levenshtein_sim (V3)
#City Including (V15) e City_levenshtein_sim (V11)
#Zip Including (V23) e Zip_levenshtein_sim (V19)
#Street Including (V31) e Street_levenshtein_sim (V27)
#Phone_Equality (V92) e phone_diff (V89)
#Coordinates_Long_Equality (V112) e Coordinates_Long_diff (V109)'''

#Definição de pares de variáveis categóricas e numéricas
pares = [
    ('V7', 'V3'),
    ('V15', 'V11'),
    ('V23', 'V19'),
    ('V31', 'V27'),
    ('V92', 'V89'),
    ('V112', 'V109')]

#Geração de gráficos para cada par
for coluna_cat, coluna_num in pares:
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df, x=coluna_cat, y=coluna_num, hue='Class')
    plt.title(f'Relação entre {var_names.get(coluna_cat, coluna_cat)} e {var_names.get(coluna_num, coluna_num)} por Classe')
    
    # Ajusta os labels se for variável com 1,2,3
    if df[coluna_cat].dropna().nunique() <= 3:
        new_labels = [f"{val} ({category_map.get(val,'?')})" for val in sorted(df[coluna_cat].dropna().unique())]
        plt.xticks(ticks=range(len(new_labels)), labels=new_labels)

    plt.xlabel(f'{var_names.get(coluna_cat, coluna_cat)} ({coluna_cat})')
    plt.ylabel(f'{var_names.get(coluna_num, coluna_num)} ({coluna_num})')
    plt.legend(title='Class', loc='upper right')
    plt.tight_layout()
    plt.show()

#--------------------------------------

'''Sprint Bonus - Estatisticas Descritivas, Distribuição das Features Numéricas, Análise de Correlação, Propoção de Duplicatos Classe X Categoria'''

numeric_columns = ['V3', 'V11', 'V19', 'V27', 'V89', 'V109']

#Estatísticas descritivas
print("----------------------------------------------------")
print("Estatísticas descritivas das variáveis numéricas selecionadas:")
print(df[numeric_columns].describe())
print("----------------------------------------------------")

#Distribuição das features numéricas
for col in numeric_columns:
    plt.figure(figsize=(10, 6))
    
    sns.histplot(data=df, x=col, hue='Class', kde=True, element='step', stat='density')
    #sns.boxplot(data=df, x=col, y='Class', showmeans=True, palette='Set2')
    
    plt.title(f'Distribuição da variável {var_names.get(col, col)} ({col}) por Classe')
    plt.xlabel(f'{var_names.get(col, col)} ({col})')
    plt.ylabel('Densidade')
    plt.legend(title='Class', loc='upper right')
    plt.tight_layout()
    plt.show()

#Análise de correlação ----> Está ERRADO!!!!

df['Class_num'] = df['Class'].map({'Duplicado': 1, 'Não Duplicado': 0})

corr_matrix = df.corr(numeric_only=True)
# Renomeando a coluna Class_num para evitar conflito
corr_matrix = corr_matrix.join(df['Class_num'].rename('Class_num_sufixo'))

# Exibindo a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, center=0)
plt.title('Matriz de Correlação das Variáveis Numéricas com a Classe')
plt.tight_layout()
plt.show()

#Proporção de duplicados Classe X Categoria
for col in categorical_columns:
    print(f"Proporção de duplicados para a variável {var_names.get(col, col)}:")
    crosstab_result = pd.crosstab(df[col], df['Class'], normalize='index') * 100
    print(crosstab_result)
    print("----------------------------------------------------")

    