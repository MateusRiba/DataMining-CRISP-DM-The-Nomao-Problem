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
#Sprint 2 - Verificação de valores ausentes

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
#Sprint 3 - Entendimentos de variaveis categóricas

#Identificação de variáveis categóricas
# Observando o dataset, muitas variaveis chamam a atenção, porem será escolhido:
#clean_name_including(V8), City_Including: (V16), Zip_Including: (V24), Street_Including: (V32)
#GeocoderPostalcodenumber_Including: (V80), Phone_Equality: (V93), Coordinates_Long_Equality: (V113), Coordinates_Lat_Equality: (V117)

#Visualizando a tabela apenas com: as colunas categóricas
categorical_columns = ['V8', 'V16', 'V24', 'V32', 'V80', 'V93', 'V113', 'V117']

#Visualização das primeiras linhas das colunas categóricas
print("----------------------------------------------------")
print("Primeiras linhas das colunas categóricas selecionadas:")
print(df[categorical_columns].head())
print("----------------------------------------------------")

#Geração de gráfivos para cada variável categórica
for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col, hue='Class')
    plt.title(f'Distribuição da variável {col} por Classe')
    plt.xticks(rotation=45)
    plt.legend(title='Class', loc='upper right')
    plt.tight_layout()
    plt.show()

