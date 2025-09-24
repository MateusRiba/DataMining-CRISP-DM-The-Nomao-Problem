from scipy.io import arff
import pandas as pd
from sklearn.model_selection import train_test_split
import os

processed_path = "C:\\Users\\mateu\\Arquivos de Programas Faculdade\\repositorios\\DataMining-CRISP-DM-The-Nomao-Problem\\data\\processed"
arff_path = "C:\\Users\\mateu\\Arquivos de Programas Faculdade\\repositorios\\DataMining-CRISP-DM-The-Nomao-Problem\\data\\raw\\phpDYCOet.arff"

#Carregando arquivo
data, meta = arff.loadarff(arff_path)

#Conversão para DataFrame do pandas
df = pd.DataFrame(data)

#Exibindo as primeiras linhas e os nomes das colunas do DataFrame
print(df.columns)
print(df.head())

#Coluna alvo é a coluna 'Class'
X = df.drop('Class', axis=1) #Dados de entrada
y = df['Class'] #Variavel alvo

#Divisção dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Visualização das dimensões dos conjuntos
print(f"Dimensões do conjunto de treino: X: {X_train.shape}, y: {y_train.shape}")
print(f"Dimensões do conjunto de teste: X: {X_test.shape}, y: {y_test.shape}")

#--------------------------------------
#Verificação de balanceamento das classes

train_class_distribution = y_train.value_counts(normalize=True)
test_class_distribution = y_test.value_counts(normalize=True)

print("Distribuição das classes no conjunto de treino:")
print(train_class_distribution)

print("\nDistribuição das classes no conjunto de teste:")
print(test_class_distribution)

#--------------------------------------