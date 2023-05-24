# -*- coding: utf-8 -*-

#%% Análise de Cluster

# Instalando os pacotes

# Digitar o seguinte comando no console: pip install -r requirements.txt

# Importando os pacotes

import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import scipy.stats as stats
from sklearn.cluster import KMeans
import seaborn as sns
import numpy as np

#%% Importando os dados

## Fonte: Fávero & Belfiore (2017, Capítulo 9)

dados_vest = pd.read_excel('Vestibular.xlsx')

#%% Visualizando os dados e cada uma das variáveis

print(dados_vest, "\n")

print(dados_vest.info())

#%% ZScore

# Selecionado apenas variáveis métricas e realizando o ZScore

vest = dados_vest.drop(columns=['estudante'])
                       
# Muitas vezes, é importante realizar o procedimento Z-Score nas variáveis
# Quando as variáveis estiverem em unidades de medidas distintas
# Poderia ser feito da seguinte forma, embora aqui não utilizaremos
                
# for coluna in vest.columns:
    # vest[coluna] = stats.zscore(vest[coluna])

# Neste caso, vamos utilizar as variáveis originais
# Todas são notas de 0 a 10

print(vest)

#%% Cluster Hierárquico Aglomerativo

# Gerando o dendrograma 

## Inicialmente, vamos utilizar: 
## Distância euclidiana e método de encadeamento single linkgage

plt.figure(figsize=(16,8))

dendrogram = sch.dendrogram(sch.linkage(vest, method = 'single', metric = 'euclidean'), labels = list(dados_vest.estudante))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Pessoas', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 4.5, color = 'red', linestyle = '--')
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

#%% Gerando a variável com a indicação do cluster no dataset

## Deve ser realizada a parametrização:
    ## Número de clusters
    ## Medida de distância
    ## Método de encadeamento
    
## Como já observamos 3 clusters pelo dendrograma, vamos selecionar "3" clusters
## A medida de distância e o método de encadeamento são mantidos

cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(vest)

# Retorna uma lista de valores com o cluster de cada observação

print(indica_cluster_sing, "\n")

dados_vest['cluster_single'] = indica_cluster_sing

print(dados_vest)

#%% Comparando clusters resultantes por diferentes métodos de encadeamento

# Complete linkage

cluster_comp = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(vest)

print(indica_cluster_comp, "\n")

dados_vest['cluster_complete'] = indica_cluster_comp

print(dados_vest)

#%% Comparando clusters resultantes por diferentes métodos de encadeamento

# Average linkage

cluster_avg = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')
indica_cluster_avg = cluster_avg.fit_predict(vest)

print(indica_cluster_avg, "\n")

dados_vest['cluster_average'] = indica_cluster_avg

print(dados_vest)

## Inclusive, poderiam ser alteradas as medidas de distância também

#%% Cluster Não Hierárquico K-means

# Considerando que identificamos 3 possíveis clusters na análise hierárquica

kmeans = KMeans(n_clusters = 3, init = 'random').fit(vest)

#%% Para identificarmos os clusters gerados

kmeans_clusters = kmeans.labels_

print(kmeans_clusters)

dados_vest['cluster_kmeans'] = kmeans_clusters

print(dados_vest)

#%% Identificando as coordenadas centróides dos clusters finais

cent_finais = pd.DataFrame(kmeans.cluster_centers_)
cent_finais.columns = vest.columns
cent_finais.index.name = 'cluster'
cent_finais

#%% Analisando o banco de dados

vest

#%% Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

pred_y = kmeans.fit_predict(vest)
sns.scatterplot(x='matemática', y='física', data=dados_vest, hue='cluster_kmeans', palette='viridis', s=60)
plt.scatter(cent_finais['matemática'], cent_finais['física'], s = 40, c = 'red', label = 'Centróides', marker="X")
plt.title('Clusters e centróides', fontsize=16)
plt.xlabel('Matemática', fontsize=16)
plt.ylabel('Física', fontsize=16)
plt.legend()
plt.show()

#%% Identificação da quantidade de clusters

# Método Elbow para identificação do nº de clusters
## Elaborado com base na "inércia": distância de cada obervação para o centróide de seu cluster
## Quanto mais próximos entre si e do centróide, menor a inércia

inercias = []
K = range(1,vest.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(vest)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, inercias, 'bx-')
plt.axhline(y = 20, color = 'red', linestyle = '--')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Inércias', fontsize=16)
plt.title('Método do Elbow', fontsize=16)
plt.show()

# Normalmente, busca-se o "cotovelo", ou seja, o ponto onde a curva "dobra"

#%% Estatística F

# Análise de variância de um fator:
# As variáveis que mais contribuem para a formação de pelo menos um dos clusters

def teste_f_kmeans(kmeans, dataframe):
    
    variaveis = dataframe.columns

    centroides = pd.DataFrame(kmeans.cluster_centers_)
    centroides.columns = dataframe.columns
    centroides
    
    print("Centróides: \n", centroides ,"\n")

    df = dataframe[variaveis]

    unique, counts = np.unique(kmeans.labels_, return_counts=True)

    dic = dict(zip(unique, counts))

    qnt_clusters = kmeans.n_clusters

    observacoes = len(kmeans.labels_)

    df['cluster'] = kmeans.labels_

    output = []

    for variavel in variaveis:

        dic_var={'variavel':variavel}

        # variabilidade entre os grupos

        variabilidade_entre_grupos = np.sum([dic[index]*np.square(observacao - df[variavel].mean()) for index, observacao in enumerate(centroides[variavel])])/(qnt_clusters - 1)

        dic_var['variabilidade_entre_grupos'] = variabilidade_entre_grupos

        variabilidade_dentro_dos_grupos = 0

        for grupo in unique:

            grupo = df[df.cluster == grupo]

            variabilidade_dentro_dos_grupos += np.sum([np.square(observacao - grupo[variavel].mean()) for observacao in grupo[variavel]])/(observacoes - qnt_clusters)

        dic_var['variabilidade_dentro_dos_grupos'] = variabilidade_dentro_dos_grupos

        dic_var['F'] =  dic_var['variabilidade_entre_grupos']/dic_var['variabilidade_dentro_dos_grupos']
        
        dic_var['sig F'] =  1 - stats.f.cdf(dic_var['F'], qnt_clusters - 1, observacoes - qnt_clusters)

        output.append(dic_var)

    df = pd.DataFrame(output)
    
    print(df)

    return df

# Os valores da estatística F são bastante sensíveis ao tamanho da amostra

output = teste_f_kmeans(kmeans,vest)

#%% Gráfico 3D dos clusters

import plotly.express as px 
import plotly.io as pio

pio.renderers.default='browser'

fig = px.scatter_3d(dados_vest, 
                    x='matemática', 
                    y='química', 
                    z='física',
                    color='cluster_kmeans')
fig.show()

#%% Análise de Cluster: Exemplo 2

# Importando os dados
# Fonte: Fávero & Belfiore (2017, Capítulo 9)

dados_varejista = pd.read_excel('Regional Varejista (Cluster).xlsx')

#%% Visualizando os dados

print(dados_varejista, "\n")

print(dados_varejista.info(), "\n")

#%% Visualizando as estatítiscas univariadas

print(dados_varejista[['atendimento','sortimento', 'organização']].describe())

#%% Ajustando o banco de dados

# Retirando todos os dados que não são numéricos do dataset

varejista = dados_varejista.drop(columns=['loja','regional'])

print(varejista)

#%% Cluster Hierárquico Aglomerativo

# Gerando o dendrograma

plt.figure(figsize=(16,8))
dendrogram = sch.dendrogram(sch.linkage(varejista, method = 'complete', metric = 'euclidean'))
plt.axhline(y = 60, color = 'red', linestyle = '--')
plt.title('Dendrograma')
plt.xticks([]) # Procedimento para retirar os pontos do eixo x quando existirem muitas observações
plt.ylabel('Distância Euclidiana')
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

#%% Clusters gerados pelo método de encadeamento 'single linkage'

cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(varejista)

print(indica_cluster_sing)

dados_varejista['cluster_single'] = indica_cluster_sing
dados_varejista['cluster_single'] = dados_varejista['cluster_single'].astype('category')

print(dados_varejista)

#%% Clusters gerados pelo método de encadeamento 'average linkage'

cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'average')
indica_cluster_sing = cluster_sing.fit_predict(varejista)

print(indica_cluster_sing)

dados_varejista['cluster_average'] = indica_cluster_sing
dados_varejista['cluster_average'] = dados_varejista['cluster_average'].astype('category')

print(dados_varejista)

#%% Clusters gerados pelo método de encadeamento 'complete linkage'

cluster_sing = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'complete')
indica_cluster_sing = cluster_sing.fit_predict(varejista)

print(indica_cluster_sing)

dados_varejista['cluster_complete'] = indica_cluster_sing
dados_varejista['cluster_complete'] = dados_varejista['cluster_complete'].astype('category')

print(dados_varejista)

#%% Plotando as observações e seus clusters (single)

plt.figure(figsize=(10,10))

fig = sns.scatterplot(x='atendimento', y='sortimento', s=60, data=dados_varejista, hue='cluster_single')
plt.title('Clusters', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.show()

#%% Método K-Means

# Considerando que identificamos 3 possíveis clusters na análise hierárquica

kmeans_varejista = KMeans(n_clusters = 3, init = 'random').fit(varejista)

#%% Para identificarmos os clusters gerados

kmeans_clusters = kmeans_varejista.labels_

print(kmeans_clusters)

dados_varejista['cluster_kmeans'] = kmeans_clusters

#%% Coordenadas dos centróides dos clusters finais

cent_finais = pd.DataFrame(kmeans_varejista.cluster_centers_)
cent_finais.columns = varejista.columns
cent_finais.index.name = 'cluster'
cent_finais

#%% Plotando as observações e seus centróides dos clusters

plt.figure(figsize=(10,10))

sns.scatterplot(x='atendimento', y='sortimento', data=dados_varejista, hue='cluster_kmeans', palette='viridis', s=60)
plt.scatter(cent_finais['atendimento'], cent_finais['sortimento'], s = 40, c = 'red', label = 'Centróides', marker="X")
plt.title('Clusters e centróides', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.legend()
plt.show()

#%% Método Elbow para identificação do nº de clusters

## Obs.: Quanto mais próximos entre si e do centróide, menor a inércia

inercias = []
K = range(1,varejista.shape[0])
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(varejista)
    inercias.append(kmeanModel.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, inercias, 'bx-')
plt.axhline(y = 4000, color = 'red', linestyle = '--')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Inércias', fontsize=16)
plt.title('Método do Elbow', fontsize=16)
plt.show()

#%% Estatística F das variáveis

teste_f_kmeans(kmeans_varejista,varejista)

#%% Gráfico 3D dos clusters

import plotly.express as px 
import plotly.io as pio

pio.renderers.default='browser'

fig = px.scatter_3d(dados_varejista, 
                    x='atendimento', 
                    y='sortimento', 
                    z='organização',
                    color='cluster_kmeans')
fig.show()

#%% FIM