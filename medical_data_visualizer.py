import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Carregar o arquivo CSV que contém os dados médicos
df = pd.read_csv('medical_examination.csv')

# Calcular o índice de massa corporal (IMC) usando a fórmula: peso (kg) / altura (m)^2
bmi = df['weight'] / (df['height'] / 100) ** 2

# Adicionar uma nova coluna 'overweight' indicando se a pessoa está acima do peso (IMC > 25)
df['overweight'] = (bmi > 25).astype(int)

# Normalizar os dados de colesterol e glicose:
# Valores 1 são considerados normais (substituídos por 0), e valores maiores que 1 indicam anomalias (substituídos por 1)
df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']] > 1).astype(int)

# Função para desenhar o gráfico categórico
def draw_cat_plot():
    # Reformatação dos dados usando `pd.melt` para agrupar as variáveis de interesse com a coluna 'cardio'
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Desenhar um gráfico de barras que mostra a contagem de cada categoria, separando os dados por 'cardio'
    # e destacando as diferentes categorias ('value')
    g = sns.catplot(x='variable', hue='value', col='cardio', kind='count', data=df_cat)
    
    # Ajustar o rótulo do eixo y para 'total'
    g.set_ylabels('total')

    # Como `catplot()` retorna um objeto FacetGrid, vamos pegar a figura associada para salvar
    fig = g.fig

    # Salvar o gráfico como uma imagem PNG
    fig.savefig('catplot.png')

    return fig


# Função para desenhar o mapa de calor de correlação
def draw_heat_map():
    # Aplicar filtros nos dados para garantir que eles estão dentro de intervalos razoáveis:
    # - A pressão diastólica (ap_lo) deve ser menor ou igual à pressão sistólica (ap_hi)
    # - Filtrar a altura e o peso para ficar dentro dos 95% percentis (evitando outliers)
    valid_ap = df['ap_lo'] <= df['ap_hi']
    valid_height = df['height'].between(df['height'].quantile(0.025), df['height'].quantile(0.975))
    valid_weight = df['weight'].between(df['weight'].quantile(0.025), df['weight'].quantile(0.975))

    # Criar um DataFrame com os dados filtrados
    df_heat = df[valid_ap & valid_height & valid_weight]

    # Calcular a matriz de correlação entre as variáveis
    corr = df_heat.corr()

    # Criar uma máscara para ocultar a metade superior da matriz de correlação (pois é simétrica)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Definir o tamanho da figura
    fig, ax = plt.subplots(figsize=(11, 9))

    # Desenhar o mapa de calor utilizando Seaborn, com anotações nas células e ajustando os limites do gráfico
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', linewidths=1, cmap='icefire', center=0, vmin=-0.15, vmax=0.3, cbar_kws={'shrink': 0.5})

    # Salvar o mapa de calor como uma imagem PNG
    fig.savefig('heatmap.png')

    return fig
