import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random

def sumarizar_experimentos(resultados):
    print(resultados[['n_iteracoes', 'n_convergencias', 'fitness_medio_ultima_geracao', 'melhor_fitness']].apply(['mean', 'std', 'min', 'max']).to_string())
    print(f"Taxa de convergência: {len(resultados.query('melhor_fitness == 1'))*100/30:.0f}%")

def avaliar_convergencia(resultados, n_experimento):
    fitness_df = resultados.explode(['fitness_medio_geracao', 'melhor_fitness_geracao'])[['n_experimento', 'fitness_medio_geracao', 'melhor_fitness_geracao']]
    fitness_df = fitness_df.query(f'n_experimento == {n_experimento}').copy()
    fitness_df['geracao'] = range(len(fitness_df))

    plt.figure(figsize=(20, 8))
    sns.lineplot(data=fitness_df, x='geracao', y='fitness_medio_geracao', color='gray', lw=2, label='Fitness médio')
    sns.lineplot(data=fitness_df, x='geracao', y='melhor_fitness_geracao', color='red', lw=2, label='Melhor indivíduo')
    plt.ylabel('fitness')
    plt.show()

def fitness_medio_experimentos(resultados):
    fitness_df = resultados.explode('fitness_medio_geracao')[['n_experimento', 'fitness_medio_geracao']]
    fitness_df['geracao'] = fitness_df.groupby('n_experimento').fitness_medio_geracao.transform(lambda x: range(len(x)))
    fitness_df['n_experimento'] = fitness_df['n_experimento'].astype(str)

    fitness_medio_df = fitness_df.groupby('geracao').fitness_medio_geracao.mean().reset_index()

    plt.figure(figsize=(20, 8))
    sns.lineplot(data=fitness_df, x='geracao', y='fitness_medio_geracao', hue='n_experimento', palette=['gray']*30, legend=False, lw=2)
    sns.lineplot(data=fitness_medio_df, x='geracao', y='fitness_medio_geracao', color='red', lw=2, label='Média do fitness médio em cada geração')
    plt.show()

def qtd_convergencias_experimentos(resultados):
    plt.figure(figsize=(12, 4))
    sns.countplot(data=resultados, x='n_convergencias', color='royalblue')
    plt.show()

def qtd_iteracoes_experimentos(resultados):
    plt.figure(figsize=(12, 4))
    sns.histplot(resultados, x='n_iteracoes', binwidth=25, color='royalblue')
    plt.show() 

def fitness_medio_final_experimentos(resultados):
    plt.figure(figsize=(12, 4))
    sns.histplot(resultados, x='fitness_medio_ultima_geracao', binwidth=0.01, color='royalblue')
    plt.show() 