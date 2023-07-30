import numpy as np
import random
import math

class GeneticAlgorithm():
    def __init__(self, tamanho_individuo=30, tamanho_populacao=100, n_filhos=2, tamanho_torneio=5, max_iteracoes=10000, prob_cruzamento=0.9, prob_mutacao=0.4, funcao='ackley'):
        self.tamanho_individuo = tamanho_individuo
        self.tamanho_populacao = tamanho_populacao
        self.n_filhos = n_filhos
        self.tamanho_torneio = tamanho_torneio
        self.max_iteracoes = max_iteracoes
        self.prob_cruzamento = prob_cruzamento
        self.prob_mutacao = prob_mutacao
        self.funcao = funcao

        self.populacao = []
        self.geracoes = []
        self.iteracoes = 0

    def gerar_individuo(self):
        if self.funcao == 'ackley':
            individuo = np.array([random.uniform(-32.768, 32.768) for i in range(self.tamanho_individuo)])
        elif self.funcao == 'rastrigin':
            individuo = np.array([random.uniform(-5.12, 5.12) for i in range(self.tamanho_individuo)])
        elif self.funcao == 'schwefel':
            individuo = np.array([random.uniform(-500, 500) for i in range(self.tamanho_individuo)])
        elif self.funcao == 'rosenbrock':
            individuo = np.array([random.uniform(-5, 10) for i in range(self.tamanho_individuo)])
        else:
            raise ValueError('Função inválida.')
        return individuo.copy()

    def calcular_fitness(self, individuo):
        if self.funcao == 'ackley':
            termo_1 = -20 * np.exp(-0.2*np.sqrt(np.sum(individuo**2)) / self.tamanho_individuo)
            termo_2 = np.exp(np.sum(np.cos(individuo*2*np.pi)) / self.tamanho_individuo)
            fitness = termo_1 - termo_2 + 20 + np.exp(1)
        elif self.funcao == 'rastrigin':
            fitness = 10*self.tamanho_individuo + np.sum(individuo**2 - 10*np.cos(individuo*2*np.pi))
        elif self.funcao == 'schwefel':
            fitness = 418.9829*self.tamanho_individuo - np.sum(individuo * np.sin(np.sqrt(np.abs(individuo))))
        elif self.funcao == 'rosenbrock':
            fitness = 0
            for i in range(self.tamanho_individuo - 1):
                fitness += 100*(individuo[i + 1] - individuo[i]**2)**2 + (individuo[i] - 1)**2
        return fitness

    def selecionar_sobreviventes(self, filhos):
        populacao_total = self.populacao + filhos
        populacao_ordenada = sorted(populacao_total, key=self.calcular_fitness)
        return populacao_ordenada[:self.tamanho_populacao]

    def mutacao(self, individuo):
        for i in range(self.tamanho_individuo):
            if random.random() < self.prob_mutacao:
                if self.funcao == 'ackley':
                    individuo[i] = max(min(individuo[i] * np.random.normal(0, 5), 32.768), -32.768)
                elif self.funcao == 'rastrigin':
                    individuo[i] = max(min(individuo[i] * np.random.normal(0, 1), 5.12), -5.12)
                elif self.funcao == 'schwefel':
                    individuo[i] = max(min(individuo[i] * np.random.normal(0, 75), 500), -500)
                elif self.funcao == 'rosenbrock':
                    individuo[i] = max(min(individuo[i] * np.random.normal(0, 1), 10), -5)
        return individuo

    def cruzamento(self, pai_1, pai_2):
        if random.random() < self.prob_cruzamento:
            ponto_corte = random.randint(1, self.tamanho_individuo - 1)
            
            filho_1 = np.concatenate((pai_1[:ponto_corte], pai_2[ponto_corte:]))
            filho_2 = np.concatenate((pai_2[:ponto_corte], pai_1[ponto_corte:]))

            return filho_1.copy(), filho_2.copy()
        else:
            return pai_1.copy(), pai_2.copy()

    def selecionar_pais(self):
        candidatos = random.sample(self.populacao, self.tamanho_torneio)
        
        candidatos = sorted(candidatos, key=self.calcular_fitness)
        return candidatos[:2]

    def gerar_populacao_inicial(self):
        return [self.gerar_individuo() for _ in range(self.tamanho_populacao)]
    
    def fit(self, verbose=True):
        self.populacao = self.gerar_populacao_inicial()

        melhor_individuo = max(self.populacao, key=self.calcular_fitness)
        melhor_fitness = self.calcular_fitness(melhor_individuo)

        geracao = dict()
        geracao['populacao'] = [individuo.copy() for individuo in self.populacao]
        geracao['fitness'] = [self.calcular_fitness(individuo) for individuo in self.populacao]
        geracao['melhor_individuo'] = melhor_individuo
        geracao['melhor_fitness'] = melhor_fitness
        geracao['fitness_medio'] = np.mean(geracao['fitness'])

        self.geracoes = []
        self.geracoes.append(geracao.copy())

        self.iteracoes = 0
        while self.iteracoes < self.max_iteracoes and melhor_fitness > 0:
            filhos = []
            for _ in range(self.n_filhos // 2):
                pai_1, pai_2 = self.selecionar_pais()
                filho_1, filho_2 = self.cruzamento(pai_1, pai_2)
                filho_1 = self.mutacao(filho_1)
                filho_2 = self.mutacao(filho_2)
                filhos.extend([filho_1, filho_2])

            self.populacao = self.selecionar_sobreviventes(filhos)

            melhor_individuo = max(self.populacao, key=self.calcular_fitness)
            melhor_fitness = self.calcular_fitness(melhor_individuo)
            geracao = dict()
            geracao['populacao'] = [individuo.copy() for individuo in self.populacao]
            geracao['fitness'] = [self.calcular_fitness(individuo) for individuo in self.populacao]
            geracao['melhor_individuo'] = melhor_individuo
            geracao['melhor_fitness'] = melhor_fitness
            geracao['fitness_medio'] = np.mean(geracao['fitness'])
            self.geracoes.append(geracao.copy())

            self.iteracoes += 1
        
        if verbose:
            if melhor_fitness == 0:
                print(f'Uma solução ótima foi encontrada após {self.iteracoes} iterações.\nSolução ótima: {melhor_individuo}\nFitness da solução ótima: {melhor_fitness}')
            else:
                print(f'Não foi possível encontrar uma solução ótima em {self.iteracoes} iterações.\nMelhor solução: {melhor_individuo}\nFitness da melhor solução: {melhor_fitness}')

    def run_experiments(self, num_experiments, verbose=True):
        experimentos = []
        for i in range(num_experiments):
            resultados_experimento = dict()
            if verbose:
                print(f'[EXPERIMENTO {i}]')
                print()
            self.fit(verbose=verbose)
            
            resultados_experimento['n_experimento'] = i
            resultados_experimento['n_iteracoes'] = self.iteracoes
            resultados_experimento['n_convergencias'] = self.geracoes[-1]['fitness'].count(1)
            resultados_experimento['fitness_medio_ultima_geracao'] = self.geracoes[-1]['fitness_medio']
            resultados_experimento['melhor_individuo'] = self.geracoes[-1]['melhor_individuo']
            resultados_experimento['melhor_fitness'] = self.geracoes[-1]['melhor_fitness']
            resultados_experimento['fitness_medio_geracao'] = [geracao['fitness_medio'] for geracao in self.geracoes]
            resultados_experimento['melhor_fitness_geracao'] = [geracao['melhor_fitness'] for geracao in self.geracoes]
            
            experimentos.append(resultados_experimento)
            if verbose:
                print('='*50)
        return experimentos
        