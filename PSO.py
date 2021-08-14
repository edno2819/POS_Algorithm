import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MEAN_BEST = 5


# 1 - CALCULE A FUNÇÃO FITNESS PARA CADA INDIVIDUO
# 2- ACHE A MELHOR POSSIÇÃO ATÉ O MOMENTO DE CADA PARTICULA = P
# 3- ACHE O MELHOR INDIVIDUO DO BANDO = G
# 4- ATUALIZE A VELOCIDADE DE CADA PARTICULA
#   * DELTA1 = RANDOM (0,1) * CONSTANTE1
#   * DELTA2 = RANDOM (0,1) * CONSTANTE2
#   * VNOVA = (P - PAtual)* DELTA1 + (G - PAtual)* DELTA2
#   * POSICAO NOVA = Posicao_antiga + VNOVA


class Randomwalk:

    def __init__(self, iteracoes, population, variaveis, delta1, delta2):
        self.population = population 
        self.iterations = iteracoes
        self.variveis = variaveis
        self.delta1 = delta1
        self.delta2 = delta2
        self.validadores = {'mean': [], 'std':[], 'min':[]}
        

    def generate_init(self):
        dic={}
        xs=np.random.uniform(0, 1, self.variveis)
        for c in range(1, self.variveis+1):
            dic[c]=xs[c-1]
            dic[f'best{c}'] = 0 
        dic['best']=1000
        return dic
    
    def calculate(self, x):
        return x + np.random.uniform(0,1)

    def apply_function(self, population):
        x1 = population['x1']
        x2 = population['x2']
        x3 = population['x3']
        population['result']=(10*(x1-1)**2)+(20*(x2-2)**2)+(30*(x3-3)**2)
        return population
    
    def to_df(self, population, df=False):
        if type(df)==bool:
            dataset = pd.DataFrame([])
        else:
            dataset = df
        for item in population:
            dataset=dataset.append(item, ignore_index=True)
        return dataset

    def convert(self, population):
        def form(variavel, x):
            dic = {1: (-3, 3), 2: (-2, 4), 3: (0, 6)}
            return dic[variavel][0] + (dic[variavel][1] - dic[variavel][0]) * x

        for variavel in range(1, self.variveis+1):
            population[f'x{variavel}'] = population[variavel].apply(lambda x: form(variavel, x))

        return population
    
    def calculate_new_variable(self, population, column):
        def new_value(row):
            delta1 = np.random.uniform(0, 1) * self.delta1
            delta2 = np.random.uniform(0, 1) * self.delta2
            vnova = ( row[f'best{column}'] - row[column])* delta1 + (self.BEST[column] - row[column])* delta2
            return row[column] + vnova

        population[column] =  population.apply(new_value, axis=1)

        return population

    def sum_step(self, population):
        for column in range(1, self.variveis+1):
            population = self.calculate_new_variable(population, column)
        
        return population
    

    def best_x(self, population, column):
        def max_value(row):
            if row['best']>row[f'result']:
                value = row[column]
                if column == self.variveis:
                    row['best'] = row[f'result']
            else:
                value = row[f'best{column}']
            return value

        population[f'best{column}'] =  population.apply(max_value, axis=1)

        return population
    
    def refesh_bests(self, population):
        for column in range(1, self.variveis+1):
            population = self.best_x(population, column)
        
        return population

    def refresh(self, population):
        population = self.sum_step(population)
        population = self.convert(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        self.BEST = population.iloc[0,:]
        population = self.refesh_bests(population)
        return population
    

    def run(self):
        population = [self.generate_init() for c in range(self.population)]
        population = self.to_df(population)
        population = self.convert(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        self.BEST = population.iloc[0,:]
        population = self.refesh_bests(population)

        for iteration in range(self.iterations):
            population = self.refresh(population)

            self.validadores['min'].append(population['result'][:MEAN_BEST].min())
            self.validadores['mean'].append(population['result'][:MEAN_BEST].mean())
            self.validadores['std'].append(population['result'][:MEAN_BEST].std())
        
            if  population['result'][0]<0.05:
                population.reset_index(inplace=True)        
                return population

        population.reset_index(inplace=True)        
        return population

    def plot_conv(self):
        index=[c+1 for c in range(len(self.validadores['mean']))]
        a=[self.validadores['mean'][c] - self.validadores['std'][c] for c in range(len(index))]
        b=[self.validadores['mean'][c]+self.validadores['std'][c] for c in range(len(index))]
        sns.set_style('dark')
        plt.figure(figsize=(10,6))
        plt.grid()
        plt.plot(index, self.validadores['mean'], label='Mean')
        plt.fill_between(range(len(index)), a, b, alpha=0.5, label='Std. Dev')
        plt.plot(index, self.validadores['min'], label ='Minimum')
        plt.xlabel('Generation')
        plt.xlabel('Fitness Values')
        plt.title('Fitness by Generation')
        plt.legend()
        plt.show()


rw = Randomwalk(10, 100, 3, .5, .5)
population=rw.run()
rw.plot_conv()
