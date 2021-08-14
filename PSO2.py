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
        xs= np.random.uniform(min(GABARITO), max(GABARITO), self.variveis).round(2)
        for c in range(1, self.variveis+1):
            dic[c]=xs[c-1]
            dic[f'best{c}'] = 0 
        dic['best']=100000
        return dic

    def apply_function(self, population):
        population['result']=0
        for n in range(self.variveis):
            population['result'] += abs(population.iloc[:,n] - GABARITO[n])

        return population
    
    def to_df(self, population, df=False):
        if type(df)==bool:
            dataset = pd.DataFrame([])
        else:
            dataset = df
        for item in population:
            dataset=dataset.append(item, ignore_index=True)
        return dataset
    
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
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        self.BEST = population.iloc[0,:]
        population = self.refesh_bests(population)
        return population
    

    def run(self):
        population = [self.generate_init() for c in range(self.population)]
        population = self.to_df(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        self.BEST = population.iloc[0,:]
        population = self.refesh_bests(population)

        for iteration in range(self.iterations):
            population = self.refresh(population)

            self.validadores['min'].append(population['result'][:MEAN_BEST].min())
            self.validadores['mean'].append(population['result'][:MEAN_BEST].mean())
            self.validadores['std'].append(population['result'][:MEAN_BEST].std())
        
            if  population['result'][0]<0.03:
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


GABARITO = [52.547,72.154, 53.694, 57.771, 115.88, 105.59, 75.368, 126.02, 52.756, 85.100, 80.525, 111.24, 113.62, 64.95, 89.181, 85.647,
            101.71, 106.75, 110.37, 72.082, 104.38, 102.41, 63.009, 59.52, 89.869, 126.78, 77.231, 96.821, 67.905, 110.1]  

rw = Randomwalk(12, 100, 30, .8, .3)
population=rw.run()
rw.plot_conv()
