import numpy as np
import random 
import matplotlib.pyplot as plt

class Solution:
    def __init__(self, somatorio, media, maximo):
        self.somatorio = somatorio
        self.media = media
        self.maximo = maximo
    
    def mostrarDados(self):
        print(f"Sum: {self.somatorio}\nAverage: {self.media}\nMax: {self.maximo}")

class Individuo:
    def __init__(self, fenotipo, genotipo):
        self.fenotipo = fenotipo
        self.genotipo = genotipo
        self.fitness = fitness(self.genotipo)

def fitness(genotipo):
    pass

def crossover(ind1, ind2):
    crossover_point = np.random.randint(0, len(ind1.genotipo))
    
    filho1_genotipo = np.concatenate([ind1.genotipo[:crossover_point], ind2.genotipo[crossover_point:]])
    filho2_genotipo = np.concatenate([ind2.genotipo[:crossover_point], ind1.genotipo[crossover_point:]])
    
    filho1_fenotipo = genotipoToFenotipo(filho1_genotipo)
    filho2_fenotipo = genotipoToFenotipo(filho2_genotipo)
    
    filho1 = Individuo(filho1_fenotipo, filho1_genotipo)
    filho2 = Individuo(filho2_fenotipo, filho2_genotipo)
    
    return filho1, filho2

def mutacao(individuo, taxa_mutacao):
    if (random.random() <= taxa_mutacao):
        pos = random.randint(0, len(individuo.genotipo) - 1)
        valor = random.randint(0, len(individuo.genotipo) - 1)
        individuo.genotipo[pos] = valor
        individuo.fenotipo = genotipoToFenotipo(individuo.genotipo)
        individuo.fitness = fitness(individuo.genotipo)

def selecaoPorTorneio(populacao, k=3):
    torneio = random.sample(populacao, k)
    melhor = max(torneio, key=lambda ind: ind.fitness)
    return melhor

def fenotipoToGenotipo(fenotipo):
    pass

def genotipoToFenotipo(genotipo):
    pass

"""
Seleciona o melhor indivíduo da população
"""
def getMelhorIndividuo(populacao):
    melhor_fitness = 0
    melhor_individuo = None
    
    for individuo in populacao:
        if (individuo.fitness > melhor_fitness):
            melhor_individuo = individuo
            
    return melhor_individuo

def getInitialFenotipo():
    pass

def fenotipoToGenotipo(fenotipo):
    pass

"""
Definições iniciais
"""
geracoes = 1000
tam_populacao = 4
taxa_mutacao = 0.05
populacao_inicial = []
for x in range(tam_populacao):
    fenotipo = getInitialFenotipo()
    genotipo = fenotipoToGenotipo(fenotipo)
    populacao_inicial.append(Individuo(fenotipo, genotipo))

"""
Avaliação inicial
"""
solutions = []
sol_inicial = Solution(sum(ind.fitness for ind in populacao_inicial), 0, 0)
sol_inicial.media = sol_inicial.somatorio / len(populacao_inicial)
sol_inicial.maximo = max(ind.fitness for ind in populacao_inicial)
solutions.append(sol_inicial)

"""
Evolução
"""
populacao = populacao_inicial[:]
for _ in range(geracoes):
    nova_populacao = []
    while len(nova_populacao) < len(populacao):
        # Seleção de pais
        pai1 = selecaoPorTorneio(populacao)
        pai2 = selecaoPorTorneio(populacao)
        
        # Crossover entre os pais
        filho1, filho2 = crossover(pai1, pai2)
        
        # Verifica e aplica a mutação quando for o caso
        mutacao(filho1, taxa_mutacao)
        mutacao(filho2, taxa_mutacao)
       
        # Acrescenta os dois novos indivíduos a nova população
        nova_populacao.extend([filho1, filho2])
        
    populacao = nova_populacao
    solucao = Solution
    solucao = Solution(sum(ind.fitness for ind in populacao), 0, max(ind.fitness for ind in populacao))
    solucao.media = solucao.somatorio / len(populacao)
    solutions.append(solucao)    

# Visualização evolutiva dos resultados
somatorios = [sol.somatorio for sol in solutions]
plt.figure(figsize=(10, 5))
plt.plot(somatorios, marker='o')
plt.title('Evolução do Somatório ao longo das Gerações')
plt.xlabel('Geração')
plt.ylabel('Somatório')
plt.grid(True)
plt.show()

# Visualização dá última solução
print(f"Solução final:\n {getMelhorIndividuo(populacao).fenotipo}")