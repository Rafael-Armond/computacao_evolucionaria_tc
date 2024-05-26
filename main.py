import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

"""
Estrutura do Fenotipo: [[],[],[],...,[]] - Matriz de zeros representando o centro de convencoes, onde houver 1 sera o PA.
Estrutura do Genotipo: [(x1, y1), (x2, y2), ..., (xn, yn)]
"""

"""
Carregando os dados do problema
"""
df_clientes = pd.read_csv('clientes.csv', header=None, names=['x', 'y', 'bandwidth'])

'''
Mostra a distribuição dos clientes
'''
plt.figure(figsize=(8,8))
plt.scatter(df_clientes['x'], df_clientes['y'], alpha=0.8)
plt.title('Distribuição Espacial dos Clientes')
plt.xlabel('Posição X')
plt.ylabel('Posição Y')
plt.grid(True, which='major', linestyle='-', linewidth=0.5)
plt.axis([0, 400, 0, 400])
plt.xticks(range(0, 401, 25))
plt.yticks(range(0, 401, 25))
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.5) 
plt.show()
plt.close()

class Solution:
    def __init__(self, somatorio, media, maximo):
        self.somatorio = somatorio
        self.media = media
        self.maximo = maximo
    
    def mostrarDados(self):
        print(f"Sum: {self.somatorio}\nAverage: {self.media}\nMax: {self.maximo}")

class Individuo:
    def __init__(self, fenotipo, atribuicoes = None):
        self.fenotipo = fenotipo
        self.genotipo = genotipo
        self.fitness = fitness(self.genotipo)
        self.atribuicoes = atribuicoes if atribuicoes is not None else gerarAtribuicoes(self.genotipo)

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
    genotipo = np.zeros((400, 400), dtype=int)
 
    for coord in fenotipo:
        genotipo[coord[0], coord[1]] = 1

    return genotipo

def genotipoToFenotipo(genotipo):
    fenotipo = []
    
    for index_linha, linha in enumerate(genotipo):
        for index_coluna, valor in enumerate(linha):
            if valor == 1: 
                fenotipo.append((index_linha, index_coluna))
                
    return fenotipo

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

"""
Gera fenotipo inicial
"""
def getInitialFenotipo():
    fenotipo = np.zeros((400,400))
    fenotipo_coordenadas = []
    atribuicoes = []

    # Distribuir PAs aleatoriamente na malha de 400x400 metros
    grid_points_x = np.arange(0, 400 + 1, 5)
    grid_points_y = np.arange(0, 400 + 1, 5)
    for _ in range(max_pas):
        x = np.random.choice(grid_points_x)
        y = np.random.choice(grid_points_y)
        fenotipo[x][y] = 1
        fenotipo_coordenadas.append((x, y))

    # Encontrar o PA mais próximo que pode acomodar o cliente sem exceder a capacidade
    pa_largura_banda_usada = {i: 0 for i in range(max_pas)}
    for _, cliente in df_clientes.iterrows():
        atribuido = False
        distancias = [np.sqrt((pa[0] - cliente['x'])**2 + (pa[1] - cliente['y'])**2) for pa in fenotipo_coordenadas]
        possiveis_pas = sorted(range(distancias), key=lambda k: distancias[k])

        for pa_index in possiveis_pas:
            if (distancias[pa_index] <= max_distancia and 
                pa_largura_banda_usada[pa_index] + cliente['bandwidth'] <= max_capacidade_pa):
                atribuicoes.append(pa_index)
                pa_largura_banda_usada[pa_index] += cliente['bandwidth']
                atribuido = True
                break
            
        if not atribuido:
            atribuicoes.append(possiveis_pas[0])

    return fenotipo, atribuicoes

"""
Faz a atribuição de PA's a clientes
"""
def gerarAtribuicoes(fenotipo):
    atribuicoes = []
    fenotipo_coordenadas = []

    for i in range(fenotipo.shape[0]):
        for j in range(fenotipo.shape[1]):
            if (fenotipo[i][j] == 1):
                fenotipo_coordenadas.append((i, j))

    pa_largura_banda_usada = {i: 0 for i in range(max_pas)}
    for _, cliente in df_clientes.iterrows():
        atribuido = False
        distancias = [np.sqrt((pa[0] - cliente['x'])**2 + (pa[1] - cliente['y'])**2) for pa in fenotipo_coordenadas]
        possiveis_pas = sorted(range(distancias), key=lambda k: distancias[k])

        for pa_index in possiveis_pas:
            if (distancias[pa_index] <= max_distancia and 
                pa_largura_banda_usada[pa_index] + cliente['bandwidth'] <= max_capacidade_pa):
                atribuicoes.append(pa_index)
                pa_largura_banda_usada[pa_index] += cliente['bandwidth']
                atribuido = True
                break
            
        if not atribuido:
            atribuicoes.append(possiveis_pas[0])

    return atribuicoes

"""
Definições iniciais
"""
geracoes = 1000
tam_populacao = 4
taxa_mutacao = 0.05
max_pas = 30
max_distancia = 85 # Máxima distância entre um PA e o cliente servido por esse PA
max_capacidade_pa = 54
populacao_inicial = []
for x in range(tam_populacao):
    fenotipo, atribuicoes = getInitialFenotipo()
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

# Visualização dá última melhor solução da população
print(f"Solução final:\n {getMelhorIndividuo(populacao).fenotipo}")