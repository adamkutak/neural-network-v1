import graph
import nnF
import math
import random
import pygame

def generatePopulation(size,inLabels,outLabels,neuronList):
    nnList = []
    for s in range(size):
        name = str(s)
        nnList.append(nnF.generateNeuralNetwork(name,len(inLabels),inLabels,len(outLabels),outLabels,neuronList))
    return nnList

def reproduce(nnList,i,MUTATION_STD,size,inLabels,outLabels,neuronList):
    alpha = nnList[i]
    nnList = []
    for s in range(size):
        name = str(s)
        temp = nnF.generateNeuralNetwork(name,len(inLabels),inLabels,len(outLabels),outLabels,neuronList)
        for e in range(len(alpha.edges)):
            temp.edges[e].weight = alpha.edges[e].weight + random.gauss(0,MUTATION_STD)
        nnList.append(temp)
    return nnList

def findAlphaIndex(nnList,input,y):
    costList = []
    for s in range(len(nnList)):
        a,maxI = nnList[s].calculate(input)
        costList.append(nnList[s].getQuadraticCost(a,y))
    minI = min(costList)
    return minI,costList.index(minI)

def evolution(GENERATIONS,SIZE,MUTATION_STD,inLabels,outLabels,neuronList,input,output):
    nnList = generatePopulation(SIZE,inLabels,outLabels,neuronList)
    for x in range(GENERATIONS):
        minI,i = findAlphaIndex(nnList,input,output)
        if(x%100==0):
            print("generation ", x, " cost: ",minI)
        nnList = reproduce(nnList,i,MUTATION_STD,SIZE,inLabels,outLabels,neuronList)

#example test of input = [1,0], output = [ 0,1]
SIZE = 1000
MUTATION_STD = 0.01
GENERATIONS = 10000

input = [1,0]
output = [0,1]
inLabels = ['A','B']
outLabels = ['C','D']
neuronList = [2]
#evolution(GENERATIONS,SIZE,MUTATION_STD,inLabels,outLabels,neuronList,input,output)

pygame.init()
nnList = generatePopulation(SIZE,inLabels,outLabels,neuronList)
screen_length = 2550
screen_width = 1000
screen = pygame.display.set_mode([screen_length,screen_width])

running = True
while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running = False

    screen.fill((255,255,255))

    #course surfaces
    surf1 = pygame.Surface()
    pygame.display.flip()
