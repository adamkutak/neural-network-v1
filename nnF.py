import graph
import math
import statistics
import random
import pandas as pd
import numpy as np

def generateNeuralNetwork(name,inCount,inLabels,outCount,outLabels,neuronList):
    nn = graph.Graph(name)
    nn.setLayers(len(neuronList))
    vc = 0 #vertex id counter
    ec = 0 #edge id counter
    #adding vertexes
    for x in range(inCount):
        nn.addVertice("input",vc,inLabels[x],0)
        vc += 1
    for l in range(len(neuronList)):
        for r in range(neuronList[l]):
            nn.addVertice("neuron",vc,'n/a',random.gauss(0,0.5),l+1)
            vc += 1
    for y in range(outCount):
        nn.addVertice("output",vc,outLabels[y],0,len(neuronList)+1)
        vc += 1
    #adding edges
    mid = inCount
    left = 0
    right = inCount + neuronList[0]
    for z in range(1,len(neuronList)+2):
        for a in range(left,mid):
            for b in range(mid,right):
                nn.addEdge(ec,a,b,random.gauss(0,0.5))
                ec += 1
        left = mid
        mid = right
        if(len(neuronList) > z):
            right = right + neuronList[z]
        else:
            right = right + outCount
    return nn

def trainNeuralNetwork(nn,inData,outData,algoType):
    if(algoType=='stochastic'):
        return trainNeuralNetwork_stoch(nn,inData,outData)
    else:
        return trainNeuralNetwork_minibatch(nn,inData,outData)

def trainNeuralNetwork_stoch(nn,inData,outData):
    inwidth = len(inData)
    trainingIterations = len(inData[0])
    costSum = []
    for x in range(trainingIterations):
        inLine = []
        for y in range(inwidth):
            inLine.append(inData[y][x])
        outLine = outData[x]
        outVector = nn.teachNetwork(inLine,outLine,"stochastic")
        cost = nn.getQuadraticCost(outVector,outLine)

def trainNeuralNetwork_minibatch(nn,inData,outData):
    inwidth = len(inData)
    trainingIterations = len(inData[0])
    costSum = []
    mb_size = 10
    for x1 in range(trainingIterations//mb_size):
        weightMasterList = []
        biasMasterList = []
        for x2 in range(mb_size):
            inLine = []
            x = x1*mb_size+x2
            for y in range(inwidth):
                inLine.append(inData[y][x])
            outLine = outData[x]
            outVector, weightGradients, biasGradients = nn.teachNetwork(inLine,outLine,"minibatch")
            cost = nn.getQuadraticCost(outVector,outLine)
            if(x2==0):
                weightMasterList = weightGradients
                biasMasterList = biasGradients
            else:
                for wq in range(len(weightMasterList)):
                    weightMasterList[wq][1] += weightGradients[wq][1]
                for iq in range(len(biasMasterList)):
                    biasMasterList[iq][1] += biasGradients[iq][1]

        #update weights and biases of the graph
        for ed in weightMasterList:
            nn.edges[ed[0]].weight = nn.edges[ed[0]].weight - ed[1]/mb_size
        for vert in biasMasterList:
            nn.vertexes[vert[0]].bias = nn.vertexes[vert[0]].bias - vert[1]/mb_size

def testNeuralNetwork(nn,inData,outData,z):
    inwidth = len(inData)
    #outwidth = len(outData)
    testingIterations = len(inData[0])
    aCount = 0
    for x in range(testingIterations):
        inLine = []
        for y in range(inwidth):
            inLine.append(inData[y][x])
        outLine = outData[x]
        outVector,maxI = nn.calculate(inLine)
        if(outVector.index(max(outVector))==outLine.index(max(outLine))):
            aCount += 1

    accuracy = 100*aCount/(testingIterations)
    print("epoch ",z," testing accuracy: ", round(accuracy,5),"%")
    return accuracy

def loadData(filename,inputCount,genDefaults=False):
    if(genDefaults):
        inData = pd.read_csv(filename).T.reset_index().values.tolist()
        outCount = len(inData)-inputCount
        inLabels = []
        outLabels = []
        for x in range(0,inputCount):
            inLabels.append(inData[x].pop(0))
        for y in range(inputCount,len(inData)):
            outLabels.append(inData[y].pop(0))
        outData = []
        for z in range(len(inData)-1,inputCount-1,-1):
            outData.insert(0,inData.pop(z))

        return inData,outData,inLabels,outLabels,outCount

    else:
        df = pd.read_csv(filename).sample(frac=1)
        inData = df.values.T.tolist()
        outData = []
        # if(len(inData)-inputCount==1):
        #     outData = [[inData.pop(-1)]]
        #     return inData,outData
        for z in range(len(inData)-1,inputCount-1,-1):
            outData.insert(0,inData.pop(z))
        return inData,outData

def gnnAuto(name,inputCount,filename,neurons = []):
    if(len(neurons)==0):
        neurons = [int(inputCount*0.76),int(inputCount*0.76)]
    inData,outData,inLabels,outLabels,outCount = loadData(filename,inputCount,genDefaults=True)
    nn = generateNeuralNetwork(name,inputCount,inLabels,outCount,outLabels,neurons)
    return nn,inData,outData

def outDataBinaryConverter(outData,outTypes):
    outData = outData[0]
    newOutData = []
    for x in range(len(outData)):
        ind = 0
        for i in range(len(outTypes)):
            if(outTypes[i] in str(outData[x])):
                ind = i
        newOutData.append([0]*len(outTypes))
        newOutData[x][ind] = 1
    return newOutData

def normalizeInput(inData):
    for x in range(len(inData)):
        std = statistics.pstdev(inData[x])
        mean = statistics.mean(inData[x])
        if(std!=0):
            for y in range(len(inData[x])):
                inData[x][y] = (inData[x][y] - mean)/std
    return inData
