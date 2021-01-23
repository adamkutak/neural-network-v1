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

#create your own test
# nn = generateNeuralNetwork("MyFirstNN",2,['size','height'],2,['metal','liquid'],[3])
# nn.saveGraph('savetest.txt')
# nn.loadGraph('save2.txt')
# print(nn.calculate([0.5,0.3]))

#autogenerate test
# nn,inData,outData = gnnAuto("irisClassification",4,"iris_flowers.csv")
# nn.saveGraph("iristest.txt")
# print(nn.calculate([1,1,1,1]))












#THIS SECTION IS FOR ACTUALLY MAKING AND TESTING A NEURAL NETWORK
    #TODO: MOVE THIS GARBAGE OUT OF THIS SECTION AND INTO SEPARATE FILES LOL

#Iris flower classification
trainFile = "iris_flowers.csv"
testFile = "iris_testing.csv"
in_labels = ['sepal length','sepal width','petal length','petal width']
out_labels = ['setosa','versicolor','virginica']
nn = generateNeuralNetwork("IrisClassification",4,in_labels,3,out_labels,[4])


epochs = 100
for x in range(epochs):
    #train network
    inData,outData = loadData(trainFile,len(in_labels))
    outData = outDataBinaryConverter(outData,out_labels)
    trainNeuralNetwork(nn,inData,outData,"stochastic")
    #test network accuracy
    inData,outData = loadData(testFile,len(in_labels))
    outData = outDataBinaryConverter(outData,out_labels)
    testNeuralNetwork(nn,inData,outData,x)

#updated MNIST using CSV (normalized variant)
#UPDATE: THIS DOES NOT WORK RIP
# trainFile = "mnist_train.csv"
# testFile = "mnist_test.csv"
# out_labels = ['0','1','2','3','4','5','6','7','8','9']
# in_labels = []
# for cl in range(28*28):
#     in_labels.append(str(cl+1))
# inData,outData = loadData(trainFile,len(in_labels))
# inData = normalizeInput(inData)
# outData = outDataBinaryConverter(outData,out_labels)
# print(inData[0])
# print(outData[0])
# print('done training data preprocessing')
# #MNIST testing data
# inTest,outTest = loadData(testFile,len(in_labels))
# inTest = normalizeInput(inTest)
# outTest = outDataBinaryConverter(outTest,out_labels)
# inTestSample = []
# outTestSample = outData[:100]
# for y in range(len(inTest)):
#     inTestSample.append(inTest[y][:100])
# print('done testing data preprocessing')
# #create network
# nnHandwriting = generateNeuralNetwork("HandwrittenDigitClassification",28*28,in_labels,10,out_labels,[50,20])
# #epoch size
# epoch_size = 1000
# #each epoch do
# for x in range(60):
#     #data sample
#     inTrainSample = []
#     outTrainSample = outData[x*epoch_size:(x+1)*epoch_size]
#     for y in range(len(inData)):
#         inTrainSample.append(inData[y][x*epoch_size:(x+1)*epoch_size])
#     trainNeuralNetwork(nnHandwriting,inTrainSample,outTrainSample,"stochastic")
#     testNeuralNetwork(nnHandwriting,inTestSample,outTestSample,x)
#     nnHandwriting.saveGraph("HandwritingNewNormalized.txt")

#XOR test
# filename = "xortest.csv"
# in_labels = ['A','B']
# out_labels = ['C']
# inData,outData = loadData(filename,len(in_labels))
# nn = generateNeuralNetwork("xor",2,in_labels,1,out_labels,[2])
# trainNeuralNetwork(nn,inData,outData)

#MNIST handwritten digit classification
# def loadNPZ(path):
#     with np.load(path) as f:
#         x_train, y_train = f['x_train'], f['y_train']
#         x_test, y_test = f['x_test'], f['y_test']
#         x_train, x_test = x_train/255.0, x_test/255.0
#
#         return (x_train, y_train), (x_test, y_test)
#
# (x_master, y_master), (x_test_master, y_test_master) = loadNPZ('mnist.npz')
# out_labels = ['0','1','2','3','4','5','6','7','8','9']
# in_labels = []
# for cl in range(28*28):
#     in_labels.append(str(cl+1))
# nnHandwriting = generateNeuralNetwork("HandwrittenDigitClassification",28*28,in_labels,10,out_labels,[200,80,20])
# #nnHandwriting.loadGraph("Handwriting0118.txt")
#
# epoch_size = 1000
# for epoch in range(0,60):
#     x_train = x_master[epoch*epoch_size:(epoch+1)*epoch_size]
#     y_train = y_master[epoch*epoch_size:(epoch+1)*epoch_size]
#     x_test = x_test_master[:100]
#     y_test = y_test_master[:100]
#
#     #generate labels and data for training
#     outData = [y_train.tolist()]
#     outData = outDataBinaryConverter(outData,out_labels)
#     inData = [[-1]*len(x_train) for i in range(28*28)]
#     for x in range(len(x_train)):
#         temp = x_train[x].tolist()
#         for r in range(28):
#             for c in range(28):
#                 inData[28*r+c][x] = temp[r][c]
#
#     #data for testing
#     outTest = [y_test.tolist()]
#     outTest = outDataBinaryConverter(outTest,out_labels)
#     inTest = [[-1]*len(x_test) for i in range(28*28)]
#     for x in range(len(x_test)):
#         temp = x_test[x].tolist()
#         for r in range(28):
#             for c in range(28):
#                 inTest[28*r+c][x] = temp[r][c]
#
#     #train the network
#     trainNeuralNetwork(nnHandwriting,inData,outData,"stochastic")
#     testNeuralNetwork(nnHandwriting,inTest,outTest,epoch)
#     nnHandwriting.saveGraph("Handwriting3Layer.txt")
#
# #final network testing
# x_test = x_test_master[:5000]
# y_test = y_test_master[:5000]
# outTest = [y_test.tolist()]
# outTest = outDataBinaryConverter(outTest,out_labels)
# inTest = [[-1]*len(x_test) for i in range(28*28)]
# for x in range(len(x_test)):
#     temp = x_test[x].tolist()
#     for r in range(28):
#         for c in range(28):
#             inTest[28*r+c][x] = temp[r][c]
# testNeuralNetwork(nnHandwriting,inTest,outTest,0)
