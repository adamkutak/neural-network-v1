import math
import random

class Graph:
    def __init__(self,name):
        self.name = name
        self.vertexes = []
        self.edges = []
        self.inList = []
        self.neuronList = []
        self.outList = []
        self.LEARNING_RATE = 0.01

    def addVertice(self,t,id,label,bias,layer = 0):
        if(t=="input"):
            self.vertexes.append(inputVertice(id,label))
            self.inList.append(id)
        elif(t=="neuron"):
            self.vertexes.append(neuron(id,label,layer,bias))
            self.neuronList[layer-1].append(id)
        elif(t=="output"):
            self.vertexes.append(outputVertice(id,label,layer,bias))
            self.outList.append(id)
        else:
            self.vertexes.append(Vertice(id,label))

    def setLayers(self,neuronLayers):
        for l in range(neuronLayers):
            self.neuronList.append([])

    def addEdge(self,id,out,inb,weight):
        self.edges.append(Edge(id,out,inb,weight,self))

    def removeVertice(self,id):
        self.vertexes.pop(id)

    def removeEdge(self,id):
        self.edges.pop(id)

    def saveGraph(self,filename):
        outfile = open(filename,'w')
        outfile.write(self.name + "\n")
        a = ""
        b = ""
        c = ""
        for x in self.inList:
            a = a + str(x) + ","
        for y1 in self.neuronList:
            for y2 in y1:
                b = b + str(y2) + ","
            b = b + "|"
        for z in self.outList:
            c = c + str(z) + ","
        outfile.write(a[:-1] + "\n")
        outfile.write(b[:-1] + "\n")
        outfile.write(c[:-1] + "\n")
        outfile.write("\n")
        typecodes = ['i','n','o','v']
        for v in self.vertexes:
            if(type(v) is inputVertice):
                x = 0
                bias = 0
            elif(type(v) is neuron):
                x = 1
                bias = v.bias
            elif(type(v) is outputVertice):
                x = 2
                bias = v.bias
            else:
                x = 3
                bias = 0
            outfile.write(typecodes[x]+"|"+str(v.id)+"|"+str(v.label)+"|"+str(v.layer)+"|"+str(v.y)+"|"+str(bias) + "\n")
        outfile.write("\n")
        for e in self.edges:
            outfile.write(str(e.id)+"|"+str(e.out)+"|"+str(e.inb)+"|"+str(e.weight) + "\n")
        outfile.close()

    def loadGraph(self,filename):
        infile = open(filename,'r')
        self.name = infile.readline().rstrip("\n")
        self.inList = infile.readline().rstrip("\n").split(',')
        for x1 in range(len(self.inList)):
            self.inList[x1] = int(self.inList[x1])
        t1 = infile.readline().rstrip("\n").split("|")
        self.neuronList = []
        for x2 in range(len(t1)):
            t2 = t1[x2][:-1].split(",")
            self.neuronList.append([])
            for xt in range(len(t2)):
                self.neuronList[x2].append(int(t2[xt]))
        self.outList = infile.readline().rstrip("\n").split(',')
        for x3 in range(len(self.outList)):
            self.outList[x3] = int(self.outList[x3])
        infile.readline()
        self.vertexes = []
        v = infile.readline().rstrip("\n")
        while(v!="" and v!="""\n"""):
            v = v.split("|")
            if(v[0]=="i"):
                temp = inputVertice(int(v[1]),v[2])
            elif(v[0]=="n"):
                temp = neuron(int(v[1]),v[2],int(v[3]),float(v[5]))
            elif(v[0]=="o"):
                temp = outputVertice(int(v[1]),v[2],int(v[3]),float(v[5]))
            elif(v[0]=="v"):
                temp = Vertice(int(v[1]),v[2])
            self.vertexes.append(temp)
            v = infile.readline().rstrip("\n")
        self.edges = []
        e = infile.readline().rstrip("\n")
        while(e!="" and e!="""\n"""):
            e = e.split("|")
            self.edges.append(Edge(int(e[0]),int(e[1]),int(e[2]),float(e[3]),self))
            e = infile.readline().rstrip("\n")
        infile.close()

    def calculate(self,inputVector):
        if(len(inputVector)!=len(self.inList)):
            return "error: mismatched input vector"
        for x in self.inList:
            self.vertexes[x].updateY(inputVector[x])
            #print("input ", x, " = ",self.vertexes[x].getY())
        for y in self.neuronList:
            for y2 in y:
                self.vertexes[y2].updateY(self)
                #print("neuro ", y2, " = ",self.vertexes[y2].getY())
        for z in self.outList:
            self.vertexes[z].updateY(self)
            #print("ouput ", z, " = ",self.vertexes[z].getY())
        sum = 0
        for v in self.outList:
            sum += math.exp(self.vertexes[v].getY())
        outputVector = []
        maxI = []
        maxProb = 0
        for w in self.outList:
            outputVector.append(self.vertexes[w].updateProbability(sum))
            if(outputVector[-1] > maxProb):
                maxProb = outputVector[-1]
                maxI = []
                maxI.append(w)
            elif(outputVector[-1] == maxProb):
                maxI.append(w)

        return outputVector,maxI

    def convertToReal(self,outputVector,maxI,type):
        if(len(maxI)!=1):
            return "equal outcome error"
        elif(type=="probability"):
            return outputVector[0]
        elif(type=="label"):
            return self.vertexes[maxI].label

    def getQuadraticCost(self,a,y):
        c = 0
        for x in range(len(a)):
            c += pow(y[x]-a[x],2)
        c = c*0.5
        return c

    def teachNetwork_stochastic(self,inputVector,expectedVector):
        outputVector, maxI = self.calculate(inputVector)
        for out in range(len(self.outList)):
            o = self.outList[out]
            err = self.vertexes[o].updateError(expectedVector[out])
            self.vertexes[o].updateBias(self.vertexes[o].bias - self.LEARNING_RATE*err)
            for i in self.vertexes[o].inEdges:
                cw = self.vertexes[self.edges[i].out].y*err
                self.edges[i].weight = self.edges[i].weight - self.LEARNING_RATE*cw
        for layer in range(len(self.neuronList)-1,-1,-1):
            for neur in range(len(self.neuronList[layer])):
                n = self.neuronList[layer][neur]
                err = self.vertexes[n].updateError(self)
                self.vertexes[n].updateBias(self.vertexes[n].bias - self.LEARNING_RATE*err)
                for i in self.vertexes[n].inEdges:
                    cw = self.vertexes[self.edges[i].out].y*err
                    self.edges[i].weight = self.edges[i].weight - self.LEARNING_RATE*cw
                    #temp = self.edges[i].weight
                    #print(self.edges[i].weight, " = ", temp ," - ", self.LEARNING_RATE*cw)

        for x in range(len(outputVector)):
            outputVector[x] = round(outputVector[x],4)
        return outputVector

    def teachNetwork_minibatch(self,inputVector,expectedVector):
        weightGradients = []
        biasGradients= []
        outputVector, maxI = self.calculate(inputVector)
        for out in range(len(self.outList)):
            o = self.outList[out]
            err = self.vertexes[o].updateError(expectedVector[out])
            biasGradients.append([o,self.LEARNING_RATE*err])
            for i in self.vertexes[o].inEdges:
                cw = self.vertexes[self.edges[i].out].y*err
                weightGradients.append([i,self.LEARNING_RATE*cw])
        for layer in range(len(self.neuronList)-1,-1,-1):
            for neur in range(len(self.neuronList[layer])):
                n = self.neuronList[layer][neur]
                err = self.vertexes[n].updateError(self)
                biasGradients.append([n,self.LEARNING_RATE*err])
                for i in self.vertexes[n].inEdges:
                    cw = self.vertexes[self.edges[i].out].y*err
                    weightGradients.append([i,self.LEARNING_RATE*cw])
                    #temp = self.edges[i].weight
                    #print(self.edges[i].weight, " = ", temp ," - ", self.LEARNING_RATE*cw)

        for x in range(len(outputVector)):
            outputVector[x] = round(outputVector[x],4)
        return outputVector, weightGradients, biasGradients

    def teachNetwork(self,inputVector,expectedVector,algoType):
        if(algoType=="stochastic"):
            return self.teachNetwork_stochastic(inputVector,expectedVector)
        else:
            return self.teachNetwork_minibatch(inputVector,expectedVector)

class Vertice:
    def __init__(self,id,label):
        self.id = id
        self.label = label
        self.inEdges = []
        self.outEdges = []
        self.error = 0

    def outDeg(self):
        return len(self.outEdges)

    def inDeg(self):
        return len(self.inEdges)

    def newEdge(self,edgeID,type):
        if(type=="in"):
            self.inEdges.append(edgeID)
        elif(type=="out"):
            self.outEdges.append(edgeID)

    def travel(self,edge,g):
        if(edge not in self.outEdges):
            return "error: vertex doesn't have this edge"
        else:
            return g.edges[edge].self.inb

    def backtravel(self,edge,g):
        if(edge not in self.inEdges):
            return "error: vertex doesn't have this edge"
        else:
            return g.edges[edge].out

class Edge:
    def __init__(self,id,out,inb,weight,g):
        self.id = id
        self.inb = inb
        self.out = out
        self.weight = weight
        g.vertexes[inb].newEdge(id,"in")
        g.vertexes[out].newEdge(id,"out")

    def updateWeight(self,new):
        self.weight = new

    def getWeight(self):
        return self.weight

class inputVertice(Vertice):
    def __init__(self,id,label):
        super().__init__(id,label)
        self.layer = 0
        self.y = 0

    def updateY(self,value):
        self.y = value

    def scaleY(self,scaletype):
        self.y = 0

    def getY(self):
        return self.y

class neuron(Vertice):
    def __init__(self,id,label,layer,bias):
        super().__init__(id,label)
        self.layer = layer
        self.bias = bias
        self.y = 0
        self.z = 0

    def getY(self):
        return self.y

    def updateY(self,g):
        self.z = self.bias
        for x in self.inEdges:
            self.z += g.vertexes[g.edges[x].out].getY()*g.edges[x].getWeight()
            #print("y: ",g.vertexes[g.edges[x].out].getY()," weight: ",g.edges[x].getWeight())
        #print("z neuron:", self.z)

        #this is just so the sigmoid function doesn't overflow with decimal places
        if(self.z > 100):
            self.z = 100
        elif(self.z < -100):
            self.z = -100

        self.y = 1/(1+math.exp(-1*self.z))
        #print("y neuron:", self.y)

    def updateBias(self,newB):
        self.bias = newB

    def updateError(self,g):
        sigPrime = self.y*(1-self.y)
        s = 0
        for x in self.outEdges:
            weigh = g.edges[x].weight
            errUp = g.vertexes[g.edges[x].inb].error
            s += weigh*errUp
        self.error = s*sigPrime
        #print("neuron error: ",self.error)
        return self.error

class outputVertice(Vertice):
    def __init__(self,id,label,layer,bias):
        super().__init__(id,label)
        self.layer = layer
        self.bias = bias
        self.prob = 0
        self.z = 0
        self.y = 0

    def getY(self):
        return self.y

    def updateY(self,g):
        self.z = self.bias
        for x in self.inEdges:
            self.z += g.vertexes[g.edges[x].out].getY()*g.edges[x].getWeight()
        #print("z output:", self.z)
        #this is just so the sigmoid function doesn't overflow with decimal places
        if(self.z > 100):
            self.z = 100
        elif(self.z < -100):
            self.z = -100

        self.y = 1/(1+math.exp(-1*self.z))
        #print("y output:",self.y)
    def updateProbability(self,sum):
        self.prob = math.exp(self.y)/sum
        return self.prob

    def updateBias(self,newB):
        self.bias = newB

    def updateError(self,yP):
        sigPrime = self.y*(1-self.y)
        costPrime = self.prob - yP
        self.error = sigPrime*costPrime
        return self.error
