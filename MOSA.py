import copy
import math
import random
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import CNNModel as cnnModelObj
from matplotlib.ticker import FuncFormatter
from Hyperparameters import parameters

# Constant seed number 
random.seed(parameters['seedNumber'])

class MOSA:

    datasetName = "CIFAR"
    # Variables
    s = None # Current Solution
    s_prime = None # s': The solution created after the random move
    s_best = None # Best Solution
    modelNo = 0 # Model No
    archive = [] # Potentially Pareto Optimal List
    allSolutions = []
    results = [] # Results Pickle
    
    filePath = "Results/"

    burnin_DeltaF = {"FMNIST": 0.01693, "BALANCED":0.03672, 'LETTERS':0.06544, 'MNIST':	0.08664, 'CIFAR': 0.09377}
    
    initAccProb = 0.5
    cr = 0.85 # Cooling Rate
    NEW_BLOCK_PROB = 0.0625

    T_init = 0.577078016
    T_final = 0.12 # Final temperature deÄŸil iterasyon sayÄ±sÄ± sonlandÄ±rma kriteri olacak.

    nbr_total_iter = 500 #Solution Budget
    nbr_outer_iter = int(math.log(T_final / T_init) / math.log(cr))
    nbr_inner_iter = int(nbr_total_iter / nbr_outer_iter)

    print(nbr_outer_iter)
    print(nbr_inner_iter)

    T_current = T_init
    

    def rndProb(self):
        return random.random()
    
    # Calculate the acceptance probability
    def accProb(self, S, S_prime, temperature):
        deltaF = self.calculateDeltaF(S, S_prime, self.archive)
        return math.exp(-deltaF / temperature)
    
    def __init__(self, x_train, y_train, x_valid, y_valid, batch_size, learning_rate):

        self.clearFile("models.txt")
        self.clearFile("archive.txt")
        self.clearFile("model_history.txt")

        self.initParameters = {"x_train": x_train, "x_valid": x_valid,
                               "y_train": y_train, "y_valid": y_valid,
                               "batch_size": batch_size, "learning_rate": learning_rate}
        
        self.modelNo = self.modelNo + 1
        self.s = cnnModelObj.CNNModel(self.NEW_BLOCK_PROB)
        self.s.buildInitSolution()
        self.s.trainModel(**self.initParameters, modelNo=self.modelNo)
        self.s_best = self.s
        self.archive.append(self.s)

        self.results.append({'tr_acc': self.s.trainAccuracy, 'val_acc': self.s.validationAccuracy, 'flops': self.s.flops, 'totalParameter': self.s.parameterCount, 'time': self.s.trainTime, 'status':True})
        
        # Log
        self.writeFile("Initial Solution Created...\n")
        print('Initial Solution Created...')
        self.writeFile(f"Initial Solution Objective Value: {self.s.objectiveValue} \n")
        print(f"Initial Solution Objective Value: {self.s.objectiveValue}")
    
    def clearFile(self, fileName):
        f = open(f"{self.filePath}{fileName}", 'w')
        f.close()

    def writeFile(self, text, _filePath="models.txt"):
        f = open(self.filePath + _filePath, "a")
        f.write(text)
    
    def isSolutionDominateArchive(self, S, archive):
         for a in self.archive:
             if self.isDominate(S, a): # a solution in Archive is dominated by new
                return True 
    
    def isArchiveDominateSolution(self, archive, S):
         for a in self.archive:
             if self.isDominate(a, S): # a solution in Archive dominates new
                return True   
        
    def calculateDeltaF(self, S, S_prime, archive):
        F_S = self.getDominatedCount(S, archive) + 1
        F_S_prime = self.getDominatedCount(S_prime, archive) + 1
        return (1 / (len(archive) + 2)) * (F_S_prime - F_S)
    
    # Count archive elements that dominate S
    def getDominatedCount(self, S, archive):
        S.dominationCount = 0
        for solution in archive:
            if self.isDominate(solution, S):
                S.dominationCount = S.dominationCount + 1 
        return S.dominationCount
    

    def calcArchiveDomination(self, archive):
        # Reset Archive solution domination count
        for a in archive:
            a.dominationCount = 0
        
        for i, _ in enumerate(archive):
            for j, _ in enumerate(archive):
                if self.isDominate(archive[i], archive[j]):
                    #print(str(archive[i].objectiveValue) + "-" + str(archive[i].parameterCount) + " dominating " + str(archive[j].objectiveValue) + "-" + str(archive[j].parameterCount))
                    archive[j].dominationCount = archive[j].dominationCount + 1
        
        return archive
    
    def updateArchive(self, S, archive):
        # add S in archive
        archive.append(S)
        
        archive = self.calcArchiveDomination(archive)
        
        tempArchive = []
        for a in archive:
            if a.dominationCount < 1:
                tempArchive.append(a)
        
        return tempArchive

    # Is S weakly dominate S'
    def isDominate(self, S, S_prime):
        if S.objectiveValue <= S_prime.objectiveValue:
            if S.flops <= S_prime.flops:
                if S.objectiveValue < S_prime.objectiveValue or S.flops < S_prime.flops:
                    return True
        
        return False
    
    def printArchive(self, archive):
        for a in archive:
            print("Model:" + str(a.objectiveValue) + "-" + str(a.flops) + "-" + str(a.dominationCount))
        print("*" * 50)
    
    def millions(self, x, pos):
        'The two args are the value and tick position'
        if x > 1e-6:
            return '%1.1fM' % (x * 1e-6)
        else:
            return '%1.1fK' % (x * 1e-5)
        
    def saveList(self, _list, name):
        pickle_out = open(f"Results/{name}.pickle","wb")
        pickle.dump(_list, pickle_out)
        pickle_out.close()

    def startAlgorithm(self):

        inner_counter = 0
        outer_counter = 0
        archiveList = []

        while outer_counter < self.nbr_outer_iter or self.modelNo < self.nbr_total_iter:
            
            # Outer Loop Info
            self.writeFile(f"{'#' * 10} Outer Iteration: {outer_counter} {'#' * 10}\n")
            print("-" * 50)

            inner_counter = 0

            while inner_counter < self.nbr_inner_iter:
                
                # Inner Loop Info
                self.writeFile(f"{'#' * 10} Inner Iteration: {inner_counter} {'#' * 10}\n")
                print("-" * 50)

                # Apply Local Move
                self.modelNo = self.modelNo + 1
                print(f"MODEL NO: {self.modelNo}")

                if self.modelNo > self.nbr_total_iter:
                    break

                if self.modelNo % 50 == 0:
                    self.NEW_BLOCK_PROB = self.NEW_BLOCK_PROB * 1.4

                self.s_prime = cnnModelObj.CNNModel(self.NEW_BLOCK_PROB)
                self.s_prime.buildCNN(copy.deepcopy(self.s.topologyDict))
                self.s_prime.trainModel(**self.initParameters, modelNo=self.modelNo)
                
                print("S - " + str(self.s.objectiveValue) + "-" + str(self.s.flops))
                print("S_prime - " + str(self.s_prime.objectiveValue) + "-" + str(self.s_prime.flops))
                print('S dominate S_prime:', self.isDominate(self.s, self.s_prime))
                print("*"*50)

                ##################### Choose Next and Update ######################
                acceptModel = False
                # Current dominates new
                if self.isDominate(self.s, self.s_prime): 
                    p_acc = self.accProb(self.s, self.s_prime, self.T_current)
                    if self.rndProb() < p_acc:
                        self.s = self.s_prime
                        acceptModel = True 
                    
                else:
         
                    if self.isSolutionDominateArchive(self.s_prime, self.archive): # a solution in Archive is dominated by new
                        self.s = self.s_prime
                        self.archive = self.updateArchive(self.s_prime, self.archive)
                        acceptModel = True 
                    
                    elif self.isArchiveDominateSolution(self.archive, self.s_prime): # a solution in Archive dominates new
                        
                        a_star = random.choice(self.archive)
                        rnd01 = self.rndProb()
                        
                        if self.isDominate(self.s_prime, self.s): # S' dominates S
                            p_acc = self.accProb(a_star, self.s_prime, self.T_current)
                            if rnd01 < p_acc:
                                self.s = self.s_prime
                                acceptModel = True
                                 
                            else:
                                self.s = a_star
                                acceptModel = True 
                        
                        elif self.isDominate(self.s, self.s_prime) == False: # S does not dominate S'
                            p_acc = self.accProb(self.s, self.s_prime, self.T_current)
                            if rnd01 < p_acc:
                                p_acc = self.accProb(a_star, self.s_prime, self.T_current)
                                if rnd01 < p_acc:
                                    self.s = self.s_prime
                                    acceptModel = True 
                                else:
                                    self.s = a_star
                                    acceptModel = True
                            else:
                                p_acc = self.accProb(a_star, self.s, self.T_current)
                                if rnd01 >= p_acc:
                                    self.s = a_star
                                    acceptModel = True
                        
                    else: # new does not dominate or is dominated by solutions in Archive
                        self.s = self.s_prime
                        self.archive = self.updateArchive(self.s_prime, self.archive)
                        acceptModel = True

                ##################### Choose Next and Update ######################

                if acceptModel:
                     # New Solution Accept Info
                    self.writeFile(f"New Solution Accepted... - Objective: {self.s.objectiveValue} \n")
                    # Put Accepted Solution in Archive List
                    archiveList.append((self.s, self.s.objectiveValue))
                
                self.results.append({'tr_acc': self.s_prime.trainAccuracy, 'val_acc': self.s_prime.validationAccuracy, 'flops': self.s_prime.flops, 'totalParameter': self.s_prime.parameterCount, 'time': self.s_prime.trainTime, 'status':acceptModel})
                
                self.printArchive(self.archive)
                # Increase Inner Counter
                inner_counter = inner_counter + 1

            if outer_counter < self.nbr_outer_iter: 
                self.T_current = self.T_current * self.cr
                
                # Increase Outer Counter
                outer_counter = outer_counter + 1

        try:
            self.saveList(self.results, "mosa_sols")
        except Exception as e:
            print(f"Pickle Write Error... {e}")

        for index, solution in enumerate(self.archive):
            self.writeFile(str(solution.topologyDict) + "\n", _filePath="archive.txt")
            self.writeFile(f"Objective: {solution.objectiveValue} \n", _filePath="archive.txt")
            self.writeFile(f"Flops: {solution.flops} \n", _filePath="archive.txt")
            self.writeFile(f"{'*' * 50} \n", _filePath="archive.txt")
            # serialize model to JSON - kerasModel Silindi
            model_json = str(solution.modelJSON)
            with open(f"Results/model_{index}.json", "w") as json_file:
                json_file.write(model_json)
