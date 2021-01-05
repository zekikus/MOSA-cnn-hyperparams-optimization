import copy
import pickle
import math
import random
import numpy as np
import tensorflow as tf
import CNNModel as cnnModelObj
from Hyperparameters import parameters

# Constant seed number 
random.seed(parameters['seedNumber'])

class SA:

    datasetName = "CIFAR10"
    # Variables
    s = None # Current Solution
    s_prime = None # s': The solution created after the random move
    s_best = None # Best Solution
    modelNo = 0 # Model No
    
    results = []
    filePath = "Results/"

    burnin_DeltaF = {"FMNIST": 0.01504, "MNIST": 0.00687, "CIFAR10": 0.08857, "DIGITS": 0.00149, "LETTERS": 0.01121, "BALANCED":0.00827}
    
    nbr_total_iter = 500 # Solution Budget
    nbr_outer_iter = 25 # 20 değeri için de test edilecek
    nbr_inner_iter = nbr_total_iter / nbr_outer_iter
    
    initAccProb = 0.5
    cr = 0.99 # Cooling Rate
    NEW_BLOCK_PROB = 0.0625

    T_init = -burnin_DeltaF[datasetName] / math.log(initAccProb)

    T_current = T_init
    T_final = 0 # Final temperature değil iterasyon sayısı sonlandırma kriteri olacak.

    def saveList(self, _list, name):
        pickle_out = open(f"Results/{name}.pickle","wb")
        pickle.dump(_list, pickle_out)
        pickle_out.close()

    def rndProb(self):
        return random.random()
    
      # Calculate the acceptance probability
    def accProb(self, deltaE, temperature):
        return math.exp(-deltaE / temperature)
    
    def __init__(self, x_train, y_train, x_valid, y_valid, batch_size, learning_rate):

        self.clearFile("models.txt")
        self.clearFile("result.txt")
        self.clearFile("model_history.txt")

        self.initParameters = {"x_train": x_train, "x_valid": x_valid,
                               "y_train": y_train, "y_valid": y_valid,
                               "batch_size": batch_size, "learning_rate": learning_rate}
        
        self.modelNo = self.modelNo + 1
        self.s = cnnModelObj.CNNModel(self.NEW_BLOCK_PROB)
        self.s.buildInitSolution()
        self.s.trainModel(**self.initParameters, modelNo=self.modelNo)
        self.s_best = self.s
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
        print(text)

    def startAlgorithm(self):

        inner_counter = 0
        outer_counter = 0
        archiveList = []

        while outer_counter < self.nbr_outer_iter:
            
            # Outer Loop Info
            self.writeFile(f"{'#' * 10} Outer Iteration: {outer_counter} {'#' * 10}\n")
            print("-" * 50)

            inner_counter = 0

            while inner_counter < self.nbr_inner_iter:
                
                # Inner Loop Info
                self.writeFile(f"{'#' * 10} Inner Iteration: {inner_counter} {'#' * 10}\n")
                print("-" * 50)

                if self.modelNo % 50 == 0:
                    self.NEW_BLOCK_PROB = self.NEW_BLOCK_PROB * 1.4

                # Apply Local Move
                self.modelNo = self.modelNo + 1
                self.s_prime = cnnModelObj.CNNModel(self.NEW_BLOCK_PROB)
                self.s_prime.buildCNN(copy.deepcopy(self.s.topologyDict))
                self.s_prime.trainModel(**self.initParameters, modelNo=self.modelNo)

                # Calculate Energy Change
                deltaE = self.s_prime.objectiveValue - self.s.objectiveValue
                self.writeFile(f"DeltaE:{deltaE}\n")

                acceptModel = False
                if deltaE < 0:
                    # Kabul Edilecek
                    acceptModel = True
                   
                elif deltaE == 0:
                    # Parametre sayısı kontrol edilcek
                    if self.s_prime.parameterCount < self.s.parameterCount:
                        acceptModel = True
                    else:
                         # Reddedildi
                        self.writeFile(f"New Solution Rejected... - Objective: {self.s_prime.objectiveValue} \n")

                else:
                    # Olasılık hesaplanacak,
                    if self.rndProb() <= self.accProb(deltaE, self.T_current):
                        acceptModel = True
                    else:
                        # Reddedildi
                        self.writeFile(f"New Solution Rejected... - Objective: {self.s_prime.objectiveValue} \n")

                if acceptModel:
                     # New Solution Accept Info
                    self.writeFile(f"New Solution Accepted... - Objective: {self.s_prime.objectiveValue} \n")

                    # Current Solution Update
                    self.s = self.s_prime

                    # Put Accepted Solution in Archive List
                    archiveList.append((self.s, self.s.objectiveValue)) 
                
                self.results.append({'tr_acc': self.s_prime.trainAccuracy, 'val_acc': self.s_prime.validationAccuracy, 'flops': self.s_prime.flops, 'totalParameter': self.s_prime.parameterCount, 'time': self.s_prime.trainTime, 'status':acceptModel})
                # Increase Inner Counter
                inner_counter = inner_counter + 1

            self.T_current = self.T_current * self.cr
            
            
            # Increase Outer Counter
            outer_counter = outer_counter + 1

        try:
            self.saveList(self.results, "sau_sols")
        except Exception as e:
            print(f"Pickle Write Error... {e}")

        # Sort Archive Solutions
        sortedList = sorted(archiveList, key=lambda x: x[1])[:5]
        for index, solution in enumerate(sortedList):
            self.writeFile(str(solution[0].topologyDict) + "\n", _filePath="result.txt")
            self.writeFile(f"Objective: {solution[1]} \n", _filePath="result.txt")
            self.writeFile(f"{'*' * 50} \n", _filePath="result.txt")
            # serialize model to JSON - kerasModel Silindi
            model_json = str(solution[0].modelJSON)
            with open(f"Results/model_{index}.json", "w") as json_file:
                json_file.write(model_json)
