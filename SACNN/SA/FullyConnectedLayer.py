import random
import math
import copy
import numpy as np
from keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization
from Hyperparameters import parameters

# Constant seed number 
random.seed(parameters['seedNumber'])

class FullyConnectedLayer:
    
    # Parameter Dictionary
    activationParameters = parameters['learningProcess']
    _parameters = parameters['fullyConnected']
    
    # Hyper-Parameters
    unitCount = None
    dropoutRate = None
    activation = None

    def applyLocalMove(self, input, initParams):
        newParams = copy.deepcopy(initParams)
        rndParameterName = random.choice(list(self._parameters.keys()))
        tempParams = np.array(self._parameters[rndParameterName])
        oldParamValue = initParams[rndParameterName]
        
        # Select Hyper-parameter value round up or down
        valueRound = random.choice(['up', 'down'])
       
        # Select new hyper-parameter value for selected hyper-parameter name
        if valueRound == 'up':
            newParams[rndParameterName] = tempParams[np.where(tempParams > oldParamValue)][0] if len(tempParams[np.where(tempParams > oldParamValue)]) != 0 else tempParams[np.where(tempParams < oldParamValue)][-1]
        else:
            newParams[rndParameterName] = tempParams[np.where(tempParams < oldParamValue)][-1] if len(tempParams[np.where(tempParams < oldParamValue)]) != 0 else tempParams[np.where(tempParams > oldParamValue)][0]

        self.unitCount = newParams["unitCount"]
        self.dropoutRate = newParams["dropoutRate"]
        self.activation = newParams['activation']

        parameters = {"unitCount": self.unitCount, "dropoutRate": self.dropoutRate, "activation": self.activation}
        
        return parameters

    def addManuelFullyConnectedLayer(self, unitCount, dropoutRate, activation, _input):
        
        # Add Hidden Layer
        _input = Dense(units=unitCount)(_input)

        # Add Activation Function
        if activation == 'leaky_relu':
            _input = LeakyReLU()(_input)
        else:
            _input = Activation(activation)(_input)

        # Add Batch Norm.
        _input = BatchNormalization()(_input)

        # Add Dropout
        output = Dropout(rate=dropoutRate)(_input)

        return output

    def expandFullyBlock(self, selectedActFunct, _input):
        unitCount = random.choice(self._parameters['unitCount'])
        dropoutRate = random.choice(self._parameters['dropoutRate'])
        activation = selectedActFunct

        blockParams = {"Fully":{'unitCount':unitCount, 'dropoutRate':dropoutRate, 'activation':activation}}
        output = self.addManuelFullyConnectedLayer(_input = _input, **blockParams['Fully'])

        return blockParams, output

    def calculateNumberOfParameter(self, inputUnitCount, outputUnitCount):
        return (inputUnitCount + 1) * outputUnitCount 