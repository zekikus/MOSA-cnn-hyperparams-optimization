import random
import math
import copy
import numpy as np
from Hyperparameters import parameters

np.random.seed(parameters['seedNumber'])

from keras.layers import Dense, Dropout, LeakyReLU, Activation, BatchNormalization

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

    def expandFullyBlock(self, prevBlockParams, selectedActFunct, _input):
        
        if 'Fully' in prevBlockParams:
            prevBlockParams = prevBlockParams['Fully']
            unitCount = min(prevBlockParams['unitCount'] * 2, max(self._parameters['unitCount']))
            dropoutRate = max(self._parameters['dropoutRate'])
            activation = prevBlockParams['activation']
        else:
            unitCount = min(self._parameters['unitCount'])
            dropoutRate = min(self._parameters['dropoutRate'])
            activation = selectedActFunct

        blockParams = {"Fully":{'unitCount':unitCount, 'dropoutRate':dropoutRate, 'activation':activation}}
        output = self.addManuelFullyConnectedLayer(_input = _input, **blockParams['Fully'])

        return blockParams, output

    def controlParameters(self, prevBlockParams, newBlockParams, blockNo, fullyHyperParams, selectedActFunc):

        minUnitCount = min(fullyHyperParams['unitCount'])
        maxUnitCount = max(fullyHyperParams['unitCount'])
        selectedUnitCount = newBlockParams['unitCount']

        selectedDropoutRate = newBlockParams['dropoutRate']
        
        if ('Fully' in prevBlockParams) and (selectedUnitCount < prevBlockParams['Fully']['unitCount']):
            prevBlockParams = prevBlockParams['Fully']
            newBlockParams['unitCount'] = min(maxUnitCount, prevBlockParams['unitCount'] * 2)

        if 'Fully' in prevBlockParams:
            prevBlockParams = prevBlockParams['Fully']
            if prevBlockParams['unitCount'] <= selectedUnitCount:
                newBlockParams['dropoutRate'] = max(prevBlockParams['dropoutRate'], selectedDropoutRate)
        
        newBlockParams['activation'] = selectedActFunc
        return newBlockParams

    def calculateNumberOfParameter(self, inputUnitCount, outputUnitCount):
        return (inputUnitCount + 1) * outputUnitCount 