import random
import math
import copy
import numpy as np

from Hyperparameters import parameters
np.random.seed(parameters['seedNumber'])

from keras.layers import MaxPooling2D, AveragePooling2D, Dropout

# Constant seed number 
random.seed(parameters['seedNumber'])

class PoolingLayer:
    # Parameter Dictionary
    _parameters = parameters['pool']

    # Hyper-Parameters
    kernelSize = None
    stride = None
    poolingType = None
    dropoutRate = None

    def applyLocalMove(self, _input, initParams, blockNo):

        newParams = copy.deepcopy(initParams) # Copy Initial Hyper-parameters in current Pool. Layer
        parameterKeys = list(set(self._parameters.keys())) # Get existing hyper-params names except stride in pool. hyper-params
        if blockNo == 1:
            parameterKeys = list(set(self._parameters.keys()) - {'dropoutRate'})    

        rndParameterName = random.choice(parameterKeys) # Select Random Hyper-parameter name
        tempParams = np.array(self._parameters[rndParameterName])
        oldParamValue = initParams[rndParameterName]

        # Select Hyper-parameter value round up or down
        valueRound = random.choice(['up', 'down'])
        if rndParameterName == 'dropoutRate':
            valueRound = 'up'

        if valueRound == 'up':
            newParams[rndParameterName] = tempParams[np.where(tempParams > oldParamValue)][0] if len(tempParams[np.where(tempParams > oldParamValue)]) != 0 else tempParams[np.where(tempParams < oldParamValue)][-1]
        else:
            newParams[rndParameterName] = tempParams[np.where(tempParams < oldParamValue)][-1] if len(tempParams[np.where(tempParams < oldParamValue)]) != 0 else tempParams[np.where(tempParams > oldParamValue)][0]
       
        self.kernelSize = newParams["kernelSize"]
        self.stride = 2
        self.poolingType = newParams["poolType"]
        self.dropoutRate = newParams["dropoutRate"]

        # Repair Pooling Layer Parameters
        outputSize = self.calculateOutputSize(int(_input.shape[1]), self.kernelSize, self.stride)
        if outputSize < 1:
            self.kernelSize, self.stride = self.repairPoolingLayer(copy.deepcopy(self._parameters), self.kernelSize, self.stride, int(_input.shape[1]))

        parameters = {"kernelSize": self.kernelSize, "stride": self.stride, "poolType": self.poolingType, "dropoutRate": self.dropoutRate}

        return parameters

    # Add Pooling + Dropout    
    def addManuelPoolingLayer(self, _input, kernelSize, stride, poolType, dropoutRate):
        
        if poolType == "MAX":
            _input = MaxPooling2D(pool_size=(int(kernelSize), int(kernelSize)), strides=int(stride))(_input)
        else:
            _input = AveragePooling2D(pool_size=(int(kernelSize), int(kernelSize)), strides=int(stride))(_input)
        
        output = Dropout(rate=dropoutRate)(_input)
        return output

    def addRandomPoolingLayer(self, initParams, prevBlock, blockNo, _input):
        dropoutRate = 0.2
        if blockNo != 1:
            prevBlock = prevBlock['Pool']
            oldVal = prevBlock['dropoutRate']
            params = np.array(self._parameters['dropoutRate']) 
            dropoutRate = params[np.where(params > oldVal)][0] if len(params[np.where(params > oldVal)]) != 0 else oldVal
        
        # Repair Pooling Layer Parameters
        kernelSize = random.choice(self._parameters['kernelSize'])
        outputSize = self.calculateOutputSize(int(_input.shape[1]), kernelSize, 2)
        if outputSize < 1:
            kernelSize, _ = self.repairPoolingLayer(copy.deepcopy(self._parameters), kernelSize, 2, int(_input.shape[1]))

        params = {"kernelSize": kernelSize, "stride": 2,
                 "poolType": random.choice(self._parameters['poolType']), "dropoutRate": dropoutRate}
        
        output = self.addManuelPoolingLayer(_input=_input, **params)
        return params, output

    def repairPoolingLayer(self, parameters, selectedKernelSize, selectedStride, inputSize):

        kernelSizeRange = sorted([val for val in parameters['kernelSize'] if val < selectedKernelSize], reverse= True)

        for kernelSize in kernelSizeRange:
            outputSize = self.calculateOutputSize(inputSize, kernelSize, selectedStride)
            if outputSize >= 1:
                return kernelSize, 2

        return min(parameters["kernelSize"]), 2
    
    def calculateOutputSize(self, inputSize, kernel, stride):
        return int(math.floor((inputSize - kernel) / stride)) + 1
