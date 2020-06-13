import random
import math
import copy
import numpy as np
import PoolingLayer as poolObject
from Hyperparameters import parameters

np.random.seed(parameters['seedNumber'])

from keras import regularizers
from keras.layers import Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout

# Constant seed number 
random.seed(parameters['seedNumber'])

class ConvolutionLayer:

    # Parameter Dictionary
    activationParameters = parameters['learningProcess']
    _parameters = parameters["conv"]

    # Hyper-parameters
    kernelSize = None
    kernelCount = None
    stride = None
    padding = None
    activation = None

    # New convolution layer hyper-parameters are determined by applying local movements to the existing convolution layer.
    def applyLocalMove(self, _input, initParams):

        newParams = copy.deepcopy(initParams) # Copy Initial Hyper-parameters in current Convolution Layer
        parameterKeys = list(set(self._parameters.keys()) - {'padding', 'dropoutRate'}) # Get existing hyper-params names except padding and act. in conv. hyper-params
        rndParameterName = random.choice(parameterKeys) # Select Random Hyper-parameter name
        tempParams = np.array(self._parameters[rndParameterName]) 
        oldParamValue = initParams[rndParameterName]

        # Select Hyper-parameter value round up or down
        valueRound = random.choice(['up', 'down'])
        if rndParameterName == 'kernelSize':
            valueRound = 'down'
        elif rndParameterName == 'kernelCount':
            valueRound = 'up'    
       
        # Select new hyper-parameter value for selected hyper-parameter name
        if valueRound == 'up':
            newParams[rndParameterName] = tempParams[np.where(tempParams > oldParamValue)][0] if len(tempParams[np.where(tempParams > oldParamValue)]) != 0 else tempParams[np.where(tempParams < oldParamValue)][-1]
        else:
            newParams[rndParameterName] = tempParams[np.where(tempParams < oldParamValue)][-1] if len(tempParams[np.where(tempParams < oldParamValue)]) != 0 else oldParamValue
            
        self.kernelSize = newParams["kernelSize"]
        self.kernelCount = newParams["kernelCount"]
        self.padding = "SAME"
        self.activation = newParams['activation']

        parameters = {"kernelSize": self.kernelSize, "kernelCount": self.kernelCount, 
                      "padding": self.padding, "stride": 1, "activation": self.activation}

        return parameters

    # Add Convolution + Activation + Batch Norm.
    def addManuelConvLayer(self, kernelSize, kernelCount, stride, padding, activation, _input):
        
        # Add Convolution Layer
        _input = Conv2D(filters=int(kernelCount), kernel_size=(int(kernelSize), int(kernelSize)), 
                        strides=(int(stride), int(stride)), padding=padding, 
                        kernel_regularizer=regularizers.l2(1e-4))(_input)
        
        # Add Activation Function
        if activation == 'leaky_relu':
            _input = LeakyReLU()(_input)
        else:
            _input = Activation(activation)(_input)
        
        # Add Batch Norm.
        output = BatchNormalization()(_input)
        return output

    def addManuelStriveConLayer(self, kernelSize, kernelCount, stride, padding, activation, dropoutRate, _input):
        _input = Conv2D(filters=int(kernelCount), kernel_size=(int(kernelSize), int(kernelSize)), 
                        strides=(int(stride), int(stride)), padding=padding)(_input)
        
        if activation == 'leaky_relu':
            _input = LeakyReLU()(_input)
        else:
            _input = Activation(activation)(_input)
        
        output = Dropout(rate=dropoutRate)(_input)
        return output

    def addRandomStriveConvLayer(self, initParams, prevBlock, blockNo, _input):
        params = self.applyLocalMoveStriveConv(initParams, prevBlock, blockNo, initParams['Conv']['activation'], _input)
        output = self.addManuelStriveConLayer(_input=_input, **params)
        return params, output

    def applyLocalMoveStriveConv(self, initParams, prevBlock, blockNo, selectedActFunc, _input):
        dropoutRate = 0.2
        if blockNo != 1:
            prevBlock = prevBlock['Strive']
            oldVal = prevBlock['dropoutRate']
            params = np.array(self._parameters['dropoutRate']) 
            dropoutRate = params[np.where(params > oldVal)][0] if len(params[np.where(params > oldVal)]) != 0 else oldVal

        # Repair Strive Layer Parameters
        kernelSize = random.choice(parameters['pool']['kernelSize'])
        outputSize = self.calculateOutputSize(int(_input.shape[1]), kernelSize, 2)
        if outputSize < 1:
            kernelSize, _ = self.repairLayer(copy.deepcopy(parameters['pool']), kernelSize, 2, int(_input.shape[1]))

        return {"kernelSize": kernelSize, "kernelCount": initParams['Conv']['kernelCount'], "padding": "VALID", "stride": 2,
                 "activation": selectedActFunc, "dropoutRate": dropoutRate}
        

    def expandConvBlock(self, prevBlockParams, convLayerCount, striveOrPool, selectedActFunction, _input):
        stride = prevBlockParams['Conv']['stride']
        kernelCount = min(prevBlockParams['Conv']['kernelCount'] + 32, max(self._parameters['kernelCount']))
        kernelSize = min(prevBlockParams['Conv']['kernelSize'], min(self._parameters['kernelSize']))
        
        poolKernelSize = random.choice(parameters['pool']['kernelSize'])
        poolDropoutRate = max(parameters['pool']['dropoutRate'])

        # Repair Pooling or Strive Layer Parameters
        outputSize = self.calculateOutputSize(int(_input.shape[1]), poolKernelSize, 2)
        if outputSize < 1:
            poolKernelSize, _ = self.repairLayer(copy.deepcopy(parameters['pool']), poolKernelSize, 2, int(_input.shape[1]))

        blockParams = {"#Conv": convLayerCount, "#Pool":0, "#Strive": 0, "Pool":{}, "Strive":{},
                       "Conv":{'kernelSize':kernelSize, 'kernelCount':kernelCount, 'stride':stride, 'padding':'same','activation':selectedActFunction}}

        # Add Conv. Layers in New Conv. Block
        for i in range(blockParams['#Conv']):
            _input = self.addManuelConvLayer(_input=_input, **blockParams['Conv'])

        # Add Pooling or Strive Layer in New Conv. Block
        if striveOrPool == 'Pool':
            blockParams["Pool"] = {'kernelSize':poolKernelSize, 'stride': 2, 'poolType': 'MAX', 'dropoutRate':poolDropoutRate}
            pool = poolObject.PoolingLayer()
            output = pool.addManuelPoolingLayer(_input=_input, **blockParams['Pool'])
            blockParams["#Pool"] = 1
        else:
            blockParams["Strive"] = {"kernelSize": poolKernelSize, "kernelCount": kernelCount, "padding": "VALID", "stride": 2,
                                     "activation": selectedActFunction, "dropoutRate": poolDropoutRate}
            output = self.addManuelStriveConLayer(_input=_input, **blockParams['Strive'])
            blockParams['#Strive'] = 1

        return blockParams, output
        
    def controlParameters(self, prevBlockParams, newBlockParams, blockNo, convHyperParams, selectedActFunc):

        minKernelCount = min(convHyperParams['kernelCount'])       
        maxKernelCount = max(convHyperParams['kernelCount'])
        selectedKernelCount = newBlockParams['kernelCount']

        minKernelSize = min(convHyperParams['kernelSize'])
        maxKernelSize = max(convHyperParams['kernelSize'])
        selectedKernelSize = newBlockParams['kernelSize']

        diffKernelSize = (maxKernelSize - convHyperParams['kernelSize'][-2]) # Amount of Increase in kernelSize Hyper-param

        if blockNo == 1:
            if selectedKernelCount == (minKernelCount + 32):
                selectedKernelCount = random.choice([selectedKernelCount, minKernelCount])
            
            newBlockParams['kernelCount'] = min(minKernelCount + 32, selectedKernelCount)
            newBlockParams['kernelSize'] = random.choice([(maxKernelSize - diffKernelSize), maxKernelSize])
        else:
            prevBlockParams = prevBlockParams['Conv']
            if selectedKernelCount <= prevBlockParams['kernelCount']:
                newBlockParams['kernelCount'] = min(prevBlockParams['kernelCount'] + 32, maxKernelCount)
            if (selectedKernelCount - 32) > prevBlockParams['kernelCount']:
                # Increase Variety
                newBlockParams['kernelCount'] = random.choice([selectedKernelCount - 32, selectedKernelCount])
            if (selectedKernelSize + diffKernelSize < prevBlockParams['kernelSize']):
                newBlockParams['kernelSize'] = random.choice([selectedKernelSize, selectedKernelSize + diffKernelSize]) 
            """
            Kernel Size aynı kalırsa 
            if selectedKernelSize == prevBlockParams['kernelSize']:
                newBlockParams['kernelSize'] = max(selectedKernelSize - diffKernelSize, minKernelSize)
            """
        newBlockParams['activation'] = selectedActFunc
        return newBlockParams
    
    def repairLayer(self, parameters, selectedKernelSize, selectedStride, inputSize):

        kernelSizeRange = sorted([val for val in parameters['kernelSize'] if val < selectedKernelSize], reverse= True)

        for kernelSize in kernelSizeRange:
            outputSize = self.calculateOutputSize(inputSize, kernelSize, selectedStride)
            if outputSize >= 1:
                return kernelSize, 2

        return min(parameters["kernelSize"]), 2
    
    def calculateOutputSize(self, inputSize, kernel, stride):
        return int(math.floor((inputSize - kernel) / stride)) + 1
