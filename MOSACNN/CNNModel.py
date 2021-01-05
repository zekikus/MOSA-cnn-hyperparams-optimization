import copy
import random as rnd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import regularizers, optimizers
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import ConvolutionLayer as convObject
import PoolingLayer as poolObject
import FullyConnectedLayer as fullyObject
from Hyperparameters import parameters
from TimeHistory import TimeHistory
from early_stopping_v2 import EarlyStoppingV2
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1.keras.backend import clear_session
from tensorflow.compat.v1.keras.backend import get_session


# Constant seed number 
rnd.seed(parameters['seedNumber'])
tf.compat.v1.disable_v2_behavior() 

class CNNModel:
    
    conv = convObject.ConvolutionLayer()
    pool = poolObject.PoolingLayer()
    fully = fullyObject.FullyConnectedLayer()
    
    # Constants
    MIN_CONV_BLOCK = 2
    MAX_CONV_BLOCK = 4
    MIN_CONV_LAYER = 2
    MAX_CONV_LAYER = 4
    MAX_FULLY_BLOCK = 2
    CHANGE_PARAM_PROB = 0.5
    REMOVE_FULLY_BLOCK_PROB = 0.2
    ADD_FULLY_BLOCK_PROB = 0.8
    ADD_CONV_LAYER_PROB = 0.8
    REMOVE_CONV_LAYER_PROB = 0.2
    NEW_BLOCK_PROB = 0.0625
    
    # Variables
    input_shape = (32,32,3)
    numberOfClasses = 10
    topologyDict = dict()
    modelJSON = dict()
    kerasModel = None
    parameterCount = None
    flops = None
    trainAccuracy = None
    trainTime = None
    validationAccuracy = None
    objectiveValue = None
    dominationCount = 0
    modelActFunction = None # Relu, Elu or Leaky_Relu
    modelSubSamplingMethod = None # Strive or Pooling
    filePath = "Results/"

    def __init__(self, block_prob):
        self.NEW_BLOCK_PROB = block_prob

    def writeFile(self, text, fileName="models.txt"):
        f = open(self.filePath + fileName, "a")
        f.write(text)

    # Reset Keras Session
    def reset_keras(self):
        sess = get_session()
        clear_session()
        sess.close()
        sess = get_session()

        # use the same config as you used to create the session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.visible_device_list = "0"
        set_session(tf.compat.v1.Session(config=config))

    def get_flops(self, test):
        session = tf.compat.v1.Session()
        graph = tf.compat.v1.get_default_graph()

        with graph.as_default():
            with session.as_default():
                
                model = test

                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

                # We use the Keras session graph in the call to the profiler.
                flops = tf.compat.v1.profiler.profile(graph=graph,
                                                    run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops

    def buildInitSolution(self):
        topology = {"CONV_BLOCK_1":{"#Conv": 2, "#Pool":1, "#Strive": 0,"Conv":{'kernelSize':5, 'kernelCount':32, 'stride':1, 'padding':'same','activation':'elu'},
                              "Pool":{'kernelSize':3, 'stride':2, 'poolType': 'MAX', 'dropoutRate':0.2}, "Strive":{}},
                    "CONV_BLOCK_2":{"#Conv": 3, "#Pool":1, "#Strive": 0, "Conv":{'kernelSize':3, 'kernelCount':64, 'stride':1, 'padding':'same','activation':'elu'},
                               "Pool":{'kernelSize':3, 'stride':2, 'poolType': 'MAX', 'dropoutRate':0.4}, "Strive":{}},
                    "FULLY_BLOCK_1":{"Fully":{'unitCount':128, 'dropoutRate':0.5, 'activation':'elu'}},
                    "Topology": {"#ConvBlock":2, "#FullyBlock":1}}

        convBlockNo = topology['Topology']['#ConvBlock']
        fullyBlockNo = topology['Topology']['#FullyBlock']
        createdBlockNo = 0

        # Create New Model
        _input = Input(self.input_shape)
        cpy_input = _input
        
        # Create Convolution Blocks
        for block, attr in topology.items():
            
            if block == "Topology": continue

            # Create Convolution Blocks
            if createdBlockNo < convBlockNo:
                for conv_layer in range(attr['#Conv']):
                    _input = self.conv.addManuelConvLayer(**attr['Conv'], _input=_input)
                _input = self.pool.addManuelPoolingLayer(**attr['Pool'], _input=_input)
            
            # Create Fully Connected Layers
            else:
                # Create Flatten Layer
                if createdBlockNo + 1 == convBlockNo + 1:
                    _input = Flatten()(_input)
                _input = self.fully.addManuelFullyConnectedLayer(**attr['Fully'], _input=_input)

            # Increase Block No
            createdBlockNo = createdBlockNo + 1
        
        # Create Softmax Layer
        output = Dense(self.numberOfClasses, activation='softmax')(_input)
        
        # Build Model
        model = Model(inputs = cpy_input, outputs = output)
        
        self.topologyDict = topology
        self.kerasModel = model
        self.parameterCount = model.count_params()
        self.modelJSON = model.to_json()

        ##### BURADA
        self.flops = self.get_flops(model)
        print(f"#Params: {self.parameterCount}, FLOPS: {self.flops}")
        #####

        # Free Memory - Yeni Eklendi
        del model
        return topology

    def buildConvolutionBlock(self, _input, currentBlock, blockNo, striveOrPool):

        currentSubSamplingMethod = 'Pool'
        if currentBlock['#Pool'] == 0:
            currentSubSamplingMethod = 'Strive'

        # Decide Add or Remove Conv. Layer in this Conv. Block
        if currentBlock['#Conv'] < self.MAX_CONV_LAYER:
            if rnd.uniform(0, 1) < self.ADD_CONV_LAYER_PROB:
                currentBlock['#Conv'] = currentBlock['#Conv'] + 1
        elif currentBlock['#Conv'] > self.MIN_CONV_LAYER:
            if rnd.uniform(0, 1) < self.REMOVE_CONV_LAYER_PROB:
                currentBlock['#Conv'] = currentBlock['#Conv'] - 1

        currentBlock['Conv']['activation'] = self.modelActFunction
        # Change Convolution Layer Params with 0.5 Prob
        if rnd.uniform(0, 1) > self.CHANGE_PARAM_PROB:
            currentBlock['Conv'] = self.conv.applyLocalMove(_input, currentBlock['Conv'])

        # Add Conv. Layer up to Number of Convolution Layer in That Convolution Block
        for i in range(currentBlock['#Conv']):
            _input = self.conv.addManuelConvLayer(_input=_input, **currentBlock['Conv'])

        # Change Pooling Layer or Strive Layer Params with 0.5 Prob
        if currentSubSamplingMethod == striveOrPool and  rnd.uniform(0, 1) > self.CHANGE_PARAM_PROB:
            if striveOrPool == 'Pool':
                currentBlock[striveOrPool] = self.pool.applyLocalMove(_input, currentBlock['Pool'], blockNo)
            else:
                currentBlock[striveOrPool] = self.conv.applyLocalMoveStriveConv(currentBlock, blockNo, self.modelActFunction, _input)
        
        methods = {"Pool_Rnd": self.pool.addRandomPoolingLayer,
                   "Pool_Fix": self.pool.addManuelPoolingLayer,
                   "Strive_Rnd": self.conv.addRandomStriveConvLayer,
                   "Strive_Fix": self.conv.addManuelStriveConLayer}

        if len(currentBlock[striveOrPool]) != 0:
            # Fixed Random and Manuel StriveConvLayer Activation Function Problem
            if striveOrPool == 'Strive':
                currentBlock['Strive']['kernelCount'] = currentBlock['Conv']['kernelCount']
                currentBlock[striveOrPool]['activation'] = self.modelActFunction
            
            # Fixed Negative Output Size Error
            if _input.shape[1] < currentBlock[striveOrPool]['kernelSize']:
                currentBlock[striveOrPool]['kernelSize'] = min(parameters['pool']['kernelSize'])
            _input = methods[striveOrPool + "_Fix"](_input=_input, **currentBlock[striveOrPool])
        else:
            currentBlock[striveOrPool], _input = methods[striveOrPool + "_Rnd"](currentBlock, blockNo, _input)
       
        # Clean Topology Dictionary
        cleanDictKey = 'Pool'
        if striveOrPool == 'Pool':
            cleanDictKey = 'Strive'
        currentBlock[cleanDictKey] = dict()
        currentBlock['#' + cleanDictKey] = 0

        currentBlock['#' + striveOrPool] = 1

        return currentBlock, _input
    
    def buildFullyBlock(self, _input, currentBlock, blockNo):
        currentBlock['Fully']['activation'] = self.modelActFunction
        if rnd.uniform(0, 1) > self.CHANGE_PARAM_PROB:
            currentBlock['Fully'] = self.fully.applyLocalMove(_input, currentBlock['Fully'])
        _input = self.fully.addManuelFullyConnectedLayer(_input=_input, **currentBlock['Fully'])

        return currentBlock, _input

    def buildCNN(self, initSolTopology):

        # Log Variable
        log = ""
        print(f"NEW BLOCK PROB: {self.NEW_BLOCK_PROB}")

        # Determine Model Common Functions
        self.modelActFunction = rnd.choice(parameters['learningProcess']['activation'])
        self.modelSubSamplingMethod = rnd.choice(['Strive', 'Pool'])

        createNewConvBlock = False
        removeFullyBlocks = False

        # Set Current Solution (S) Topology to New Solution (S')
        self.topologyDict = initSolTopology
        convBlockCount = self.topologyDict['Topology']['#ConvBlock']
        fullyBlockCount = self.topologyDict['Topology']['#FullyBlock']
        
        # Decide whether to use Fully Connected Blocks or not.
        if fullyBlockCount != 0 and rnd.uniform(0, 1) < self.REMOVE_FULLY_BLOCK_PROB:
            for i in range(1, fullyBlockCount + 1):
                del self.topologyDict['FULLY_BLOCK_' + str(i)]
            
            fullyBlockCount = 0
            removeFullyBlocks = True
            self.topologyDict['Topology']['#FullyBlock'] = fullyBlockCount
        
        # Create Model
        _input = Input(self.input_shape)
        cpy_input = _input

        # Create Convolution Blocks
        for block_no in range(1, convBlockCount + 1):
            currentBlock = self.topologyDict['CONV_BLOCK_' + str(block_no)]
            self.topologyDict['CONV_BLOCK_' + str(block_no)], _input = self.buildConvolutionBlock(_input, currentBlock, block_no, self.modelSubSamplingMethod)
            # Append Log Result
            log = log + "Conv_Block_" + str(block_no) + "\n"
            log = log + str(self.topologyDict['CONV_BLOCK_' + str(block_no)]) + "\n" + ("-" * 50) + "\n"

        # Decide whether to add New Convolution Block
        # input.shape[1] >= min(parameters['pool']['kernelSize']) Input Minimum kernel size'dan büyükse yeni blok ekle
        if self.topologyDict['Topology']['#ConvBlock'] < self.MAX_CONV_BLOCK:
            if rnd.uniform(0, 1) < self.NEW_BLOCK_PROB:
                createNewConvBlock = True
                convBlockCount = convBlockCount + 1
                convLayerCount = int(round(rnd.uniform(self.MIN_CONV_LAYER, self.MAX_CONV_LAYER)))
                self.topologyDict['Topology']['#ConvBlock'] = convBlockCount
                self.topologyDict['CONV_BLOCK_' + str(convBlockCount)], _input = self.conv.expandConvBlock(convLayerCount, self.modelSubSamplingMethod, self.modelActFunction, _input)
                # Append Log Result
                log = log + "Conv_Block_" + str(convBlockCount) + "\n"
                log = log + str(self.topologyDict['CONV_BLOCK_' + str(convBlockCount)]) + "\n" + ("-" * 50) + "\n"

        # Flatten Layer
        _input = Flatten()(_input)

        # Create Fully Connected Blocks
        for block_no in range(1, fullyBlockCount + 1):
            currentBlock = self.topologyDict['FULLY_BLOCK_' + str(block_no)]
            self.topologyDict['FULLY_BLOCK_' + str(block_no)], _input = self.buildFullyBlock(_input, currentBlock, block_no)
            # Append Log Result
            log = log + "Fully_Block_" + str(block_no) + "\n"
            log = log + str(self.topologyDict['FULLY_BLOCK_' + str(block_no)]) + "\n" + ("-" * 50) + "\n"

        # Decide whether to add New Fully Connected Blocks or not
        if fullyBlockCount < self.MAX_FULLY_BLOCK and removeFullyBlocks == False: 
            if rnd.uniform(0, 1) < self.ADD_FULLY_BLOCK_PROB:
                fullyBlockCount = fullyBlockCount + 1
                self.topologyDict['Topology']['#FullyBlock'] = fullyBlockCount
                self.topologyDict['FULLY_BLOCK_' + str(fullyBlockCount)], _input = self.fully.expandFullyBlock(self.modelActFunction, _input)
                # Append Log Result
                log = log + "Fully_Block_" + str(fullyBlockCount) + "\n"
                log = log + str(self.topologyDict['FULLY_BLOCK_' + str(fullyBlockCount)]) + "\n" + ("-" * 50) + "\n"
        
        # Generalization - 29.04.19
        if fullyBlockCount == 0:
            _input = Dropout(rate=0.5)(_input)

        # Create Softmax Layer
        output = Dense(self.numberOfClasses, activation='softmax')(_input)
    
        # Build Model
        model = Model(inputs = cpy_input, outputs = output)
        
        self.kerasModel = model
        self.parameterCount = model.count_params() # YENİ EKLENDİ - HATA VARSA SİL
        self.writeFile(text=log)

        self.modelJSON = model.to_json() # YENİ EKLENDİ - HATA VARSA SİL
        ##### BURADA
        self.flops = self.get_flops(model)
        print(f"#Params: {self.parameterCount}, FLOPS: {self.flops}")
        #####
        # Free Memory - Yeni Eklendi
        del model
    
    def trainModel(self,x_train, y_train, x_valid, y_valid, batch_size, learning_rate, modelNo):
        #training
        epochs = 100
        batch_size = batch_size
        opt_method = optimizers.Adam(lr=learning_rate)
        self.kerasModel.compile(loss='categorical_crossentropy',
                optimizer=opt_method,
                metrics=['accuracy'])

        time_callback = TimeHistory()
        self.earlyStop = EarlyStoppingV2()
        # Train History
        history = self.kerasModel.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_valid, y_valid), callbacks=[self.earlyStop, time_callback], verbose=2)
        
        self.trainAccuracy = history.history['acc'][-1]
        self.validationAccuracy = history.history['val_acc'][-1]
        self.objectiveValue = 1 - self.validationAccuracy

        _history = f"{'*' * 20} \n Model: {modelNo}\n"
        for epoch in range(self.earlyStop.stopped_epoch):
            _history = _history + f"Epoch: {epoch + 1}, tr_loss: {history.history['loss'][epoch]}, tr_acc: {history.history['acc'][epoch]}, val_loss: {history.history['val_loss'][epoch]}, val_acc: {history.history['val_acc'][epoch]}\n"
        self.writeFile(text=_history, fileName='model_history.txt')
        
        self.trainTime = sum(time_callback.times)

        # Write Log
        log = f"Model No: {modelNo}\n"
        log = log + f"Parameter Count: {self.parameterCount} \n"
        log = log + f"Flops: {self.flops} \n"
        log = log + f"Train Accuracy: {self.trainAccuracy} \n"
        log = log + f"Validation Accuracy: {self.validationAccuracy} \n"
        log = log + f"Objective Value: {self.objectiveValue} \n"                    
        self.writeFile(text=log)

        tf.compat.v1.reset_default_graph()
        #K.clear_session()
        self.reset_keras()
        del self.kerasModel


"""
cnnModel = CNNModel()
topology = cnnModel.buildInitSolution()


for i in range(200):
    print("Model", i)
    newCnn = CNNModel()
    newCnn.buildCNN(copy.deepcopy(topology))
    topology = newCnn.topologyDict
    print("*" * 50)
    K.clear_session()
"""
