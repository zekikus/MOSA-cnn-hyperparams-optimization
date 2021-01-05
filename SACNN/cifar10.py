import os
import keras
import numpy as np
import SA as sannealing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Variables
input_shape = [None, 32, 32, 3]
number_of_classes = 10

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Create Reduced Training Data
# In order to speed up the search process, a reduced sample of 50% of the original training samples are selected randomly; 
# and 10% of this reduced sample is used as the reduced validation set.
np.random.seed(226)
Xtrain = np.empty((int(x_train.shape[0] / 2),32,32,3), dtype='uint8')
Ytrain = np.empty(int(y_train.shape[0] / 2), dtype='uint8')
Xtest = np.empty((int(x_test.shape[0] / 2),32,32,3), dtype='uint8')
Ytest = np.empty((int(y_test.shape[0] / 2)), dtype='uint8')

for i in range(int(x_train.shape[0] / 2)):
    rnd = np.random.randint(0, x_train.shape[0])
    Xtrain[i,:,:,:] = x_train[rnd,:,:,:]
    Ytrain[i] = y_train[rnd]

for j in range(int(x_test.shape[0] / 2)):
    rnd = np.random.randint(0, y_test.shape[0])
    Xtest[j,:,:,:] = x_test[rnd,:,:,:]
    Ytest[j] = y_test[rnd]
# -----------------------------------------------------

# Convert output label to one hot vector
Ytrain = to_categorical(Ytrain, number_of_classes)
Ytest = to_categorical(Ytest, number_of_classes)
# -----------------------------------------------------

# 10% of this reduced sample is used as the reduced validation set.
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, 
                                                test_size = int(Xtrain.shape[0] * 0.1), random_state = 42)

print("Shape of training features: {}".format(Xtrain.shape))
print("Shape of training lables: {}".format(Ytrain.shape))
print("Shape of testing features: {}".format(Xvalid.shape))
print("Shape of testing lables: {}".format(Yvalid.shape))
# -----------------------------------------------------

# Data Normalization
Xtrain = Xtrain.astype('float32')
Xvalid = Xvalid.astype('float32')
Xtrain /= 255
Xvalid /= 255
# -----------------------------------------------------

# Set hyper-parameters
learning_rate = 0.0001
epoch = 50
batch_size = 32

# SA Parameters
parameters = {'x_train': Xtrain , 'y_train': Ytrain, 'x_valid': Xvalid, 'y_valid': Yvalid, 'batch_size':batch_size, 'learning_rate':learning_rate}

# Start SA Algorithm
alg = sannealing.SA(**parameters)
alg.startAlgorithm()

## Outputs: 
 # model_history.txt: The loss and accuracy values (per epoch) of each model produced for training and validation are stored.
 # models.txt: Store iteration number, model_no, #parameters, Flops, train accuracy, validation accuracy and model topology
 # sau_sols.pickle: It stores information about solutions on the archive.
 # *.json files: Stores information about the topology of solutions on the archive (Keras Model).