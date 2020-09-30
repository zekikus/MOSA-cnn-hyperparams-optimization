import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In[2]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


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


# In[4]:


input_shape = [None, 32, 32, 3]
number_of_classes = 10

#Conver output label to one hot vector
Ytrain = to_categorical(Ytrain, number_of_classes)
Ytest = to_categorical(Ytest, number_of_classes)


# In[5]:


Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, 
                                                test_size = int(Xtrain.shape[0]*0.1), random_state = 42)

print("Shape of training features: {}".format(Xtrain.shape))
print("Shape of training lables: {}".format(Ytrain.shape))
print("Shape of testing features: {}".format(Xvalid.shape))
print("Shape of testing lables: {}".format(Yvalid.shape))


# In[6]:


#Hyper parameters
learning_rate = 0.01
epoch = 50
batch_size = 32


# In[8]:


Xtrain = Xtrain.astype('float32')
Xvalid = Xvalid.astype('float32')
Xtrain /= 255
Xvalid /= 255


# In[9]:


import MOSA as mosa



# In[ ]:


parameters = {'x_train': Xtrain , 'y_train': Ytrain, 'x_valid': Xvalid, 'y_valid': Yvalid, 'batch_size':batch_size, 'learning_rate':0.0001}

alg = mosa.MOSA(**parameters)
alg.startAlgorithm()


# In[ ]:




