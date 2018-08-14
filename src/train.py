import numpy as np
import sys
from keras import Model
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
import random
from keras import optimizers
from keras.utils import multi_gpu_model
import os 


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
            
        return super(ModelMGPU, self).__getattribute__(attrname)


def load_X():
    print("loading...X")
    x1 = np.load('../../../data_pooling_xy/X_all.npy')
    x1 = x1[:,:,:,:,0:1]
    max_t = x1.max()
    x1 = x1/max_t
    print(x1.shape)
    
    return x1

def load_y():
    print("loading...y")
    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    new_y = np.zeros((32*32*1423,70))
    j = 0
    for t in range(1423):
        #print(j)
        for x in range(32):
            for y_ in range(32):
                new_y[j] = y[t,:,x,y_]
                j +=1
    return new_y


def VGG_3D():
    print("Build model!!")
    model = Sequential()
    #model.add(BatchNormalization(input_shape=(70,3,3,5)))
    model.add(Convolution3D(256,3, strides=(1,1,1), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros', input_shape=(70,3,3,1)))
    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(MaxPooling3D((3,3,3), strides=(1,1,1)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(Convolution3D(64,3, strides=(1,1,1),activation='selu', padding='same'))
    #model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(Convolution3D(64,3, strides=(1,1,1),activation='relu', padding='same'))
#    model.add(MaxPooling3D((2,2,2), strides=(2,2,2)))

#    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
#    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
#    model.add(MaxPooling3D((4,4,4), strides=(2,2,2)))
#    model.add(Dropout(0.2))
#    model.add(BatchNormalization())

    model.add(Flatten())
    for i in range(4):
        model.add(Dense(256, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
        #model.add(LeakyReLU(alpha=0.1))
        #model.add(BatchNormalization())
    model.add(Dense(70, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model

X = load_X()
y = load_y()
# check X
print(X[0,:,0,0,:])


model = VGG_3D()
#sgd = optimizers.SGD(lr=0.3, decay=1e-9, momentum=0.5, nesterov=True)
#model.compile(optimizer = sgd, loss='mean_squared_error')
parallel_model = ModelMGPU(model, 3)
parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
print(model.summary())
dirpath = "../../model/nor_onlyuse_th/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)


filepath= dirpath + "/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', 
                            save_best_only=False, period=5)

parallel_model.fit(X,y, batch_size=512, epochs=150, shuffle=True, callbacks = [checkpoint])

