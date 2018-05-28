import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import LeakyReLU
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, MaxPooling2D
from keras.callbacks import *
from keras.layers.normalization import BatchNormalization
import random
from keras import optimizers
import os 

def load_X():
    print("loading...X")
    x1 = np.load('../../../data_pooling_xy/X_all.npy')
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

X = load_X()
y = load_y() 

def VGG_3D():
    print("Build model!!")
    model = Sequential()
    model.add(BatchNormalization(input_shape=(70,3,3,5)))
    model.add(Convolution3D(256,3, strides=(1,1,1), activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(Convolution3D(128,3, strides=(1,1,1),activation='relu', padding='same',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(MaxPooling3D((3,3,3), strides=(1,1,1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

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
    for i in range(10):
        model.add(Dense(256, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
        #model.add(LeakyReLU(alpha=0.1))
        model.add(BatchNormalization())
    model.add(Dense(70, activation = 'linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
    return model

model = VGG_3D()
print(model.summary())
dirpath = "../../../model/mog_like_3/"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)


filepath="../../../model/mog_like_3/weights-improvement-{epoch:03d}-{loss:.3e}.hdf5"
sgd = optimizers.SGD(lr=0.3, decay=1e-9, momentum=0.5, nesterov=True)
model.compile(optimizer = sgd, loss='mean_squared_error')
checkpoint = ModelCheckpoint(filepath, monitor='loss', 
                            save_best_only=False, mode='min', period=5)

model.fit(X,y, batch_size=2048, epochs=150, shuffle=True, callbacks = [checkpoint])


