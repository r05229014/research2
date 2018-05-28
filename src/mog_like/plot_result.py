import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
import os

def load_y():
    print("loading...y")
    y = np.load('../../../data_8km_mean_y/wqv_32.npy')
    y = y[400:800]    
    new_y = np.zeros((32*32*400,70))
    j = 0
    for t in range(400):
        #print(j)
        for x in range(32):
            for y_ in range(32):
                new_y[j] = y[t,:,x,y_]
                j +=1
    return new_y

def load_X():
    print("loading...X")
    x1 = np.load('../../../data_pooling_xy/X_400_799.npy')
    #x2 = np.load('../../../data_pooling_xy/X_1200_1423.npy')
    #X = np.concatenate((x1,x2), axis=0)
    print(x1.shape)
        
    return x1

img_dir = '../../../img/mog/' 
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

y = load_y()
X = load_X()
model = load_model('../../../model/mog_like_3/weights-improvement-150-5.220e-07.hdf5')
for i in range(y.shape[0]):
    pre = model.predict(X[i:i+1], batch_size=1)
    print(i)
    plt.figure(i)
    plt.plot(y[i]*2.5*10**6 , label = 'True')
    plt.plot(pre[0]*2.5*10**6 , label='pre')
    plt.legend()
    plt.savefig('../../../img/mog/image_%d' %i)
    plt.close()
    


