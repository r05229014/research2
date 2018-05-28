import numpy as np
import sys
import random
from sklearn.preprocessing import StandardScaler
import os 

th = np.load('../../../data_8km_mean_X/th_8km_mean.npy')
#w = np.load('../../../data_8km_mean_X/w_8km_mean.npy')
#qv = np.load('../../../data_8km_mean_X/qv_8km_mean.npy')
#u = np.load('../../../data_8km_mean_X/u_8km_mean.npy')
#v = np.load('../../../data_8km_mean_X/v_8km_mean.npy')

def pool_reflect(array):
    ##########################################
    #The thing we gonna do is like this :    #
    #original array like this,               #   
    #                                        #   
    #[[1,2]                                  #
    # [3,4]]                                 #
    #                                        #   
    # transfer to                            #
    #                                        #
    #[[4,3,4,3]                              #
    # [2,1,2,1]                              #
    # [4,3,4,3]                              #       
    # [2,1,2,1]]                             #
    ## ########################################

    new = np.zeros((array.shape[0], array.shape[1], array.shape[2]+2, array.shape[3]+2))
    for t in range(array.shape[0]):
        print("t is now : %s" %t)
        for z in range(array.shape[1]):
            tmp = array[t,z,:,:]
            tmp_ = np.pad(tmp,1,'reflect')
            new[t,z] = tmp_
    print(new.shape)

    return new



