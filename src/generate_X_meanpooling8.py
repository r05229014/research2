import numpy as np
from netCDF4 import Dataset
import skimage.measure

def mean_pooling(path, var_name):
    var_ = Dataset(path) # load

    var = var_.variables[var_name][0:1,:,:,:] # time = 0 
    var = skimage.measure.block_reduce(var, (1,1,8,8), np.mean) # mean pooling to 32*32 in xy

    for i in range(1,1423):
        tmp = var_.variables[var_name][i:i+1,:,:,:]   # same as t=0
        tmp = skimage.measure.block_reduce(tmp, (1,1,8,8), np.mean)
        var = np.concatenate((var, tmp), axis=0)        # concatenate
        print(var.shape)       # checksize
    np.save('../../data_8km_mean_X/' + var_name + '_8km_mean.npy', var)


path_list = ['../../data/mjoL.qv3d.nc', '../../data/mjoL.w3d.nc', 
        '../../data/mjoL.th3d.nc', '../../data/mjoL.u3dx.nc',
        '../../data/mjoL.u3dy.nc']

var_name_list = ['qv', 'w', 'th', 'u', 'v']

for path, var_name in zip(path_list, var_name_list):
    print('mean pooling variable is :' + var_name)
    mean_pooling(path, var_name)














#w_ = Dataset('../../data/mjoL.w3d.nc')

#w = w_.variables['w'][0:1,:,:,:]
#w = skimage.measure.block_reduce(w, (1,1,8,8), np.mean)
#for i in range(1,1423):
#    w_tmp = w_.variables['w'][i:i+1,:,:,:]
#    w_tmp = skimage.measure.block_reduce(w_tmp, (1,1,8,8), np.mean)
#    w = np.concatenate((w, w_tmp), axis=0)
#    print(w.shape)

#np.save('../../data_8km_mean_X/w_8km_mean.npy', w)
