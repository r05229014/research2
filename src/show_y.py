import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

z_ = Dataset('../../data/mjoL.w3d.nc')
z = z_.variables['zc'][:] 
wqv = np.load('../../data_8km_mean_y/wqv_mean.npy')
print(wqv.shape)
for i in range(500,1423):
    plt.plot(wqv[i,:]*2.5*10**6, z)
    plt.show()
