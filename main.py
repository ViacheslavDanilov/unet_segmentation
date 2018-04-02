import numpy as np
import scipy.io as sio
import h5py
import numpy as np

arrays = {}
f = h5py.File('1.mat')
for k, v in f.items():
    arrays[k] = np.array(v)

print('Done')