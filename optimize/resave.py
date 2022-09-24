import os
import pickle
import hdf5storage

from directory import set_path
_, _, datadir = set_path(os.path.abspath(__file__))


filename = 'temp.pkl'
filedir = os.path.join(datadir, filename)
var = pickle.load(open(filedir, 'rb'))

filename = 'data_concat_embed.mat'
filedir = os.path.join(datadir, filename)
# hdf5storage.write(var, '.', filedir, matlab_compatible=True)
hdf5storage.savemat(filedir, var, oned_as='column')