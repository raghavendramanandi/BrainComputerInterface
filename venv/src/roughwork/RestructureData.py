import scipy.io
import numpy as np


file = "/Users/rmramesh/A/Data/bcci/dataset_BCIcomp1.mat"
data = scipy.io.loadmat(file)
# dict_keys(['__header__', '__version__', '__globals__', 'Copyright', 'x_train', 'x_test', 'y_train'])

xtrain = data.get("x_train")

print(xtrain.transpose(2,0,1).shape)