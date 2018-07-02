# import keras
import numpy as np
import scipy.io
import pywt
# from keras.models import Sequential
# from keras.layers import Activation, Flatten
# from keras.layers import Convolution2D
# from keras.layers.core import Dense
# from keras.utils import np_utils


file = "/Users/rmramesh/A/Volume-Forecast/venv/resources/dataset_BCIcomp1.mat"
data = scipy.io.loadmat(file)

xdata = data.get("x_train")
xdata = xdata.transpose(2,1,0)


print(xdata.shape)
print(xdata[0][0])

import matplotlib.pyplot as plt

def plotFor(index):
    (cA, cD) = pywt.dwt(xdata[index][0], 'db3')
    print(cA.shape)
    print(cD.shape)
    plt.plot(cA, color='yellow', linewidth=1)
    plt.plot(cD, color='green', linewidth=1)
    plt.plot(xdata[index][0], color='pink', linewidth=1)
    plt.show()


plotFor(0)
plotFor(1)
plotFor(2)
plotFor(3)