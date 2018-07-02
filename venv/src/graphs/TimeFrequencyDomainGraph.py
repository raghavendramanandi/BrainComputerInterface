# pip install keras
# pip install tensorflow
import keras
import numpy as np
import scipy.io

from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D
from keras.layers.core import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pywt

from keras.models import load_model
from matplotlib.pyplot import specgram

def spectrum (vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps

file = "/Users/rmramesh/Downloads/EEG_BCI_MI_AllSub/SubC_6chan_2LR_s1.mat"
data = scipy.io.loadmat(file)
print(data.keys())

xdata = data.get("EEGDATA")
print(xdata.shape)
xdata = xdata.transpose(2,0,1)
print(xdata.shape)

ydata = data.get("LABELS")
print(ydata.shape)

xshape = xdata.shape
numberOfItems = xshape[0]
numberOfElectrodes = xshape[1]
numberOfReadings = xshape[2]

# Array shape after processing
nrap = int(numberOfReadings/2)

xprocessed = np.zeros([numberOfItems,nrap,nrap], dtype=float)

num = 16
Pxx, freqs, bins, im = plt.specgram(xdata[num][0], NFFT=64, Fs=256, noverlap=32)
print(ydata[num])
plt.show()