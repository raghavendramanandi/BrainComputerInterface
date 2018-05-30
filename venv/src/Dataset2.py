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

def spectrum (vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps

def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)

file = "/Users/rmramesh/Downloads/EEG_BCI_MI_AllSub/SubC_6chan_2LF_s1.mat"
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

for i in range(0,numberOfItems):
    for j in range(0,numberOfElectrodes):
        # (ca,cb) = CWavelets(selectedxdata[i][j])
        # xprocessed[i][j] = ca[2:]
        # xprocessed[i][28+j] = cb[2:]
        xprocessed[i][j] = spectrum(xdata[i][j])
        xprocessed[i][j] = xprocessed[i][j] * 100000000

# for i in range(0,numberOfItems):
#     xprocessed[i] = scale_linear_bycolumn(xprocessed[i])


# print(xprocessed[0])
# exit(0)

ydata = ydata.transpose(1,0)
ydata = list(ydata[0])

# num = 14
# print(ydata[num])
# # plt.plot(xdata[9][0], color='black', linewidth=1)
# plt.plot((xprocessed[num][2]), color='green', linewidth=1)
# plt.show()
# exit(0)

# reducce y data from 1, 2 to 0, 1 respectively
for i in range(0,numberOfItems):
    ydata[i] = int(ydata[i] / 2)

ydata = np.array(ydata)

print("Shape: ")
print(xprocessed.shape)
print(ydata.shape)
print(ydata)

print("NRAP:")
print(nrap)


X_train = xprocessed[:(numberOfItems-20)]
y_train = ydata[:(numberOfItems-20)]
X_test = xprocessed[(numberOfItems-20):]
y_test = ydata[(numberOfItems-20):]
print("Data formation: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape(X_train.shape[0], nrap, nrap,1)
X_test = X_test.reshape(X_test.shape[0], nrap, nrap,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print("Data formation: ")
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(nrap,nrap,1)))
model.add(Convolution2D(40, 1, activation='relu'))
model.add(Convolution2D(20, 1, activation='relu'))
model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(2, (nrap-2)))
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=60, epochs=3, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print("Score:")
print(score)

y_pred = model.predict(X_test.reshape(X_test.shape[0], nrap, nrap,1))

print("y predict:")
print(y_pred[:10])
print("y test:")
print(y_test[:10])

model.save("../outputmodel1/Dataset2-2.h5")
