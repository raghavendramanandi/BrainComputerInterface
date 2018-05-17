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


def spectrum (vector):
    (cA, cD) = pywt.dwt(vector, 'db3')
    return (cA, cD)

# file = "/Users/raghavendra/Documents/python/VolumeForecasting/venv/resources/dataset_BCIcomp1.mat"
file = "/Users/rmramesh/A/Volume-Forecast/venv/resources/dataset_BCIcomp1.mat"
data = scipy.io.loadmat(file)

xdata = data.get("x_train")
xdata = xdata.transpose(2,1,0)
xprocessed = np.zeros([140,576,576], dtype=float)

for i in range(0,140):
    for j in range(0,3):
        xprocessed[i][j] = spectrum(xdata[i][j])

for i in range(0,140):
    for j in range(3,575):
        xprocessed[i][j] = np.zeros([576], dtype=float)

# x1 = np.copy(xprocessed[0][0])

ydata = data.get("y_train")

#reducce y data from 1, 2 to 0, 1 respectively
print("ydata shalpe:")
for i in range(0,140):
    ydata[i][0] = ydata[i][0] / 2

ydata = ydata[:, 0]

print("Shape: ")
print(xprocessed.shape)
print(ydata.shape)

X_train = xprocessed[:100]
y_train = ydata[:100]
X_test = xprocessed[100:]
y_test = ydata[100:]

print("Data formation: ")
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

X_train = X_train.reshape(X_train.shape[0], 576, 576,1)
X_test = X_test.reshape(X_test.shape[0], 576, 576,1)

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
model.add(Convolution2D(32, 1, 1, activation='relu', input_shape=(576,576,1)))
model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(2, 576))
model.add(Flatten())
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

print("X_train:")
print(X_train.shape)
print("Y_train:")
print(Y_train.shape)

model.fit(X_train, Y_train, batch_size=25, epochs=4, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print("Score:")
print(score)

y_pred = model.predict(X_test)

print("y predict")
print("y test")

print(y_pred[:9])
print(y_test[:9])
