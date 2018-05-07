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

from keras.datasets import mnist

# Load pre-shuffled MNIST data into train and test sets

# (X_train, y_train), (X_test, y_test) = mnist.load_data()

file = "/Users/rmramesh/A/Data/bcci/dataset_BCIcomp1.mat"
data = scipy.io.loadmat(file)

xdata = data.get("x_train")
xdata = xdata.transpose(2,0,1)

ydate = data.get("y_train")
ydata = ydate[:,0]
#
# print(xdata.shape)
# print(ydata.shape)

for i in range(1,140):
    ydate[i]= ydate[i]/2

ydate = ydate[:,0]

newdata = xdata
newdata = np.zeros(shape=(140,1152,1152))
np.c_[xdata, newdata]

xdata = newdata

print (xdata[:100].shape)
print (ydata[:100].shape)
print (xdata[100:].shape)
print (ydate[100:].shape)


X_train = xdata[:100]

y_train = ydata[:100]

X_test = xdata[100:]

y_test = ydate[100:]

X_train = X_train.reshape(X_train.shape[0], 1152, 1152,1)
X_test = X_test.reshape(X_test.shape[0], 1152, 1152,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1152,1152,1)))
model.add(Convolution2D(10, 1, activation='relu'))
model.add(Convolution2D(2, 1150))
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

model.fit(X_train, Y_train, batch_size=20, nb_epoch=10, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print("Score:")
print(score)

y_pred = model.predict(X_test)
print("y predict")
print("y test")
print(y_pred[:9])
print(y_test[:9])