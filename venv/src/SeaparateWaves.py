from matplotlib import pyplot as plt
import scipy.io
import numpy as np

file = "/Users/raghavendra/Documents/python/VolumeForecasting/venv/resources/dataset_BCIcomp1.mat"
data = scipy.io.loadmat(file)
xdata = data.get("x_train")
ydata = data.get("y_train")

xdata = xdata.transpose(2,1,0)
print(xdata.shape)
print(xdata[0][0].shape)
print(ydata[0:6])

def spectrum (vector):
    '''get the power spectrum of a vector of raw EEG data'''
    A = np.fft.fft(vector)
    ps = np.abs(A)**2
    ps = ps[:len(ps)//2]
    return ps

xprocessed = spectrum(xdata[0][0])
# print(xprocessed)

import matplotlib.pyplot as plt

def plodData(a, b, c):
    plt.plot(a, color='yellow', linewidth=1)
    plt.plot(b, color='blue', linewidth=1)
    plt.plot(c, color='green', linewidth=1)
    # plt.plot(xdata[0][0], color='green', linewidth=1)
    plt.show()

index = 0
# plodData(spectrum(xdata[0][0]),spectrum(xdata[0][1]),spectrum(xdata[0][2]))
plodData(spectrum(xdata[index][0]),spectrum(xdata[index][1]),spectrum(xdata[index][2]))


# coeffs = wavedec(xdata[0][0], 'db4', level=6)
# coeffs = pywt.dwt(xdata[0][0], 'db4', level=6)
# cA2, cD1, cD2,cD3,cD4,cD5,cD6 = coeffs

# plt.subplot(7, 1, 1)
# plt.plot(x)
# plt.ylabel('Noisy Signal')
# plt.subplot(7, 1, 2)
# plt.plot(cD6)
# plt.ylabel('noisy')
# plt.subplot(7,1,3)
# plt.plot(cD5)
# plt.ylabel("gamma")
# plt.subplot(7,1,4)
# plt.plot(cD4)
# plt.ylabel("beta")
# plt.subplot(7,1,5)
# plt.plot(cD3)
# plt.ylabel("alpha")
# plt.subplot(7,1,6)
# plt.plot(cD2)
# plt.ylabel("theta")
# plt.subplot(7,1,7)
# plt.plot(cD1)
# plt.ylabel("delta")
# plt.draw()
# plt.show()