import pyedflib
import numpy as np

f = pyedflib.EdfReader("/Users/rmramesh/Downloads/S001R03.edf")
n = f.signals_in_file

signal_labels = f.getSignalLabels()
print(len(signal_labels))

sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)

print(sigbufs.shape)