import pickle
import scipy.io
import pandas as pd
import numpy as np

file = "/Users/rmramesh/A/Data/bcci/dataset_BCIcomp1.mat"
# data = scipy.io.loadmat("/Users/rmramesh/A/Data/bcci/dataset_BCIcomp1.mat")

# for i in data:
#     if '__' not in i and 'readme' not in i:
#         np.savetxt(("bbcidata/" + i + ".csv"), data[i], delimiter=',')

# mat = scipy.io.loadmat('/Users/rmramesh/A/Data/bcci/dataset_BCIcomp1.mat')
# mat = {k:v for k, v in mat.items() if k[0] != '_'}
# data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
# data.to_csv("example.csv")

data = scipy.io.loadmat(file)
# f = open('file.txt', 'wb')
# f.write(data)
# f.close()

# for i in data:
#     if '__' not in i and 'readme' not in i:
#         np.savetxt("file.csv", data[i], delimiter=',')

# import json

# # as requested in comment
# # exDict = {'exDict': exDict}
#
# with open('file.txt', 'w') as file:
#      file.write(json.dumps(data)) # use `json.loads` to do the reverse

# import cPickle as pickle
#
# with open('file.txt', 'w') as file:
#      file.write(pickle.dumps(data)) # use `pickle.loads` to do the reverse


# fout = "./here.txt"
# fo = open(fout, "w")
#
# for k, v in data.items():
#     fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
#
# fo.close()
# dict_keys(['__header__', '__version__', '__globals__', 'Copyright', 'x_train', 'x_test', 'y_train'])
# value = data.get("x_train")
# value = data.get("x_test")
# value = data.get("y_train")
# print(list(value[0][0]))
def writeToFile(fp, list):
    string = ""
    for i in list:
        string = string + str(i) + ","
    fp.write(string + "\n")

# fp = open('./x_train.txt', 'a+')
#
# value = data.get("x_train")
# for i1 in value:
#     for i2 in i1:
#         writeToFile(fp, list(i2))
# fp.close()


def itemCount(list):
    count = 0
    for i in list:
        count = count + 1
    return count
#
# value = data.get("x_train")
# print("[")
# for i1 in value:
#     print("[")
#     for i2 in i1:
#         print("[" + str(itemCount(list(i2))) + "]")
#     print("]")
# print("]")


# fp = open('./y_train.txt', 'a+')
#
# value = data.get("y_train")
# for i2 in value:
#     writeToFile(fp, list(i2))
# fp.close()

# print(data.get("x_test"))
#
# fp = open('./x_test.txt', 'a+')
#
# value = data.get("x_test")
# for i1 in value:
#     for i2 in i1:
#         writeToFile(fp, list(i2))
# fp.close()
value = data.get("x_train")
# print(itemCount(list(value)))

for i in range(0, 9):
    string = ""
    for j in range(0, 128):
        # print((i*128) + j)
        string = string + str(value[(i*128) + j][0][5]) + ","
    print(string)
    print("\n")

# print(value[1][0][0])