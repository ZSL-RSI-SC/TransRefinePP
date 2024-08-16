# import OS module
import os
import random
import csv
import scipy.io
import numpy as np
from sklearn import preprocessing


vals = []
i = 0
with open('UCM_Dataset.csv','r') as csvfile:
  csvFile = csv.reader(csvfile)
  for lines in csvFile:
    if i != 0:
      vals.append(lines[1:])
    else:
      i = 1
      
X = np.asarray(vals, dtype=np.float)
X_normalized = preprocessing.normalize(X, norm='l2')
X = np.transpose(X)
X_normalized = np.transpose(X_normalized)

mat = scipy.io.loadmat('res101.mat')

path = "path to the dataset"

dir_list = os.listdir(path)

labels = [*range(1,len(dir_list) + 1)]

label_map = []
label_map.append([])
for i in labels:
	label_map.append([])
for val in range(len(mat['labels'])):
	label_map[int(mat['labels'][val])].append(val+1)

#split
np.random.seed(2039)	#2014 2015 2016 2017 2018 2019 give seed number for every new split
np.random.shuffle(labels)
test_seen = 16 # number of seen classes
test_unseen = len(dir_list) - test_seen
test_unseen = labels[test_seen:]
test_seen = labels[0:test_seen]

#train-test split
train = 0.8
validate = 0.8
test = 1 - train

A = []			#Train
B = []			#Test_Seen
V = []			#Validate
for i in test_seen:
	A.extend(label_map[i][0:int(validate*train*len(label_map[i]))])
	V.extend(label_map[i][int(validate*train*len(label_map[i])):int(train*len(label_map[i]))])
	B.extend(label_map[i][int(train*len(label_map[i])):])
		
C = []
for i in test_unseen:
	C.extend(label_map[i])

np.random.shuffle(A)
np.random.shuffle(B)
np.random.shuffle(C)
np.random.shuffle(V)

A = np.reshape(A, (-1, 1))
B = np.reshape(B, (-1, 1))
C = np.reshape(C, (-1, 1))
V = np.reshape(V, (-1, 1))

obj_arr = np.zeros((6,), dtype=np.object)
obj_arr[0] = A
obj_arr[1] = B
obj_arr[2] = C
obj_arr[3] = X
obj_arr[4] = X_normalized
obj_arr[5] = V


scipy.io.savemat('att_splits.mat', mdict={'train_loc': obj_arr[0],'val_loc': obj_arr[5], 'test_seen_loc': obj_arr[1], 'test_unseen_loc': obj_arr[2], 'original_att': obj_arr[3], 'att': obj_arr[4]})
