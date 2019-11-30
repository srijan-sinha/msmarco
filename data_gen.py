#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-q', action="store", dest="qrel_file", help="qrel train file")
parser.add_argument('-t', action="store", dest="top1000_file", help="top1000 train file")
parser.add_argument('-d', action="store", dest="data_file_small", help="query passage file (output)")
parser.add_argument('-i', action="store", dest="data_id_file_small", help="query_id passage_id file (output)")
parser.add_argument('-a', action="store", dest="data_file_large", help="query passage file (output)")
parser.add_argument('-b', action="store", dest="data_id_file_large", help="query_id passage_id file (output)")

results = parser.parse_args()
qrel_file = results.qrel_file
top1000_file = results.top1000_file
data_file_small = results.data_file_small
data_id_file_small = results.data_id_file_small
data_file_large = results.data_file_large
data_id_file_large = results.data_id_file_large

df = pd.read_csv(qrel_file, delimiter='\t', header=None)
arr = df.values

relevant_q_d = arr[:, [0, 2]]
relevant_q_d = relevant_q_d.tolist()
relevant_q = arr[:, 0:1]
relevant_d = arr[:, 2:3]

f = open(top1000_file, 'r')

train_set = []
train_set_id = []
def add_to_train_set(arr, rel=1):
    train_set.append([rel, arr[2], arr[3]])
    train_set_id.append([rel, temp[0], temp[1]])

def write_to_file(file1, file2):
    train_set_temp = np.array(train_set)
    write = pd.DataFrame(train_set_temp)
    write.to_csv(file1, sep='\t', index=False, header=False)
    train_set_id_temp = np.array(train_set_id)
    write = pd.DataFrame(train_set_id_temp)
    write.to_csv(file2, sep='\t', index=False, header=False)

start = time()
count1 = 0
for i in range(70000000):
    datapoint = f.readline()
    temp = datapoint.strip().split('\t')
    temp[0] = int(temp[0])
    temp[1] = int(temp[1])
    if (temp[1] in relevant_d and temp[0] in relevant_q):
        if ([temp[0], temp[1]] in relevant_q_d):
            count1 += 1
            add_to_train_set(temp, rel=1)
            add_to_train_set(temp, rel=1)
        else:
            add_to_train_set(temp, rel=0)
    else:
        if (np.random.binomial(n=1,p=0.01) == 1):
            add_to_train_set(temp, rel=0)
    if ((i+1)%20000 == 0):
        print(i+1)
    if (i%500000 == 0):
        write_to_file(data_file_small, data_id_file_small)
write_to_file(data_file_large, data_id_file_large)
end = time()
print (end - start)

count = 0
for i in range(len(train_set)):
    if (train_set[i][0] == 1):
        count += 1

print ("Number of relevant examples:" + str(count))
print ("Total examples:" + str(len(train_set)))

