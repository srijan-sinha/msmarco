#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle as pkl
import numpy as np
import math
import sys

# In[22]:
# sys.argv usage
# 1: model name to find logits name 
# 2: output name, currently trash
# 3: number of joins
# 4-(4+sys.argv[3]-1) : id of joins
# example: to join 2 parts of output by model1_big:
# python joiner.py model1_big trash 2 0 1


def extractinfofromlogit(logits_eval):
    relevantscore = logits_eval[:,1]
#     onelabel = np.exp(logits_eval[:,1])
#     zerolabel = np.exp(logits_eval[:,0])
#     den = np.add(onelabel,zerolabel/4)
#     ans = np.divide(onelabel,den)
    return relevantscore
#     return ans


# In[3]:


def rankedfromlogit(logits_eval, qidl,aidl):
    relevantscore = extractinfofromlogit(logits_eval)
    scoremap = {}
    rankmap = {}
    for qid, aid, score in zip(qidl, aidl, relevantscore):
        if qid not in scoremap:
            scoremap[qid] = []
        scoremap[qid].append((aid,score))
    for qid, zipaidscore in scoremap.items():
        scorearr = [(-x[1]) for x in zipaidscore]
        sortedindex = np.argsort(scorearr)
        rankarr = [zipaidscore[x][0] for x in sortedindex]
        scorearrcorres = [zipaidscore[x][1] for x in sortedindex]
        rankmap[qid] = (rankarr,scorearrcorres)
    return rankmap


# In[26]:


def writeToFile(filename, rankmap):
   f = open(filename, "w")
   for qid, val in rankmap.items():
       (pidarr, scorearr) = val
       for i in range(0,len(pidarr)):
           tempstr = str(qid)+ " 0 "+str(pidarr[i])+" "+str(i+1)+" "+str(math.exp(scorearr[i]))+" s\n"
           f.write(tempstr)
   f.close()
   


# In[12]:


numjoin = int(sys.argv[3])
proxylist = [0,1]
joinlist=[]
logitsall = np.empty((0,2))
qidall = np.empty((0,1)).flatten().astype(int)
aidall = np.empty((0,1)).flatten().astype(int)
# modelproxy = 'model1_big'
modelproxy = sys.argv[1]
path = "/home/cse/btech/cs1160315/scratch/col864ass2/data/pytorch_code/"
trashoutputname = sys.argv[2]
finoutputname = 'test_data/trecinput_concat_'+(str(modelproxy))+'.txt'
for i in range(numjoin):
    joinlist.append(int(sys.argv[4+i]))
for i in joinlist:
    file = open(path+"test_data/logits_"+(str(modelproxy))+str(i)+".pkl", "rb")
    logitsthis = pkl.load(file)
    file = open(path+"test_data/qid_"+str(i)+".pkl", "rb")
    qidthis = pkl.load(file)
    print(qidthis)
    file = open(path+"test_data/aid_"+str(i)+".pkl", "rb")
    aidthis = pkl.load(file)
    print(aidthis)
    logitsall = np.append(logitsall,logitsthis, axis=0)
    print(logitsthis)
    qidall = np.append(qidall,qidthis,axis=0)
    aidall = np.append(aidall,aidthis,axis=0)


# In[6]:


# qidall


# In[7]:


# aidall


# In[8]:


# print(logitsall)


# In[24]:


rankmap = rankedfromlogit(logitsall, qidall, aidall)


# In[25]:


writeToFile(path+finoutputname, rankmap)


# In[ ]:




