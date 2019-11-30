#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('-e', action="store", dest="top1000_eval_file", help="top1000 eval file")
parser.add_argument('-t', action="store", dest="folder", help="folder to generate part wise data in")

results = parser.parse_args()
top1000_eval_file = results.top1000_eval_file
folder = results.folder

df = pd.read_csv(top1000_eval_file, delimiter='\t', header=None)
eval_data = df.values
eval_data = eval_data[eval_data[:,0].argsort()]

parts = []
parts.append(eval_data[0:953075,:].copy())
parts.append(eval_data[953075:1906251,:].copy())
parts.append(eval_data[1906251:2859083,:].copy())
parts.append(eval_data[2859083:3811380,:].copy())
parts.append(eval_data[3811380:4763568,:].copy())
parts.append(eval_data[4763568:5716818,:].copy())
parts.append(eval_data[5716818:,:].copy())
# parts.append(eval_data[0:18841:,:].copy())

for i in range(len(parts)):
    q_d = parts[i][:,2:]
    write = pd.DataFrame(q_d)
    file = folder + "/part" + str(i) + ".query.doc.csv"
    write.to_csv(file, sep='\t', index=False, header=False)
    
    q_d_id = parts[i][:,0:2]
    write = pd.DataFrame(q_d_id)
    file = folder + "/id.part" + str(i) + ".query.doc.csv"
    write.to_csv(file, sep='\t', index=False, header=False)

