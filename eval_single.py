#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_transformers import AdamW as BertAdam
from tqdm import tqdm, trange
import pickle as pkl
import math
# import sys
import argparse

# sys.argv: 1 is part number
#         2 is model file 
#         3 is placeholder for logitsfile
#         4 is placeholder for trecinput

parser = argparse.ArgumentParser()
parser.add_argument('-t', action="store", dest="test_folder", help="folder containing test data")
parser.add_argument('-p', action="store", dest="part", help="part number")
parser.add_argument('-l', action="store_true", dest="load_input", help="whether to load input or not")
parser.add_argument('-e', action="store_true", dest="evaluate_input", help="will evaluate and rank when on, will only rank when off")
parser.add_argument('-i', action="store", dest="input_id_folder", help="folder to load input from")
parser.add_argument('-r', action="store", dest="logits_folder", help="folder to load/store logits from, load when -e off, store otherwise")
parser.add_argument('-m', action="store", dest="model_file", help="file to load model from")
parser.add_argument('-n', action="store", dest="model_name", help="name of model, logits is named based on this")
parser.add_argument('-o', action="store", dest="trec_folder", help="folder to generate trecinput in")

results = parser.parse_args()
test_folder = results.test_folder
part = results.part
load_input = results.load_input
evaluate_input = results.evaluate_input
input_id_folder = results.input_id_folder
logits_folder = results.logits_folder
model_file = results.model_file
model_name = results.model_name

query_doc_file = test_folder + '/part' + part + '.query.doc.csv'
query_doc_id_file = test_folder + '/id.part' + part + '.query.doc.csv'
input_id_file = input_id_folder + '/eval_input_id_' + part + '.pkl'
logits_file = logits_folder +'/logits_' + model_name + part + '.pkl'
trec_input_file = trec_folder + '/trecinput_' + model_name + '_' + part + '.txt'

df = pd.read_csv(query_doc_file, delimiter='\t', header=None)
train_data = df.values
df_id = pd.read_csv(query_doc_id_file, delimiter='\t', header=None)
train_id = df_id.values

queries = train_data[:,0:1].flatten()
passages = train_data[:,1:2].flatten()
query_ids = train_id[:,0:1].flatten()
passsage_ids = train_id[:,1:2].flatten()

if (evaluate_input):

  file = open(model_file,"rb")
  model = pkl.load(file)
  file.close()

  model.cuda()


  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  def customtok(max_len, texts, sentid):
      if(sentid==0):
          texts = ["[CLS] " + text + " [SEP]" for text in texts]
      else:
          texts = [text + " [SEP]" for text in texts]
      tokenized_texts = [tokenizer.tokenize(text) for text in texts]
      templ = [tokenizer.convert_tokens_to_ids(text) for text in tokenized_texts]
      input_ids = pad_sequences(templ, maxlen=max_len, dtype="long", truncating="post", padding="post")
      return input_ids

  MAX_LENGTH_QUERY = 32
  MAX_LENGTH_PASSAGE = 96

  if (load_input):
    file = open(input_id_file, "rb")
    input_id = pkl.load(file)
  else:
    input_id_query = customtok(MAX_LENGTH_QUERY, queries, 0)
    input_id_passage = customtok(MAX_LENGTH_PASSAGE, passages, 1)

    input_id_query = np.array(input_id_query)
    input_id_passage = np.array(input_id_passage)
    input_id = np.concatenate((input_id_query, input_id_passage), axis=1)

    file = open(input_id_file, "wb")
    pkl.dump(input_id, file)
    file.close()

  attention_mask = input_id.copy()
  attention_mask[attention_mask > 0] = 1

  segment1 = np.zeros((len(input_id), MAX_LENGTH_QUERY))
  segment2 = input_id[:,MAX_LENGTH_QUERY:].copy()
  segment2[segment2 > 0] = 1
  segment_mask = np.concatenate((segment1, segment2), axis=1)

  eval_inputs = input_id
  eval_masks = attention_mask
  eval_segments = segment_mask

  eval_inputs = torch.tensor(eval_inputs, dtype=torch.long)
  eval_masks = torch.tensor(eval_masks, dtype=torch.long)
  eval_segments = torch.tensor(eval_segments, dtype=torch.long)

  batch_size = 1024

  eval_data = TensorDataset(eval_inputs, eval_masks, eval_segments)
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

  torch.cuda.get_device_name(0)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  torch.cuda.get_device_name(0)

  model.eval()

  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  logitsall = np.empty((0,2))
  for batch in eval_dataloader:
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_mask, b_input_segment = batch
      with torch.no_grad():
          (logits,) = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_input_segment)

      logits = logits.detach().cpu().numpy()
      logitsall = np.append(logitsall,logits, axis=0)


      nb_eval_steps += 1
      print(nb_eval_steps*batch_size)

  # file = open("./test_data/logits_"+(str(sys.argv[3]))+str(part)+".pkl", "wb")
  file = open(logits_file, "wb")
  pkl.dump(logitsall, file)
  file.close()

else:
  file = open(logits_file,"rb")
  model = pkl.load(file)
  file.close()

def extractinfofromlogit(logits_eval):
    relevantscore = logits_eval[:,1]
    return relevantscore

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

def writeToFile(filename, rankmap):
   f = open(filename, "w")
   for qid, val in rankmap.items():
       (pidarr, scorearr) = val
       for i in range(0,len(pidarr)):
           tempstr = str(qid)+ " 0 "+str(pidarr[i])+" "+str(i+1)+" "+str(math.exp(scorearr[i]))+" s\n"
           f.write(tempstr)
   f.close()
   
rankmap = rankedfromlogit(logitsall, query_ids, passsage_ids)

writeToFile(trec_input_file, rankmap)

