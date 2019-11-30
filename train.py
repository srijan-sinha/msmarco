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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', action="store", dest="data_file", help="query passage file")
parser.add_argument('-i', action="store", dest="data_id_file", help="query_id passage_id file")
parser.add_argument('-m', action="store", dest="model_file", help="model file without extension")
parser.add_argument('-l', action="store_true", dest="load_input", help="turn on to load input ids")
parser.add_argument('-f', action="store", dest="load_input_file", help="file to load input ids from")
parser.add_argument('-b', action="store", dest="store_input_file", help="where to store input if not loading")

results = parser.parse_args()
data_file = results.data_file
data_id_file = results.data_id_file
model_file = results.model_file
load_input = results.load_input
load_input_file = results.load_input_file
store_input_file = results.store_input_file

df = pd.read_csv(data_file, delimiter='\t', header=None)
df_id = pd.read_csv(data_id_file, delimiter='\t', header=None)
train_data = df.values
train_id = df_id.values

labels = train_data[:,0:1]
queries = train_data[:,1:2].flatten()
passages = train_data[:,2:3].flatten()
query_ids = train_id[:,1:2].flatten()
passsage_ids = train_id[:,2:3].flatten()

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
    file = open(load_input_file, "rb")
    input_id = pkl.load(file)
else:
    input_id_query = customtok(MAX_LENGTH_QUERY, queries, 0)
    input_id_passage = customtok(MAX_LENGTH_PASSAGE, passages, 1)
    input_id_query = np.array(input_id_query)
    input_id_passage = np.array(input_id_passage)
    input_id = np.concatenate((input_id_query, input_id_passage), axis=1)
    file = open(store_input_file, 'wb')
    pkl.dump(input_id, file)
    file.close()

attention_mask = input_id.copy()
attention_mask[attention_mask > 0] = 1

segment1 = np.zeros((len(input_id), MAX_LENGTH_QUERY))
segment2 = input_id[:,MAX_LENGTH_QUERY:].copy()
segment2[segment2 > 0] = 1
segment_mask = np.concatenate((segment1, segment2), axis=1)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_id, labels.astype(int), 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, input_id,
                                             random_state=2018, test_size=0.1)
train_segments, validation_segments, _, _ = train_test_split(segment_mask, input_id,
                                             random_state=2018, test_size=0.1)

train_inputs = torch.tensor(train_inputs, dtype=torch.long)
validation_inputs = torch.tensor(validation_inputs, dtype=torch.long)
train_labels = torch.tensor(train_labels, dtype=torch.long)
validation_labels = torch.tensor(validation_labels, dtype=torch.long)
train_masks = torch.tensor(train_masks, dtype=torch.long)
validation_masks = torch.tensor(validation_masks, dtype=torch.long)
train_segments = torch.tensor(train_segments, dtype=torch.long)
validation_segments = torch.tensor(validation_segments, dtype=torch.long)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_segments, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_segments, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

train_loss_set = []

epochs = 4

epochctr = 0
for _ in trange(epochs, desc="Epoch"):
  
  
    model.train()
    epochctr = epochctr +1
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
  
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch
        optimizer.zero_grad()
        outputstemp= model(b_input_ids,token_type_ids=b_input_segment, attention_mask=b_input_mask, labels=b_labels)
        loss = outputstemp[0]
        train_loss_set.append(loss.item())    
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_input_segment, b_labels = batch
        with torch.no_grad():
            (logits,) = model(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_input_segment)
    
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    file = open(model_file+str(epochctr)+".pkl", "wb")
    pkl.dump(model, file)
    file.close()

