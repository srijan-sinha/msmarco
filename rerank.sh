#!/bin/bash

# $1 output file

mkdir -p ./trec

python ./code/data_test.py -e /scratch/cse/phd/csz188014/IR_assig2_data/data_release/top1000.eval.txt -t ./test_data/

# Execute these 7 commands parallely on different hoc jobs

python ./code/eval_single.py -t ./test_data/ -p 0 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 1 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 2 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 3 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 4 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 5 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec
python ./code/eval_single.py -t ./test_data/ -p 6 -e -i ./misc/ -r ./misc -m ./misc/model.pkl -n model -o ./trec

touch $4
cat ./trec/trecinput_model_0.txt >> $1
cat ./trec/trecinput_model_1.txt >> $1
cat ./trec/trecinput_model_2.txt >> $1
cat ./trec/trecinput_model_3.txt >> $1
cat ./trec/trecinput_model_4.txt >> $1
cat ./trec/trecinput_model_5.txt >> $1
cat ./trec/trecinput_model_6.txt >> $1
