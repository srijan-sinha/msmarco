# $2 query file
# $4 output file
# $6 top1000.eval.txt
python data_test.py -e /scratch/cse/phd/csz188014/IR_assig2_data/data_release/top1000.eval.txt -t ./test_data/

python eval_single ./test_data/ 0 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 1 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 2 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 3 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 4 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 5 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec
python eval_single ./test_data/ 6 -e -i ./misc/ -r ./misc -m ./misc/model1.pkl -m model -o ./trec

touch $4
cat ./trec/trecinput_model_0.txt >> $4
cat ./trec/trecinput_model_1.txt >> $4
cat ./trec/trecinput_model_2.txt >> $4
cat ./trec/trecinput_model_3.txt >> $4
cat ./trec/trecinput_model_4.txt >> $4
cat ./trec/trecinput_model_5.txt >> $4
cat ./trec/trecinput_model_6.txt >> $4
