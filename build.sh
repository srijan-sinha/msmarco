mkdir -p misc
mkdir -p output
mkdir -p test_data
mkdir -p train_data
python data_gen.py -q /scratch/cse/phd/csz188014/IR_assig2_data/data-release/qrels.train.tsv -t /scratch/cse/phd/csz188014/IR_assig2_data/data_release/top1000.train.txt -d ./train_data/small_data.csv -i ./train_data/small_id.csv -a ./train_data/large_data.csv -b ./train_data/large_id.csv
python train.py -d ./data/large_data.csv -i ./data/large_id.csv -m ./misc/model -f blank -b ./misc/input_id_train.csv
