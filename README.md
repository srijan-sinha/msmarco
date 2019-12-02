# MS Marco
This Assigment has been made from one of the tasks of the MS Marco competition. The data required for the training and testing was located at ```/scratch/cse/phd/csz188014/IR_assig2_data/``` on HPC, IIT Delhi.
## How to Run

Create a conda environment as follows and activate it:
```
conda create --name <name> python=3.5 keras tqdm pytorch numpy pandas
conda activate <name>
```

Now in this environment the ```train.sh``` and ```rerank.sh``` scripts can be run:  
```
./train.sh  
./rerank.sh <path_to_output_text_file>
```
Make sure the training/test data is available at the above hpc location otherwise enter the correct path of the data in the scripts.  
To check the arguments of any indiidual file run:  
```python <file>.py --help```  
Once the inputs for training and test are created, the scripts can be modified to turn on input loading for faster training/testing.

