# Training

1. Install requirements and get `G2P-my_LSTM-act1_save_new_general.py` to run locally using python3

2. Generate train,dev,test data and place it in a subfolder in `data` (say you call this MYDATA). Two sample folders are given in `data`.
   - See also `../G2P_data/` for (more) sample train,dev,test splits (e.g. train.5k.0, dev.2k.0_5, test.1k.0). **NB** All data must be in tab seperated format like train.5k.0, dev.2k.0_5. If necessary, run `python3 renderTest.py < ../G2P_data/test.1k.0` to convert your test data in the same format.

3. In `G2P-my_LSTM-act1_save_new_general.py` and `neuralnets/BiLSTM_proper.py` modify paths indicated by "MODIFY" to your local settings.

4. Run `G2P-my_LSTM-act1_save_new_general.py` with some arguments. Arguments include hyperparameters, but also 
   - the path to the train,dev,test split you want to train (indicated by MYDATA) 
   - the embeddings you want to use for training (make a new copy of embeddings for each run you make in case your train,dev,test splits differ). 

   Sample runs are given in `allvsall_odd`.

# Evaluation

1. Outputs are stored in the path indicated in `neuralnets/BiLSTM_proper.py` under "MODIFY". 

2. Run `Eval/eval2.py` using the respective output files as arguments.
