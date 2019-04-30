### Training/Evaluating CoNLL-2000 POS tagging and Chunking Code 

This repository is a configuration of [ELMO-BiLSTM-CNN-CRF implementation](https://github.com/UKPLab/elmo-bilstm-cnn-crf/).


Before you can training the models, you need:
Original CoNLL-2000 shared task data, perturbed data (See VIPER); and
Pretrained ELMO and Visual ELMO embeddings (See AllenNLP_Modifications);

It is advised to create cache for the embeddings before you run the experiments:
```
         python Create_ELMo_Cache.py -datasetName $DATASET \
		-tokenColumnId 0 \
		-cuda_device 0 \
		-elmo_options 'pretrained/elmo_options.json' \
		-elmo_weights 'pretrained/elmo_weights.hdf5' \
		-pkl_path '$SAVEDIR/cached.pkl'
```
Then to train a Part-Of-Speech Tagger:
```
         python Train_POS.py -datasetName $DATASET \
		-tokenColumnId 0 \
		-cuda_device 1 \
	        -elmo_options 'pretrained/elmo_options.json' \
	        -elmo_weights 'pretrained/elmo_weights.hdf5' \
		-model_save $MODEL \
		-pkl_path '$SAVEDIR/cached.pkl'
```
And evaluate it with EvalModel_CoNLL_Format.py, where 'testSetting' is the prefix for the results file for easier analysis.
```
         python EvalModel_CoNLL_Format.py -datasetName $DATASET \
		-testFile '$TESTFILE' \
		-testSetting 'ATP01_org_'$id'.txt' \
		-model_save $MODEL \
		-result_save $RESULT \
		-cuda_device 0 \
		-task 'pos'
```
Similarly, to train a Chunker after caching:
```
	  python Train_Chunking.py -datasetName $DATASET \
		-tokenColumnId 0 \
		-cuda_device 0 \
	        -elmo_options 'pretrained/elmo_options.json' \
	        -elmo_weights 'pretrained/elmo_weights.hdf5' \
		-model_save $MODEL \
		-pkl_path 'embeddings/cached.pkl'
```
And to evaluate the chunker:
```
	  python EvalModel_CoNLL_Format.py -datasetName $DATASET \
		-testFile '$TESTFILE' \
		-testSetting 'CTP01_simple_'$id'.txt' \
		-model_save $MODEL \
		-result_save $RESULT \
		-cuda_device 0 \
		-task 'chunking'
```
For convenience, we also provide two example scripts, that creates the cache, trains the models and evaluates them on various perturbed test data. $DATASET refers to the training/development data directory, and $MODEL is the output folder to save the trained models. The pretrained embeddings should be supplied under the 'pretrained' folder. Finally, the perturbed test data should be saved with the pattern: 'data/conll2000_data/perturbed/%ratio/%filename' to run the scripts. 
