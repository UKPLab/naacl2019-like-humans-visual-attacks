from __future__ import print_function
import os
import logging
import sys
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
from keras import backend as K
import argparse
from util.postprocessing import remove_except_last_model

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)))

##################################################



def main():
    # :: Change into the working dir of the script ::
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # :: Logging level ::
    loggingLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(loggingLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loggingLevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', type=str, default='conll2000_data/clean', help='Folder path to train,dev and test files are')
    parser.add_argument('-tokenColumnId', type=int, default=0, help='Token column id in conll file')
    parser.add_argument('-cuda_device', type=int, default=0, help='cpu:-1, others: gpu')
    parser.add_argument('-elmo_options', type=str, default='pretrained/velmo_options.json', help='ELMO options file path')
    parser.add_argument('-elmo_weights', type=str, default='pretrained/velmo_weights.hdf5', help='ELMO weights file path')
    parser.add_argument('-pkl_path', type=str, default='embeddings/velmo_cache_conll2000_data_clean.pkl', help='path to save the cache file')
    parser.add_argument('-model_save', type=str, default='models', help='path to save the model file')

    args = parser.parse_args()
    train_pos(args)


def train_pos(args):
    ######################################################
    #
    # Data preprocessing
    #
    ######################################################
    datasets = {
        args.datasetName:                                       #Name of the dataset
            {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 1 contains POS and column 2 contains chunk information using BIO encoding
             'label': 'chunk_BIO',                                    #Which column we like to predict
             'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
             'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    }


    # :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
    embeddings_file = None
    elmo_options_file = args.elmo_options
    elmo_weight_file = args.elmo_weights
    elmo_mode = 'weighted_average'
    #elmo_options_file= 'pretrained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
    #elmo_weight_file = 'pretrained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'

    cudaDevice = args.cuda_device #Which GPU to use. -1 for CPU

    embLookup = ELMoWordEmbeddings(embeddings_file, elmo_options_file, elmo_weight_file, elmo_mode, elmo_cuda_device=cudaDevice)
    # You can use a cache to precompute the ELMo embeddings once. See Create_ELMo_Cache.py for an example.
    embLookup.loadCache(args.pkl_path)

    pickleFile = perpareDataset(datasets, embLookup)

    ######################################################
    #
    # The training of the network starts here
    #
    ######################################################

    #Load the embeddings and the dataset
    mappings, data = loadDatasetPickle(pickleFile)

    # Some network hyperparameters
    params = {'classifier': ['CRF'], 'LSTM-Size': [100,100], 'dropout': (0.5, 0.5)}

    model = ELMoBiLSTM(embLookup, params)
    model.setMappings(mappings)
    model.setDataset(datasets, data)
    #model.modelSavePath = "models/[ModelName]_[Epoch].h5"
    model.modelSavePath = args.model_save + "/[ModelName]_[Epoch].h5"
    model.fit(epochs=25)


    #fpath = 'models/'+args.datasetName+'_1.h5'
    fpath = args.model_save + '/' + args.datasetName + '_1.h5'
    save_dir, model_init = os.path.split(fpath)
    print(save_dir)
    print (model_init)
    # remove trained files except from the last file
    remove_except_last_model(save_dir, model_init)

if __name__ == '__main__':
    main()