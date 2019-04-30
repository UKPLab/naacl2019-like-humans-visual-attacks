from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
from util.CoNLL import readCoNLL
import os
import sys
import logging
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', type=str, default='conll2000_data/perturbed/03', help='Folder path to train,dev and test files are')
    parser.add_argument('-tokenColumnId', type=int, default=0, help='Token column id in conll file')
    parser.add_argument('-cuda_device', type=int, default=0, help='cpu:-1, others: gpu')
    parser.add_argument('-elmo_options', type=str, default='pretrained/velmo_options.json', help='ELMO options file path')
    parser.add_argument('-elmo_weights', type=str, default='pretrained/velmo_weights.hdf5', help='ELMO weights file path')
    parser.add_argument('-pkl_path', type=str, default='embeddings/velmo_cache_conll2000_data_clean.pkl', help='path to save the cache file')
    args = parser.parse_args()
    create_cache(args)


def create_cache(args):
    datasetName = args.datasetName
    tokenColId = args.tokenColumnId
    cudaDevice = args.cuda_device
    elmo_options_file = args.elmo_options
    elmo_weight_file = args.elmo_weights

    #elmo_options_file= 'pretrained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
    #elmo_weight_file = 'pretrained/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
    #elmo_options_file= 'pretrained/velmo_options.json'
    #elmo_weight_file = 'pretrained/velmo_weights.hdf5'

    # :: Logging level ::
    loggingLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(loggingLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loggingLevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    commentSymbol = None
    columns = {tokenColId: 'tokens'}

    #picklePath = "embeddings/elmo_cache_" + datasetName + ".pkl"
    #picklePath = "embeddings/velmo_cache_conll2000_data_perturbed_03.pkl"
    #picklePath = "embeddings/velmo_cache_conll2000_data_clean.pkl"
    picklePath = args.pkl_path
    embLookup = ELMoWordEmbeddings(None, elmo_options_file, elmo_weight_file, elmo_cuda_device=cudaDevice)

    print("ELMo Cache Generation")
    print("Output file:", picklePath)
    print("CUDA Device:", cudaDevice)

    splitFiles = ['train.txt', 'dev.txt', 'test.txt']

    for splitFile in splitFiles:
        inputPath = os.path.join('data', datasetName, splitFile)

        print("Adding file to cache: "+inputPath)
        sentences = readCoNLL(inputPath, columns, commentSymbol)
        tokens = [sentence['tokens'] for sentence in sentences]

        start_time = time.time()
        embLookup.addToCache(tokens)
        end_time = time.time()
        print("%s processed in %.1f seconds" % (splitFile, end_time - start_time))
        print("\n---\n")

    print("Store file at:", picklePath)
    embLookup.storeCache(picklePath)


if __name__ == '__main__':
    main()