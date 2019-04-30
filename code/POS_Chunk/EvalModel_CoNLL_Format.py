#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
import nltk
import sys
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation, addEmbeddings, readCoNLL
from neuralnets.ELMoBiLSTM import ELMoBiLSTM
from neuralnets.ELMoWordEmbeddings import ELMoWordEmbeddings
import argparse
from util.postprocessing import get_last_model_path
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', type=str, default='conll2000_data/clean', help='necessary to find the model path')
    parser.add_argument('-testFile', type=str, default='data/conll2000_data/clean/test.txt', help='conll file to test')
    parser.add_argument('-testSetting', type=str, default='CT1.txt', help='conll file to test')
    parser.add_argument('-model_save', type=str, default='models', help='path to save the model file')
    parser.add_argument('-result_save', type=str, default='results', help='path to save the results file')
    parser.add_argument('-cuda_device', type=int, default=0, help='cpu:-1, others: gpu')
    parser.add_argument('-task', type=str, default='pos', help='pos|chunking')
    args = parser.parse_args()
    evaluate(args)


def evaluate(args):
    fpath = args.model_save + '/' + args.datasetName + '_1.h5'
    #fpath = 'models/'+args.datasetName+'_1.h5'
    save_dir, model_init = os.path.split(fpath)

    modelPath, _ = get_last_model_path(save_dir, model_init)
    print(modelPath)
    inputPath = args.testFile
    inputColumns = {0: "tokens", 1:'POS', 2:'chunk_BIO'}

    resfpath = args.result_save +'/'+args.task+'/'+args.testSetting
    resfile = open(resfpath, 'w')

    # :: Load the model ::
    lstmModel = ELMoBiLSTM.loadModel(modelPath)

    # :: Prepare the input ::
    sentences = readCoNLL(inputPath, inputColumns)
    addCharInformation(sentences)
    addCasingInformation(sentences)

    # :: Map casing and character information to integer indices ::
    dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

    # :: Perform the word embedding / ELMo embedding lookup ::
    embLookup = lstmModel.embeddingsLookup
    embLookup.elmo_cuda_device = 0                          #Cuda device for pytorch - elmo embedding, -1 for CPU
    addEmbeddings(dataMatrix, embLookup.sentenceLookup)

    if(args.task=="pos"):
        # Evaluation of POS tagging
        test_acc = lstmModel.computeAcc(args.datasetName, dataMatrix)
        print("Test-Data: Accuracy: %.4f" % (test_acc))
        resfile.write("Test-Data: Accuracy: %.4f" % (test_acc))
    elif(args.task=="chunking"):
        # Evaluation of Chunking
        test_pre, test_rec, test_f1 = lstmModel.computeF1(args.datasetName, dataMatrix)
        print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        resfile.write("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))

    resfile.close()

if __name__ == '__main__':
    main()