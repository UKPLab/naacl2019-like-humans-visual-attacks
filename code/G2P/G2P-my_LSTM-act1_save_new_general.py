# SAMPLE USAGE:
# python3 PE-my.py lrelu030 144 RMSprop 2 0.15034505523909153 0.000969893080861072 glorot_uniform 0.8 


from __future__ import print_function

import logging
import os
import sys,random
from keras import initializers

import numpy as np

from neuralnets.BiLSTM_proper import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
from handleHyper import getInitializer

try:
  epochs = int(sys.argv[11])
except IndexError:
  epochs = 50

f=sys.argv[1]
size=[int(sys.argv[2])]
opt=sys.argv[3]
l = int(sys.argv[4])
d = float(sys.argv[5])
learning_rate = float( sys.argv[6] )
recurrent_init = sys.argv[7] # "orthogonal"
#recurrent_initializer = initializers.RandomNormal() 
datasize=float( sys.argv[8] )
efile=sys.argv[9]
datasetName = sys.argv[10]



recurrent_initializer,optimizer,function,maxout_k = getInitializer(recurrent_init,learning_rate,opt,f)

num_repeats=1

g = f.split("-")
if len(g)==1: f,act_flag = g[0],"None"
elif len(g)==2:
  f,act_flag = g

optimizers = ['adam', 'adadelta']
dropout_rates = [0.2, 0.5]
layers = [1, 2, 3]

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

######################################################
#
# Data preprocessing
#
######################################################


# :: Train / Dev / Test-Files ::
if datasetName.endswith("/"):
  datasetName = datasetName[:-1]
dataColumns = {1: 'tokens', 2: 'POS'}  # Tab separated columns, column 1 contains the token, 3 the universal POS tag
labelKey = 'POS'

embeddingsPath = efile  
word_embeddings = embeddingsPath.split("/")[-1]

# Parameters of the network
#recurrent_initializer = initializers.RandomNormal()
params = {'dropout': [0.25, 0.25], 'classifier': 'CRF', 'LSTM-Size': [100], 'optimizer': 'nadam',
          'charEmbeddings': None, 'miniBatchSize': 32, 'activation': 'tanh', 'init': recurrent_initializer, "learning_rate": learning_rate, "activation_flag":act_flag}

frequencyThresholdUnknownTokens = 50  # If a token that is not in the pre-trained embeddings file appears at least 50 times in the train.txt, then a new embedding is generated for this word

datasetFiles = [
    (datasetName, dataColumns),
]

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasetFiles)

######################################################
#
# The training of the network starts here
#
######################################################

# Load the embeddings and the dataset
embeddings, word2Idx, datasets = loadDatasetPickle(pickleFile)
data = datasets[datasetName]
mylen = int(len(data["trainMatrix"])*datasize)
random.seed(42)
trainM = data["trainMatrix"]
random.shuffle(trainM) 
x = trainM[mylen:]
data['trainMatrix'] = trainM[:mylen]
data['devMatrix'] = data['devMatrix']
#print(len(data["testMatrix"])); sys.exit(1)
data['testMatrix'] = data['testMatrix'] #+ x


print("Dataset:", datasetName)
print(data['mappings'].keys())
print("Label key: ", labelKey)
print("Train Sentences:", len(data['trainMatrix']))
print("Dev Sentences:", len(data['devMatrix']))
print("Test Sentences:", len(data['testMatrix']))

# MODIFY this directory path to your local setting
results_dir = '/work/scratch/se55gyhe/Act_func/seq_tag/G2P_small/results-lstm-act1'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_dir += "/"+f
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# Name of results file
f1_name = '__'.join([datasetName, word_embeddings, recurrent_init, opt, str(l), str(d), opt, str(size[0]), str(learning_rate), str(datasize), act_flag, 'f1.csv'])
print(datasetName)
print(f1_name)
f1_file = open(results_dir + '/' + f1_name, 'w')
params['output'] = f1_file

params['optimizer'] = opt
params['layers'] = l
params['dropout'] = [d, d]
params['init'] = recurrent_initializer

f1_file.write('\noptimizer: ' + opt + ' -layers: ' + str(l) + ' -dropout: ' + str(d))

params['activation'] = f
f1_file.write('\n____________________________\n')
f1_file.write('function: ' + f)
f1_file.write('\n____________________________\n')
f1_file.write('Units,1,2,3,4,5,max,min,average,standard deviation')  # average results after

f1_file.write('\n' + str(size[0]) + ',')
params['LSTM-Size'] = size
params['scores'] = []
params["dev_scores"] = []
params["earlyStopping"] = 15

for i in range(num_repeats):
                        model = BiLSTM(params)
                        model.setMappings(embeddings, data['mappings'])
                        model.setTrainDataset(data, labelKey)
                        model.verboseBuild = True
                        # MODIFY to your local setting
                        model.modelSavePath = "/work/scratch/se55gyhe/models/G2P_[DevScore]_[TestScore]_[Epoch]_%d.h5"%i
                        model.writeOutput=True
                        if sys.argv[10].endswith("/"): flag = sys.argv[10][:-1]
                        else: flag=sys.argv[10]
                        model.flag=flag
                        model.evaluate(epochs)

max = np.max(params['scores'])
min = np.min(params['scores'])
average = np.average(params['scores'])
stddev = np.std(params['scores']) * 100
print("length of scores ", len(params['scores']))
params['scores'] += [max, min, average, stddev]

for s in params['scores']:
                        f1_file.write(str(s) + ',')

f1_file.write("\t"+",".join([str(x) for x in params["dev_scores"]]))


f1_file.flush()
os.fsync(f1_file)
