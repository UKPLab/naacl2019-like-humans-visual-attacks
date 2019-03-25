import gc
import sys

import h5py
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

test_embeddings_path = sys.argv[1]
model_path = sys.argv[2]
result_path = sys.argv[3]
labels_path = sys.argv[4]


def read_data(embeddings_path, labels_path):
    with h5py.File(embeddings_path, 'r') as f:
        embeddings = np.vstack([f[str(i)] for i in range(len(f))])
    with open(labels_path, 'r') as f:
        labels = np.vstack([np.fromstring(l.strip(), dtype=np.float64, sep=',') for l in f])
    return embeddings, labels


model = Sequential([
    Dense(512, input_shape=(512,)),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(6),
    Activation('sigmoid'),
])

model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
model.load_weights(model_path)

test_embeddings, test_labels = read_data(test_embeddings_path, labels_path)
predictions = model.predict(test_embeddings, batch_size=512)

outlog = open(result_path,'w')
auc_roc = roc_auc_score(test_labels,predictions)
outlog.write('{}\n)'.format(auc_roc))
outlog.close()

outlog = open(result_path+'.predictions','w')
outlog.write('\t'.join(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])+'\n')
for pred in predictions:
    outlog.write(','.join([str(p) for p in pred])  + '\n' )
outlog.close()






