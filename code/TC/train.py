import gc
import sys

import h5py
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)))

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="2" 

train_embeddings_path = sys.argv[1]
dev_embeddings_path = sys.argv[2]
test_embeddings_path = sys.argv[3]
model_path = sys.argv[4]
random_seed = int(sys.argv[5])
train_labels_path = sys.argv[6]
dev_labels_path = sys.argv[7]
test_labels_path = sys.argv[8]

np.random.seed(random_seed)


def read_data(embeddings_path, labels_path):
    with h5py.File(embeddings_path, 'r') as f:
        embeddings = np.vstack([f[str(i)] for i in range(len(f))])
    with open(labels_path, 'r') as f:
        labels = np.vstack([np.fromstring(l.strip(), dtype=np.int32, sep=',') for l in f])
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

train_embeddings, train_labels = read_data(train_embeddings_path, train_labels_path)
dev_embeddings, dev_labels = read_data(dev_embeddings_path, dev_labels_path)

earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)
saver = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True,
                        mode='auto')

model.fit(train_embeddings, train_labels, validation_data=(dev_embeddings, dev_labels), epochs=100, batch_size=128,
          callbacks=[saver, earlystop])

train_embeddings = None
dev_embeddings = None
gc.collect()

test_embeddings, test_labels = read_data(test_embeddings_path, test_labels_path)
predictions = model.predict(test_embeddings, batch_size=512)

# todo evaluate with official ROC AUC metric
print('Final scores')
print('-' * 10)
for i, category in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    score = accuracy_score(np.round(predictions[:, i]), test_labels[:, i])
    zero_baseline = accuracy_score(test_labels[:, i] * 0, test_labels[:, i])
    print('{} = {} (zero baseline = {})'.format(category, score, zero_baseline))
