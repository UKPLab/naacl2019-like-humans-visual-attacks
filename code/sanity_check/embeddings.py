import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import argparse
import warnings
import os 
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

ELMO = "https://tfhub.dev/google/elmo/2"
NNLM = "https://tfhub.dev/google/nnlm-en-dim128/1"


def execute(tensor):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        return sess.run(tensor)

def load_elmo():
    elmo = hub.Module(ELMO, trainable=True)
    return elmo 

def embed(sentences, loaded_elmo):
    executable = loaded_elmo(
            sentences,
            signature="default",
            as_dict=True)["elmo"]# elmo, word_emb, lstm_outputs1, default
    return execute(executable)


def word_to_sentence(embeddings):
    return embeddings.mean(axis=1)


def get_embeddings_elmo(sentences,loaded_elmo):
    return word_to_sentence(embed(sentences,loaded_elmo))

if __name__ == '__main__':

    loaded_elmo = load_elmo()

    sentences = ['First sentence', 'Another']
    
    emb = embed(sentences,loaded_elmo)

    print(emb.shape)

    print(emb[:,:5])
