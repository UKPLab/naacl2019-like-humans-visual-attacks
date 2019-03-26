import argparse
import random
import sys

import numpy as np

from perturbations_store import PerturbationsStorage

parser = argparse.ArgumentParser()
parser.add_argument('-e', action="store", dest="embed")
parser.add_argument("-p", action="store", dest="prob")
parser.add_argument('--perturbations-file', action="store", dest='perturbations_file')

parsed_args = parser.parse_args(sys.argv[1:])

emb = parsed_args.embed
prob = float(parsed_args.prob)
perturbations_file = PerturbationsStorage(parsed_args.perturbations_file)

# SAMPLE USAGE:
# python3 viper_ices.py -e ../embeddings/efile.norm -p 0.4 

from gensim.models import KeyedVectors as W2Vec

model = W2Vec.load_word2vec_format(emb)

isOdd, isEven = False, False

topn = 20

mydict = {}

for line in sys.stdin:
    a = line.split()
    wwords = []
    out_x = []
    for w in a:
        for c in w:
            if c not in mydict:
                similar = model.most_similar(c, topn=topn)
                if isOdd:
                    similar = [similar[iz] for iz in range(1, len(similar), 2)]
                elif isEven:
                    similar = [similar[iz] for iz in range(0, len(similar), 2)]
                words, probs = [x[0] for x in similar], np.array([x[1] for x in similar])
                probs /= np.sum(probs)
                mydict[c] = (words, probs)
            else:
                words, probs = mydict[c]
            r = random.random()
            if r < prob:
                s = np.random.choice(words, 1, replace=True, p=probs)[0]
                perturbations_file.add(c, s)
            else:
                s = c
            out_x.append(s)
        # out_x.append(" ")
        wwords.append("".join(out_x))
        out_x = []

    print(" ".join(wwords))

perturbations_file.maybe_write()
