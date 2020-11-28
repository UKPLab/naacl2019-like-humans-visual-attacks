'''
 Use an external lookup method to disturb some text. In this case, we take the textual descriptions of each character,
 and find the nearest neighbours in the list of unicode characters by finding characters with the largest number of
 matching tokens in the text description.

 Example usage:

 python3 viper_dces.py -p 0.4 -d ../G2P_data/train.1k --conll --odd

'''

import argparse
import random
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from perturbations_store import PerturbationsStorage

def read_data_conll(fn):

 docs=[]
 doc=[]

 for line in open(fn):
  line = line.strip()
  if line=="":
    if doc!=[]:
      a,b=[],[]
      for l in doc: 
        _,x,y = l.split("\t")
        a.append(x)
        b.append(y)
      docs.append("{}\t{}".format(" ".join(a)," ".join(b)))
    doc=[]
  else:
    doc.append(line)

 if doc!=[]: 
  a,b=[],[]
  for l in doc:
        _,x,y = l.split("\t")
        a.append(x)
        b.append(y)
  docs.append("{}\t{}".format(" ".join(a)," ".join(b)))

 return docs

def read_data_standard(fn):
  docs=[]
  for line in open(fn):
    docs.append(line)
  return docs  

# load the unicode descriptions into a single dataframe with the chars as indices
descs = pd.read_csv('NamesList.txt', skiprows=np.arange(16), header=None, names=['code', 'description'], delimiter='\t')
descs = descs.dropna(0)
descs_arr = descs.values # remove the rows after the descriptions
vectorizer = CountVectorizer(max_features=1000)
desc_vecs = vectorizer.fit_transform(descs_arr[:, 0]).astype(float)
vecsize = desc_vecs.shape[1]
vec_colnames = np.arange(vecsize)
desc_vecs = pd.DataFrame(desc_vecs.todense(), index=descs.index, columns=vec_colnames)
descs = pd.concat([descs, desc_vecs], axis=1)

def char_to_hex_string(ch):
    return '{:04x}'.format(ord(ch)).upper()


disallowed = ['TAG', 'MALAYALAM', 'BAMUM', 'HIRAGANA', 'RUNIC', 'TAI', 'SUNDANESE', 'BATAK', 'LEPCHA', 'CHAM',
              'TELUGU', 'DEVANGARAI', 'BUGINESE', 'MYANMAR', 'LINEAR', 'SYLOTI', 'PHAGS-PA', 'CHEROKEE',
              'CANADIAN', 'YI', 'LYCIAN', 'HANGUL', 'KATAKANA', 'JAVANESE', 'ARABIC', 'KANNADA', 'BUHID',
              'TAGBANWA', 'DESERET', 'REJANG', 'BOPOMOFO', 'PERMIC', 'OSAGE', 'TAGALOG', 'MEETEI', 'CARIAN', 
              'UGARITIC', 'ORIYA', 'ELBASAN', 'CYPRIOT', 'HANUNOO', 'GUJARATI', 'LYDIAN', 'MONGOLIAN', 'AVESTAN',
              'MEROITIC', 'KHAROSHTHI', 'HUNGARIAN', 'KHUDAWADI', 'ETHIOPIC', 'PERSIAN', 'OSMANYA', 'ELBASAN',
              'TIBETAN', 'BENGALI', 'TURKIC', 'THROWING', 'HANIFI', 'BRAHMI', 'KAITHI', 'LIMBU', 'LAO', 'CHAKMA',
              'DEVANAGARI', 'ITALIC', 'CJK', 'MEDEFAIDRIN', 'DIAMOND', 'SAURASHTRA', 'ADLAM', 'DUPLOYAN'
             ]

disallowed_codes = ['1F1A4', 'A7AF']


# function for retrieving the variations of a character
def get_all_variations(ch):
       
    # get unicode number for c
    c = char_to_hex_string(ch)
    
    # problem: latin small characters seem to be missing?
    if np.any(descs['code'] == c):
        description = descs['description'][descs['code'] == c].values[0]
    else:
        print('Failed to disturb %s, with code %s' % (ch, c))
        return c, np.array([])
    
    # strip away everything that is generic wording, e.g. all words with > 1 character in
    toks = description.split(' ')

    case = 'unknown'

    identifiers = []
    for tok in toks:
           
        if len(tok) == 1:
            identifiers.append(tok)
            
            # for debugging 
            if len(identifiers) > 1:
                print('Found multiple ids: ')
                print(identifiers)

        elif tok == 'SMALL':
            case = 'SMALL'
        elif tok == 'CAPITAL':
            case = 'CAPITAL'

    # for debugging
    #if case == 'unknown':
    #    sys.stderr.write('Unknown case:')
    #    sys.stderr.write("{}\n".format(toks))

    # find matching chars
    matches = []
    
    for i in identifiers:        
        for idx in descs.index:
            desc_toks = descs['description'][idx].split(' ')
            if i in desc_toks and not np.any(np.in1d(desc_toks, disallowed)) and \
                    not np.any(np.in1d(descs['code'][idx], disallowed_codes)) and \
                    not int(descs['code'][idx], 16) > 30000:

                # get the first case descriptor in the description
                desc_toks = np.array(desc_toks)
                case_descriptor = desc_toks[ (desc_toks == 'SMALL') | (desc_toks == 'CAPITAL') ]

                if len(case_descriptor) > 1:
                    case_descriptor = case_descriptor[0]
                elif len(case_descriptor) == 0:
                    case = 'unknown'

                if case == 'unknown' or case == case_descriptor:
                    matches.append(idx)

    # check the capitalisation of the chars
    return c, np.array(matches)

# function for finding the nearest neighbours of a given word
def get_unicode_desc_nn(c, perturbations_file, topn=1):
    # we need to consider only variations of the same letter -- get those first, then apply NN
    c, matches = get_all_variations(c)
    
    if not len(matches):
        return [], [] # cannot disturb this one
    
    # get their description vectors
    match_vecs = descs[vec_colnames].loc[matches]
           
    # find nearest neighbours
    neigh = NearestNeighbors(metric='euclidean')
    Y = match_vecs.values
    neigh.fit(Y) 
    
    X = descs[vec_colnames].values[descs['code'] == c]

    if Y.shape[0] > topn:
        dists, idxs = neigh.kneighbors(X, topn, return_distance=True)
    else:
        dists, idxs = neigh.kneighbors(X, Y.shape[0], return_distance=True)

    # turn distances to some heuristic probabilities
    #print(dists.flatten())
    probs = np.exp(-0.5 * dists.flatten())
    probs = probs / np.sum(probs)
    
    # turn idxs back to chars
    #print(idxs.flatten())
    charcodes = descs['code'][matches[idxs.flatten()]]
    
    #print(charcodes.values.flatten())
    
    chars = []
    for charcode in charcodes:
        chars.append(chr(int(charcode, 16)))

    # filter chars to ensure OOV scenario (if perturbations file from prev. perturbation contains any data...)
    c_orig = chr(int(c, 16))
    chars = [char for char in chars if not perturbations_file.observed(c_orig, char)]

    #print(chars)

    return chars, probs

parser = argparse.ArgumentParser()
parser.add_argument("-p",action="store",dest="prob")
parser.add_argument("-d",action="store",dest="docs")
parser.add_argument('--conll', dest='conll', action='store_true')
parser.add_argument('--perturbations-file', action="store", dest='perturbations_file')
parser.set_defaults(conll=False, perturbations_file='./perturbations.txt')

parsed_args = parser.parse_args(sys.argv[1:])

prob = float(parsed_args.prob)
docs = parsed_args.docs
isConll = parsed_args.conll==True
perturbations_file = PerturbationsStorage(parsed_args.perturbations_file)

if isConll:
  docs=read_data_conll(docs)
  output_format="conll"
else:
  docs=read_data_standard(docs)
  output_format="standard"

# the main loop for disturbing the text
topn=20

mydict={}

# docs = ['a b c d e f g h i j k l m n o p q r s t u v w x y z A B C D E F G H I J K L M N O P Q R S T U V W X Y Z abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ']
print_all_alternatives = False

for line in docs:

  if isConll:
    a,b = line.rstrip("\n").split("\t")
    b = b.split()
    a = a.split()
  else:
    a = line.rstrip("\n")

  out_x = []
  for c in a:
    #print(c)
    if c not in mydict:
      similar_chars, probs = get_unicode_desc_nn(c, perturbations_file, topn=topn)
      probs = probs[:len(similar_chars)]

      # normalise after selecting a subset of similar characters
      probs = probs / np.sum(probs)

      mydict[c] = (similar_chars, probs)

    else:
      similar_chars, probs = mydict[c]

    r = random.random()
    if r<prob and len(similar_chars):
      s = np.random.choice(similar_chars, 1, replace=True, p=probs)[0]
    else:
      s = c
    out_x.append(s)

    if print_all_alternatives:

      print("{}\t{}".format(c, similar_chars))

  if isConll:
    print('idx\toriginal\tdisturbed\thex')
        
    for i in range(len(out_x)):
      print("{}\t{}\t{}\t{}".format(i+1, a[i], out_x[i], char_to_hex_string(out_x[i])))
    print() 
  else:
    print("{}".format("".join(out_x)))
