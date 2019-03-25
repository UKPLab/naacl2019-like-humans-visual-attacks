import pandas as pd
import numpy as np
import re
import pickle as pk
np.random.seed(seed=7)

data_path = './data-toxic-kaggle/train.csv'
pkl_path = './data-toxic-kaggle/toxic_comments_100.pkl'

perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl' # perturbed by Edwin's script
perturbed_path  = './data-toxic-kaggle/toxic_comments_100_mm_even_p1.pkl' # perturbed by Steffen_even script
def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)  
    string = re.sub(r"[0-9]+", " ", string)   
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " ", string) 
    string = re.sub(r"!", " ", string) 
    string = re.sub(r"\(", " ", string) 
    string = re.sub(r"\)", " ", string) 
    string = re.sub(r"\?", " ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def take_sample():
	df = pd.read_csv(data_path) 

	toxic = df.loc[df['toxic'] == 1]

	toxic_comments = toxic['comment_text'].tolist()

	print("The number of toxic comments: %d"%len(toxic_comments))

	toxic_comments_filterd = []
	for comment in toxic_comments:
		if len(comment) < 50:
			toxic_comments_filterd.append(comment)


	print("The number of comments whose lengths are less than 50: %d"%len(toxic_comments_filterd))

	np.random.shuffle(toxic_comments_filterd)

	# take 100 samples from comments with length less than 50
	toxic_comments_filterd = toxic_comments_filterd[:100]

	# clean comments
	toxic_comments_cleand = []
	for comment in toxic_comments_filterd: 
		comment_clean = clean_str(comment, True)
		toxic_comments_cleand.append(comment_clean)


	print(toxic_comments_cleand)

	with open(pkl_path, 'wb') as f:
		 pk.dump(toxic_comments_cleand, f)

def load_samples(perturbed_path, original_path ,verbose=False):

	with open(pkl_path, 'rb') as f:
		toxic_comments_clean = pk.load(f)

	# I  asked Edwin to perturb the data

	with open(perturbed_path, 'rb') as f:
		toxic_comments_perturbed = pk.load(f)

	if verbose == True:
		for i in range(100):
			print("%s --> %s"%(toxic_comments_clean[i],toxic_comments_perturbed[i]))

	return toxic_comments_clean, toxic_comments_perturbed

def convert_conll_to_pkl(txt_path,pkl_path):
	with open(txt_path, 'r') as f:
		lines = f.readlines()
	sentences = []
	sent = []
	for line in lines:
		if line != '\n':
			sent.append(line.strip())
		else:
			sentences.append(sent)
			sent = []
	out_lines = []
	for sent in sentences:
		out_lines.append(' '.join(sent))
	with open(pkl_path,'wb') as f:
		pk.dump(out_lines,f) 

if __name__=="__main__":
	#load_samples(0.1,True)

	### Use the below command if you would like to convert the output of the steffen's script to pkl
	for p in [0.1, 0.2, 0.4, 0.6, 0.8]:
		convert_conll_to_pkl(txt_path= 'data-toxic-kaggle/toxic_comments_100_mm_even_p%.1f.txt'%p,
							 pkl_path= 'data-toxic-kaggle/toxic_comments_100_mm_even_p%.1f.pkl'%p)

