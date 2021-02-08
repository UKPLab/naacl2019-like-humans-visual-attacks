from allennlp.modules.elmo import Elmo, batch_to_ids

def load_emb(json_file, hdf5_file):
	vlmo = Elmo(json_file, hdf5_file, 1, dropout=0) 
	return vlmo


def word_to_sentence(embeddings):
	return embeddings.mean(axis=1)

def embed(sentences,loaded_emb):
	# use batch_to_ids to convert sentences to character ids
	sents = []

	for sent in sentences:
		
		sents.append(sent.split())

	character_ids = batch_to_ids(sents)# 2,2,50

	embeddings = loaded_emb(character_ids)

	embeddings = embeddings['elmo_representations'][0]
	
	embeddings = embeddings.data.numpy()
	
	return embeddings

def get_embeddings(sentences,loaded_vlmo):
	
	emb = embed(sentences,loaded_vlmo)
	
	return word_to_sentence(emb)


if __name__ == "__main__":
	
	json_file = "./velmo_embeddings/options.json"

	hdf5_file = "./velmo_embeddings/new_weights_5_epoch.hdf5"

	sentences = ['First sentence', 'Another']
	
	loaded_vlmo = load_emb(json_file,hdf5_file)
	
	em = embed(sentences,loaded_vlmo)
	
	print(em.shape)

	print(em[:,:,:5])
