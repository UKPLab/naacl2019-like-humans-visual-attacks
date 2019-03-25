'''
required packages:
    allennlp 
    pandas

Note: 
    check the pathes to data in data_helper.py
    for velmo,velmo30k1ep, and selmo30k1ep: Virtual env "allennlp_wuhn" should be activated, then run 
        ipython ./sanity_check.py velmo|velmo30k1ep|selmo30k1ep|selmo30k5ep|velmo16k5ep steffen_even|edwin|random

    for elmo: Virtual env "allennlp" should be activated, then run 
        ipython ./sanity_check.py elmo steffen_even|edwin|random
'''
import embeddings_velmo as embeddings
import numpy as np
import argparse
import data_helper
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s:%(message)s')

ignore_list = [' ']
np.random.seed(seed=7)
output_path = "./data-toxic-kaggle/"

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_with_probaility(load_emb, perturbation_script, p=1.0):
    pkl_path = './data-toxic-kaggle/toxic_comments_100.pkl' 

    if perturbation_script == 'steffen_even':
        perturbed_path = './data-toxic-kaggle/toxic_comments_100_mm_even_p%.1f.pkl'%p # perturbed by Steffen_even script   
        original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
        logging.info("perturbation_script: steffen_even")

        logging.info("data loaded")
        output = []
        for index in range(len(original)):
            logging.info("index:%d"%index) 
            original_sample = original[index]
            perturbed_sample = perturbed[index]
            (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, perturbed_sample],load_emb)
            output.append("%s,%s,%.2f"%(original_sample,perturbed_sample, cos_sim(vec_orig, vec_pert)))


    elif perturbation_script == 'edwin':
        perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl' # perturbed by Edwin's script
        original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
        logging.info("perturbation_script: edwin")

        logging.info("data loaded")
        output = []
        for index in range(len(original)):
            logging.info("index:%d"%index) 
            original_sample = original[index]
            perturbed_sample = perturbed[index]
            pert = ''
            for i,ch in enumerate(original_sample):
                prob = np.random.uniform()
                if ( prob<= p) and (ch not in ignore_list):
                    # disturb
                    pert +=  perturbed_sample[i]
                else:
                    pert += ch
            #print([original_sample,pert])
            (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, pert],load_emb)
            output.append("%s,%s,%.2f"%(original_sample,pert, cos_sim(vec_orig, vec_pert)))

    else:
        raise ValueError("%s is not implemented!"%perturbation_script)

    return output

def online_compare(load_emb, original, perturbed):
    print("data loaded")
    output = []
    (vec_orig, vec_p_1) = embeddings.get_embeddings([original, perturbed],load_emb)
    return cos_sim(vec_orig, vec_p_1)

def compare_with_random(load_emb):

    pkl_path = './data-toxic-kaggle/toxic_comments_100.pkl' 

    perturbed_path = './data-toxic-kaggle/toxic_comments_100_perturbed.pkl' # perturbed by Edwin's script
    
    original, perturbed = data_helper.load_samples(perturbed_path,pkl_path)
    
    logging.info("data loaded")

    perturbed = original[:]

    np.random.shuffle(perturbed)

    logging.info("sentences are shuffled")
    
    output = []
        
    for index in range(len(original)):
            
        logging.info("index:%d"%index) 
        
        original_sample = original[index]
        
        perturbed_sample = perturbed[index]
            
        #print([original_sample, perturbed_sample])
        
        (vec_orig, vec_pert) = embeddings.get_embeddings([original_sample, perturbed_sample],load_emb)
        
        output.append("%s,%s,%.2f"%(original_sample,perturbed_sample, cos_sim(vec_orig, vec_pert)))

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("emb", help="type of embeddings")
    parser.add_argument("pert", help="type of perturbation_script")

    args = parser.parse_args()
    emb_type = args.emb
    pert_type = args.pert

    if emb_type == "elmo":

        json_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        hdf5_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" 
    
    elif emb_type == "selmo30k1ep":

        json_file = "./SELMO.30k.1ep/options.json"
        hdf5_file = "./SELMO.30k.1ep/weights.hdf5" 

    elif emb_type == "velmo":

        json_file = "./velmo_embeddings/options.json"
        hdf5_file = "./velmo_embeddings/new_weights_5_epoch.hdf5"

    elif emb_type == "velmo30k1ep":

        json_file = "./VELMO.30k.1ep/velmo_options.json"
        hdf5_file = "./VELMO.30k.1ep/velmo_weights.hdf5" 

    elif emb_type == "selmo30k5ep":

        json_file = "./SELMO.30k.5ep/selmo.30k.5ep.options.json"
        hdf5_file = "./SELMO.30k.5ep/selmo.30k.5ep.weights.hdf5" 

    elif emb_type == "velmo16k5ep":
    
        json_file = "./VELMO.16k.5ep/options.json"
        hdf5_file = "./VELMO.16k.5ep/velmo_5_epoch.hdf5"         
    
    else:
        raise ValueError("%s is not implemented!"%emb_type)

    loaded_emb =  embeddings.load_emb(json_file, hdf5_file)

    logging.info("%s loaded"%emb_type)    

    output_path = output_path+pert_type+'/'

    logging.info('pert_type: %s'%pert_type)

    if pert_type == 'random':

        comparions = compare_with_random(loaded_emb) 

        print("%s%s_comparisons"%(output_path,emb_type))

        with open("%s%s_comparisons"%(output_path,emb_type),'wt') as file:
                    
                file.write("\n".join(comparions))


    else:

        for p in [0.1, 0.2, 0.4, 0.8, 1.0]:
                
            logging.info("p = %.1f"%p)
                
            comparions = compare_with_probaility(loaded_emb,pert_type, p)
    
            with open("%s%s_comparisons_p_%.1f"%(output_path,emb_type,p),'wt') as file:
                    
                file.write("\n".join(comparions))

