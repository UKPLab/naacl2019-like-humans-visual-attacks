# Simple MLP for training models on toxic comment classification

For training new models and testing do:

		python3 train.py train_data_embeddings dev_data_embeddings test_data_embeddings path_to_store_model random_seed train_label_path dev_label_path test_label_path

For only testing on existing models do:

		python3 test.py test_data_embeddings path_to_trained_model path_to_store_results path_to_labels

When testing, make sure to test on the same models as during training.


Requires following packages:

* gensim==3.6.0
* h5py==2.8.0
* Keras==2.2.3
* nltk==3.3
* scikit-learn==0.20.0
* numpy==1.15.2
* tensorflow==1.8.0

