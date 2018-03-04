from __future__ import print_function

import re
import os
import wget

from glove import Glove
from glove import Corpus

from cleanup import extract_text
from regexes import REGEXES
from nltk.tokenize import word_tokenize

'''
	train.py
	--------
	This module trains a GloVe model on a given legal corpus to build
	legal domain-specific word vectors. The program first preprocesses
	the legal opinions using a series of regexes and then calls GloVe 
	functions to output word vectors. The final trained model is saved 
	into the current directory as "LeGlove.model".

	The following open-source github repository was used and adapted:

		https://github.com/maciejkula/glove-python

	The original GloVe project can be found here:

		https://github.com/stanfordnlp/GloVe
'''

# Constants
DUMMY_TOKEN = 'court_citation'	# dummy token used to replace legal-specific content in corpus 	
CONTEXT_WINDOW = 10				# length of the (symmetric)context window used for cooccurrence
LEARNING_RATE = 0.05			# learning rate used for model training
NUM_COMPONENTS = 100			# number of components/dimensions of output word vectors


## LeGlove #####################################################################################

def tokenize_text(plain_text):
	'''
	This function accepts a string representation
	of a legal document as input. It returns a 
	tokenized form of the input after pre-processing.
	'''

	# Clean plain text with regexes
	# TODO: Fix the regex sub below to replace in decreasing order of match lengths
	cleaned_text = plain_text
	for regex in REGEXES:
		cleaned_text = re.sub(regex, DUMMY_TOKEN, cleaned_text, flags=re.IGNORECASE)

	# Replace consecutive dummy tokens with a single dummy token
	cleaned_text = re.sub(DUMMY_TOKEN + '+', DUMMY_TOKEN, cleaned_text)

	# Use NLTK tokenizer to return tokenized form of cleaned text
	tokens = word_tokenize(cleaned_text.lower())
	return tokens

def read_corpus(data_dir):
	'''
	This function returns a generator of lists of 
	pre-processed tokens over all files in the given 
	data directory.
	'''

	for juris_dir in os.listdir(data_dir):
		# Avoid hidden files in directory
		if (juris_dir.startswith('.')): continue
		if (juris_dir != 'fiscr'): continue
		juris_dir_path = os.path.join(data_dir, juris_dir)

		for json_file in os.listdir(juris_dir_path):
			if (not json_file.endswith('.json')): continue
			json_file_path = os.path.join(juris_dir_path, json_file)
			plain_text = extract_text(json_file_path)
			if (plain_text != ''):
				tokens = tokenize_text(plain_text)
				yield tokens

def train_and_save_model(data_dir, model_name='LeGlove', num_epochs=10, parallel_threads=1):
	'''
	This function processes all the data into a training
	corpus and fits a GloVe model to this corpus. 

	Parameters:
		data_dir (string): 			master directory containing all jurisdiction-level directories
		model_name (string):		name of model to be used for output
		num_epochs (int):  			number of epochs for which to train model
		parallel_threads (int):		number of parallel threads to use for training

	The trained model is saved as "[model_name].model" into the current directory.
	'''

	corpus_model = Corpus()
	corpus_model.fit(read_corpus(data_dir), window=CONTEXT_WINDOW)

	glove = Glove(no_components=NUM_COMPONENTS, learning_rate=LEARNING_RATE)
	glove.fit(corpus_model.matrix, epochs=num_epochs,
				no_threads=parallel_threads, verbose=True)
	glove.add_dictionary(corpus_model.dictionary)

	glove.save(model_name + '.model')