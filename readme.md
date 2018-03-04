# LeGlove
A Python implementation of GloVe for legal domain-specific corpuses.

This implementation builds off of the following open-source repository: https://github.com/maciejkula/glove-python. 
The original GloVe project can be found here: https://github.com/stanfordnlp/GloVe.

# Usage

`train.py` exports one public function:
	
		train_and_save_model(data_dir, model_name='LeGlove', num_epochs=10, parallel_threads=1)

This function trains and saves a model using the legal corpus in the data directory provided. It does so by first pre-processing the corpus using a series of legal-domain specific regexes. Afterwards, it fits the co-occurence matrix of the corpus to a GloVe model that is saved to the current directory. 

Arguments:

1.  **data_dir**: The master directory containing all jurisdiction-level subdirectories. Each of these subdirectories is a list of json files containing the legal opinions. All of the json files in each subdirectory will be read and considered part of the training corpus.

2. **model_name**: Name of the model to be saved to disk. 

3. **num_epochs**: Number of epochs for which to train the model.

4. **parallel_threads**: Number of parallel threads to use for training.

Output:

[**model_name**].model is saved to disk in the current directory. This model can then be loaded to obtain all trained word vectors. 

# Dependencies

All dependencies are listed in `requirements.txt`. The necessary libraries can be installed all at once using the following command.

		pip install -r requirements.txt

While installing `glove_python`, you might run into an error with your gcc not being a sufficiently recent version. To fix this issue, you can run the command ```brew upgrade gcc``` and then try installing all the requirements again. If there are still issues with installation, check https://github.com/maciejkula/glove-python/issues/55 to see if that may be your problem.

# Example Usage: Nearest neighbors

`example.py` contains a sample program that outputs the nearest neighbors of a query word based on the word vectors of a given model. It illustrates how to train a model or load it from disk, as well as how to retrieve word vectors from the model.

To train a model, you can run the following command:

		python example.py --train_dir data/ --model_name LeGlove --query legal

To load a model, you can run the following command:

		python example.py --load_model LeGlove.model --query legal


# References

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. 

