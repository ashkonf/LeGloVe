import os
import re
from typing import Generator, List

try:
    from glove import Corpus, Glove
except ImportError:
    Glove = None
    Corpus = None

from nltk.tokenize import word_tokenize

from .cleanup import extract_text
from .regexes import REGEX_TOKENS, REGEXES

"""
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
"""

# Constants
CONTEXT_WINDOW = 10  # length of the (symmetric)context window used for cooccurrence
LEARNING_RATE = 0.05  # learning rate used for model training
NUM_COMPONENTS = 100  # number of components/dimension of output word vectors


## LeGlove #####################################################################################


def tokenize_text(plain_text: str) -> List[str]:
    """
    Tokenize and preprocess legal document text.

    This function accepts a string representation of a legal document as input.
    It returns a tokenized form of the input after preprocessing using legal
    domain-specific regex patterns.

    Args:
        plain_text: Raw text content of a legal document.

    Returns:
        List of preprocessed tokens from the input text.

    Example:
        >>> text = "The court held in Smith v. Jones, 123 F.3d 456 (2020)..."
        >>> tokens = tokenize_text(text)
        >>> print(tokens[:3])
        ['the', 'court', 'held']
    """

    # Clean plain text by replacing all regex matches
    # with corresponding tokens
    cleaned_text = plain_text
    for idx, regex in enumerate(REGEXES):
        cleaned_text = re.sub(
            regex, REGEX_TOKENS[idx], cleaned_text, flags=re.IGNORECASE
        )

    # Use NLTK tokenizer to return tokenized form of cleaned text
    tokens = word_tokenize(cleaned_text.lower())
    return tokens


def read_corpus(data_dir: str) -> Generator[List[str], None, None]:
    """
    Generate preprocessed tokens from all JSON files in the data directory.

    This function returns a generator of lists of preprocessed tokens over all
    JSON files in the given data directory. It processes files from all
    jurisdiction-level subdirectories.

    Args:
        data_dir: Master directory containing jurisdiction-level subdirectories
                 with JSON files containing legal opinions.

    Yields:
        List of preprocessed tokens for each valid legal document.

    Example:
        >>> for tokens in read_corpus("/path/to/data"):
        ...     print(f"Document has {len(tokens)} tokens")
    """

    num_files_read = 0
    for juris_dir in os.listdir(data_dir):
        # Avoid hidden files in directory
        if juris_dir.startswith("."):
            continue
        juris_dir_path = os.path.join(data_dir, juris_dir)
        if not os.path.isdir(juris_dir_path):
            continue
        print(f"Reading {juris_dir}...")

        for json_file in os.listdir(juris_dir_path):
            if not json_file.endswith(".json"):
                continue
            num_files_read += 1
            if num_files_read % 1e3 == 0:
                print(f"{int(num_files_read)} json files read...")

            json_file_path = os.path.join(juris_dir_path, json_file)
            plain_text = extract_text(json_file_path)
            if plain_text != "":
                tokens = tokenize_text(plain_text)
                yield tokens


def train_and_save_model(
    data_dir: str,
    model_name: str = "LeGlove",
    num_epochs: int = 10,
    parallel_threads: int = 1,
) -> None:
    """
    Process legal corpus data and train a GloVe model.

    This function processes all the data into a training corpus and fits a GloVe
    model to this corpus. The trained model is saved to the current directory.

    Args:
        data_dir: Master directory containing all jurisdiction-level directories
                 with JSON files containing legal opinions.
        model_name: Name of model to be used for output file.
        num_epochs: Number of epochs for which to train model.
        parallel_threads: Number of parallel threads to use for training.

    Returns:
        None. The trained model is saved as "[model_name].model" in the current directory.

    Example:
        >>> train_and_save_model("/path/to/data", "MyModel", num_epochs=15)
        # Creates "MyModel.model" file in current directory
    """

    if Corpus is None or Glove is None:
        raise ImportError(
            "glove-python is required but not installed. Install with: uv sync --extra glove"
        )

    corpus_model = Corpus()
    corpus_model.fit(read_corpus(data_dir), window=CONTEXT_WINDOW)

    glove = Glove(no_components=NUM_COMPONENTS, learning_rate=LEARNING_RATE)
    glove.fit(
        corpus_model.matrix,
        epochs=num_epochs,
        no_threads=parallel_threads,
        verbose=True,
    )
    glove.add_dictionary(corpus_model.dictionary)

    glove.save(model_name + ".model")
