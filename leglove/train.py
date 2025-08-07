import logging
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
    """Tokenize legal text and replace regex matches with placeholder tokens."""

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
    """Yield tokenized documents from JSON files in the given data directory."""

    num_files_read = 0
    for juris_dir in os.listdir(data_dir):
        # Avoid hidden files in directory
        if juris_dir.startswith("."):
            continue
        juris_dir_path = os.path.join(data_dir, juris_dir)
        if not os.path.isdir(juris_dir_path):
            continue
        logging.info(f"Reading {juris_dir}...")

        for json_file in os.listdir(juris_dir_path):
            if not json_file.endswith(".json"):
                continue
            num_files_read += 1
            if num_files_read % 1e3 == 0:
                logging.info(f"{int(num_files_read)} json files read...")

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
    """Process a legal corpus and train and save a GloVe model."""

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
