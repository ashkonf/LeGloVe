import argparse
import pprint

import numpy as np

try:
    from glove import Glove
except ImportError:
    Glove = None

from train import train_and_save_model

"""
    example.py
    --------
    This program illustrates example usage of the LeGlove modules
    train.py and vectors.py. It can either train a model from a
    given corpus or load a pre-trained model. Afterwards, the top
    K nearest neighbors of the query word by Euclidean vector
    distance are printed.

    The following open-source github repository was used and adapted:

        https://github.com/maciejkula/glove-python
"""


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Example usage of LeGlove module")
    parser.add_argument(
        "--train_dir",
        default=None,
        help="Master directory containing all jurisdiction-level directories, only for training",
    )
    parser.add_argument(
        "--model_name", default="LeGlove", help="Name for output model file"
    )
    parser.add_argument(
        "--num_epochs", default=10, help="Train LeGlove model for this number of epochs"
    )
    parser.add_argument(
        "--parallel_threads",
        default=1,
        help="Number of parallel threads to use for training",
    )
    parser.add_argument(
        "--load_model",
        default=None,
        help="Model to load for querying (i.e. 'LeGlove.model')",
    )
    parser.add_argument(
        "--query", default="legal", help="Get nearest neighbors of this word"
    )
    return parser.parse_args()


# Constants
K = 10  # number of neighbors to output


def find_nearest_neighbors(model_file: str, word: str) -> None:
    """Find and print the K nearest neighbors of a word using Euclidean distance."""
    print("The %d nearest neighbors of %s are..." % (K, word))

    # Load model and get dictionary (from word to word index) and word vectors
    if Glove is None:
        raise ImportError(
            "glove-python is required but not installed. Install with: uv sync --extra glove"
        )
    model = Glove.load(model_file)
    dictionary = model.dictionary
    word_vectors = model.word_vectors

    word_to_vector = {}
    for word in dictionary:
        word_idx = dictionary[word]
        word_to_vector[word] = word_vectors[word_idx]

    # Find closest neighbors by Euclidean distance
    nbr_distances = []
    query_vector = word_to_vector[word]

    for nbr in word_to_vector:
        nbr_vector = word_to_vector[nbr]
        dist = np.linalg.norm(query_vector - nbr_vector)
        nbr_distances.append((nbr, dist))

    # Print top K neighbors
    nearest_neighbors = sorted(nbr_distances, key=lambda x: x[1])
    pprint.pprint(nearest_neighbors[:K])


def main() -> None:
    """Train or load a model and display nearest neighbors for a query word."""
    args = parse_arguments()

    if not args.train_dir and not args.load_model:
        raise ValueError(
            "Must provide either a training directory or a model file to load"
        )

    # Option 1: Train a model
    if args.train_dir:
        train_and_save_model(
            args.train_dir,
            model_name=args.model_name,
            num_epochs=int(args.num_epochs),
            parallel_threads=int(args.parallel_threads),
        )
        model_file = args.model_name + ".model"

    # Option 2: Load a model
    else:
        model_file = args.load_model

    find_nearest_neighbors(model_file, args.query)


if __name__ == "__main__":
    main()
