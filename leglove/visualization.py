"""Visualization utilities for word vectors."""

from typing import Iterable, Mapping

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def plot_vectors(
    word_to_vector: Mapping[str, np.ndarray], words: Iterable[str]
) -> None:
    """Plot a 2D scatter of selected word vectors using PCA.

    This function reduces the dimensionality of the provided word vectors to two
    principal components and creates a scatter plot labeling each word. The plot
    is displayed using :mod:`matplotlib` if it is installed.

    Args:
        word_to_vector: Mapping from words to their vector representations.
        words: Iterable of words to plot. Each word must exist in
            ``word_to_vector``.

    Raises:
        ImportError: If :mod:`matplotlib` is not installed.
        ValueError: If any requested word is missing from ``word_to_vector``.
    """
    if plt is None:  # pragma: no cover - handled via tests with patching
        raise ImportError("matplotlib is required for plotting but is not installed")

    vectors = []
    selected_words = []
    for word in words:
        if word not in word_to_vector:
            raise ValueError(f"Word '{word}' not found in word_to_vector")
        vectors.append(word_to_vector[word])
        selected_words.append(word)

    matrix = np.array(vectors, dtype=float)
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vt[:2].T

    x_coords = reduced[:, 0]
    y_coords = reduced[:, 1]

    plt.figure()
    plt.scatter(x_coords, y_coords)

    for x, y, word in zip(x_coords, y_coords, selected_words):
        plt.annotate(word, (x, y))

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Word Vector Visualization")
    plt.tight_layout()
    plt.show()
