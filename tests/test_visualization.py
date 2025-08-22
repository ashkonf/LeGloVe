"""Tests for the visualization module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from leglove.visualization import plot_vectors


class TestPlotVectors:
    """Tests for the plot_vectors function."""

    def test_plot_vectors_calls_matplotlib(self) -> None:
        """Ensure plotting calls the expected matplotlib functions."""
        word_to_vector = {
            "legal": np.array([1.0, 0.0, 0.0]),
            "court": np.array([0.0, 1.0, 0.0]),
            "law": np.array([0.0, 0.0, 1.0]),
        }
        words = ["legal", "court", "law"]

        mock_plt = Mock()
        with patch("leglove.visualization.plt", mock_plt):
            plot_vectors(word_to_vector, words)

        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called_once()
        assert mock_plt.annotate.call_count == len(words)
        mock_plt.show.assert_called_once()

    def test_plot_vectors_missing_word(self) -> None:
        """Ensure a missing word raises a ValueError."""
        word_to_vector = {"legal": np.array([1.0, 0.0, 0.0])}
        words = ["legal", "court"]

        with patch("leglove.visualization.plt", Mock()):
            with pytest.raises(ValueError, match="Word 'court' not found"):
                plot_vectors(word_to_vector, words)
