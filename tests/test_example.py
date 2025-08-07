"""Tests for the example module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from leglove.example import find_nearest_neighbors, main


class TestFindNearestNeighbors:
    """Tests for the find_nearest_neighbors function."""

    @patch("leglove.example.Glove")
    @patch("builtins.print")
    @patch("leglove.example.pprint")
    def test_find_nearest_neighbors_basic(
        self, mock_pprint: Mock, mock_print: Mock, mock_glove_class: Mock
    ) -> None:
        """Test basic nearest neighbors functionality."""
        mock_model = Mock()
        mock_model.dictionary = {"legal": 0, "court": 1, "law": 2}
        mock_model.word_vectors = np.array(
            [
                [1.0, 0.0],  # legal
                [0.8, 0.6],  # court
                [0.6, 0.8],  # law
            ]
        )
        mock_glove_class.load.return_value = mock_model

        find_nearest_neighbors("test_model.model", "legal")

        mock_glove_class.load.assert_called_once_with("test_model.model")

        mock_print.assert_called_once_with("The 10 nearest neighbors of legal are...")

        mock_pprint.pprint.assert_called_once()

        pprint_arg = mock_pprint.pprint.call_args[0][0]
        assert isinstance(pprint_arg, list)
        assert len(pprint_arg) <= 10  # Should be at most K neighbors

    @patch("leglove.example.Glove")
    def test_find_nearest_neighbors_distances(self, mock_glove_class: Mock) -> None:
        """Test that distances are calculated correctly."""
        mock_model = Mock()
        mock_model.dictionary = {"word1": 0, "word2": 1, "word3": 2}
        mock_model.word_vectors = np.array(
            [
                [1.0, 0.0],  # word1
                [0.0, 1.0],  # word2
                [2.0, 0.0],  # word3
            ]
        )
        mock_glove_class.load.return_value = mock_model

        with patch("leglove.example.pprint") as mock_pprint:
            find_nearest_neighbors("test_model.model", "word1")

            results = mock_pprint.pprint.call_args[0][0]

            assert len(results) == 3

            result_dict = dict(results)

            assert "word1" in result_dict
            assert "word2" in result_dict
            assert "word3" in result_dict

            assert all(distance >= 0 for distance in result_dict.values())

            distances = [result[1] for result in results]
            assert distances == sorted(distances)


class TestMain:
    """Tests for the main function."""

    @patch("leglove.example.parse_arguments")
    @patch("leglove.example.train_and_save_model")
    @patch("leglove.example.find_nearest_neighbors")
    def test_main_train_mode(
        self, mock_find_neighbors: Mock, mock_train: Mock, mock_parse_args: Mock
    ) -> None:
        """Test main function in training mode."""
        mock_args = Mock()
        mock_args.train_dir = "/path/to/data"
        mock_args.load_model = None
        mock_args.model_name = "TestModel"
        mock_args.num_epochs = "15"
        mock_args.parallel_threads = "4"
        mock_args.query = "legal"
        mock_parse_args.return_value = mock_args

        main()

        mock_train.assert_called_once_with(
            "/path/to/data", model_name="TestModel", num_epochs=15, parallel_threads=4
        )

        mock_find_neighbors.assert_called_once_with("TestModel.model", "legal")

    @patch("leglove.example.parse_arguments")
    @patch("leglove.example.train_and_save_model")
    @patch("leglove.example.find_nearest_neighbors")
    def test_main_load_mode(
        self, mock_find_neighbors: Mock, mock_train: Mock, mock_parse_args: Mock
    ) -> None:
        """Test main function in load mode."""
        mock_args = Mock()
        mock_args.train_dir = None
        mock_args.load_model = "ExistingModel.model"
        mock_args.query = "court"
        mock_parse_args.return_value = mock_args

        main()

        mock_train.assert_not_called()

        mock_find_neighbors.assert_called_once_with("ExistingModel.model", "court")

    @patch("leglove.example.parse_arguments")
    def test_main_no_input_error(self, mock_parse_args: Mock) -> None:
        """Test main function raises error when no input provided."""
        mock_args = Mock()
        mock_args.train_dir = None
        mock_args.load_model = None
        mock_parse_args.return_value = mock_args

        with pytest.raises(
            ValueError,
            match="Must provide either a training directory or a model file to load",
        ):
            main()

    @patch("leglove.example.parse_arguments")
    @patch("leglove.example.train_and_save_model")
    @patch("leglove.example.find_nearest_neighbors")
    def test_main_default_model_name(
        self, mock_find_neighbors: Mock, mock_train: Mock, mock_parse_args: Mock
    ) -> None:
        """Test main function with default model name."""
        mock_args = Mock()
        mock_args.train_dir = "/path/to/data"
        mock_args.load_model = None
        mock_args.model_name = "LeGlove"  # Default value
        mock_args.num_epochs = "10"  # Default value
        mock_args.parallel_threads = "1"  # Default value
        mock_args.query = "legal"
        mock_parse_args.return_value = mock_args

        main()

        mock_train.assert_called_once_with(
            "/path/to/data", model_name="LeGlove", num_epochs=10, parallel_threads=1
        )

        mock_find_neighbors.assert_called_once_with("LeGlove.model", "legal")
