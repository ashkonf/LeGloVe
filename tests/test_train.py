"""Tests for the train module."""

import os
from unittest.mock import Mock, patch

from leglove.train import read_corpus, tokenize_text, train_and_save_model


class TestTokenizeText:
    """Tests for the tokenize_text function."""

    def test_tokenize_basic_text(self) -> None:
        """Test tokenizing basic text."""
        text = "The court held that the defendant was guilty."
        result = tokenize_text(text)

        assert isinstance(result, list)
        assert len(result) > 0
        assert "the" in result
        assert "court" in result
        assert "held" in result

    def test_tokenize_with_citations(self) -> None:
        """Test tokenizing text with legal citations."""
        text = "See Smith v. Jones, 123 F.3d 456 (2020)."
        result = tokenize_text(text)

        assert isinstance(result, list)
        assert "judicial_opinion_citation" in result

    def test_tokenize_with_law_citation(self) -> None:
        """Test tokenizing text with law citations."""
        text = "Under § 123.45 of the code."
        result = tokenize_text(text)

        assert isinstance(result, list)
        assert "law_citation" in result

    def test_tokenize_empty_text(self) -> None:
        """Test tokenizing empty text."""
        result = tokenize_text("")
        assert result == []

    def test_tokenize_case_insensitive(self) -> None:
        """Test that tokenization is case insensitive."""
        text = "THE COURT HELD"
        result = tokenize_text(text)

        assert all(token.islower() for token in result if token.isalpha())


class TestReadCorpus:
    """Tests for the read_corpus function."""

    def test_read_corpus_basic(self, sample_corpus_dir: str) -> None:
        """Test reading a basic corpus."""
        result = list(read_corpus(sample_corpus_dir))

        assert len(result) == 2  # Two JSON files created in fixture
        assert all(isinstance(tokens, list) for tokens in result)
        assert all(len(tokens) > 0 for tokens in result)

    def test_read_corpus_empty_directory(self, temp_dir: str) -> None:
        """Test reading from empty directory."""
        result = list(read_corpus(temp_dir))
        assert result == []

    def test_read_corpus_no_json_files(self, temp_dir: str) -> None:
        """Test reading directory with no JSON files."""
        subdir = os.path.join(temp_dir, "test_juris")
        os.makedirs(subdir)

        with open(os.path.join(subdir, "not_json.txt"), "w") as f:
            f.write("This is not a JSON file")

        result = list(read_corpus(temp_dir))
        assert result == []

    def test_read_corpus_hidden_directories(self, temp_dir: str) -> None:
        """Test that hidden directories are skipped."""
        hidden_dir = os.path.join(temp_dir, ".hidden")
        os.makedirs(hidden_dir)

        result = list(read_corpus(temp_dir))
        assert result == []

    @patch("leglove.train.extract_text")
    def test_read_corpus_empty_text(
        self, mock_extract_text: Mock, temp_dir: str
    ) -> None:
        """Test handling of files that return empty text."""
        mock_extract_text.return_value = ""

        juris_dir = os.path.join(temp_dir, "test_juris")
        os.makedirs(juris_dir)

        with open(os.path.join(juris_dir, "test.json"), "w") as f:
            f.write('{"html": "<p>Test</p>"}')

        result = list(read_corpus(temp_dir))
        assert result == []


class TestTrainAndSaveModel:
    """Tests for the train_and_save_model function."""

    @patch("leglove.train.Glove")
    @patch("leglove.train.Corpus")
    @patch("leglove.train.read_corpus")
    def test_train_and_save_model_basic(
        self,
        mock_read_corpus: Mock,
        mock_corpus_class: Mock,
        mock_glove_class: Mock,
        temp_dir: str,
    ) -> None:
        """Test basic model training and saving."""
        mock_corpus = Mock()
        mock_corpus.matrix = "mock_matrix"
        mock_corpus.dictionary = {"word": 0}
        mock_corpus_class.return_value = mock_corpus

        mock_glove = Mock()
        mock_glove_class.return_value = mock_glove

        mock_read_corpus.return_value = [["word1", "word2"], ["word3", "word4"]]

        train_and_save_model(
            temp_dir, model_name="TestModel", num_epochs=5, parallel_threads=2
        )

        mock_corpus_class.assert_called_once()
        mock_corpus.fit.assert_called_once()

        mock_glove_class.assert_called_once_with(no_components=100, learning_rate=0.05)
        mock_glove.fit.assert_called_once_with(
            "mock_matrix", epochs=5, no_threads=2, verbose=True
        )
        mock_glove.add_dictionary.assert_called_once_with({"word": 0})
        mock_glove.save.assert_called_once_with("TestModel.model")

    @patch("leglove.train.Glove")
    @patch("leglove.train.Corpus")
    @patch("leglove.train.read_corpus")
    def test_train_and_save_model_default_params(
        self,
        mock_read_corpus: Mock,
        mock_corpus_class: Mock,
        mock_glove_class: Mock,
        temp_dir: str,
    ) -> None:
        """Test model training with default parameters."""
        mock_corpus = Mock()
        mock_corpus.matrix = "mock_matrix"
        mock_corpus.dictionary = {"word": 0}
        mock_corpus_class.return_value = mock_corpus

        mock_glove = Mock()
        mock_glove_class.return_value = mock_glove

        mock_read_corpus.return_value = [["word1", "word2"]]

        train_and_save_model(temp_dir)

        mock_glove.fit.assert_called_once_with(
            "mock_matrix", epochs=10, no_threads=1, verbose=True
        )
        mock_glove.save.assert_called_once_with("LeGlove.model")
