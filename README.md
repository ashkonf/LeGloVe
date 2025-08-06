# LeGloVe

A Python implementation of GloVe word vectors for legal domain-specific corpuses.

## Table of Contents

- [Overview](#overview)
- [Pre-trained Word Vectors](#pre-trained-word-vectors)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Loading and Using a Trained Model](#loading-and-using-a-trained-model)
  - [Example Usage: Nearest Neighbors](#example-usage-nearest-neighbors)
- [Development](#development)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Examples](#examples)
- [References](#references)
- [License](#license)

## Overview

This implementation builds off of [this open-source repository](https://github.com/maciejkula/glove-python). 
The original GloVe project can be found [here](https://github.com/stanfordnlp/GloVe).

## Pre-trained Word Vectors

You can download a pre-trained model containing 100-dimensional word vectors here: [LeGlove.model.zip](https://drive.google.com/uc?export=download&id=1JMPie8EZAzaG7ucamrmvO9vg7y3Z2QtT). The model was trained on 63,981 Supreme Court opinions (scotus) from 1789 to 2014. Judicial opinion data is made available through CourtListener, courtesy of the Free Law Project.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. First, install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies:

```bash
uv sync
```

## Usage

### Training

`train.py` exports one public function:

```python
train_and_save_model(data_dir, model_name='LeGlove', num_epochs=10, parallel_threads=1)
```

This function trains and saves a model using the legal corpus in the data directory provided. It does so by first pre-processing the corpus using a series of legal-domain specific regexes. Afterwards, it fits the co-occurrence matrix of the corpus to a GloVe model that is saved to the current directory.

Arguments:

1. `data_dir`: The master directory containing all jurisdiction-level subdirectories. Each of these subdirectories is a list of json files containing the legal opinions. All of the json files in each subdirectory will be read and considered part of the training corpus.
2. `model_name`: Name of the model to be saved to disk.
3. `num_epochs`: Number of epochs for which to train the model.
4. `parallel_threads`: Number of parallel threads to use for training.

Output:

[**model_name**].model is saved to disk in the current directory. This model can then be loaded to obtain all trained word vectors.

### Loading and Using a Trained Model

`example.py` contains code, duplicated below for convenience, that illustrates how to load a pre-trained model (by the name of LeGlove.model).

```python
model = Glove.load('LeGlove.model')
dictionary = model.dictionary
word_vectors = model.word_vectors
```

`dictionary` is a map from the string of a word to its word index, for all words. `word_vectors` is a map from a word index to its corresponding word vector, for all word indexes (for all words).

The trained vector of a word can thus be accessed by first obtaining its word index from `dictionary` and then using this index to obtain the word vector from `word_vectors`. As an example, the following code shows how to obtain the word vector for the word "legal" using this method (stored in the variable `legal_word_vector`).

```python
legal_word_idx = dictionary['legal']
legal_word_vector = word_vectors[legal_word_idx]
```

### Example Usage: Nearest Neighbors

`example.py` contains a sample program that outputs the nearest neighbors of a query word based on the word vectors of a given model. It illustrates how to train a model or load it from disk, as well as how to retrieve word vectors from the model.

To train a model, you can run the following command:

```bash
uv run python example.py --train_dir data/ --model_name LeGlove --query legal
```

To load a model, you can run the following command:

```bash
uv run python example.py --load_model LeGlove.model --query legal
```

## Development

### Setting Up Development Environment

1. Install uv (see [Installation](#installation))
2. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/ashkonf/LeGloVe.git
   cd LeGloVe
   uv sync
   ```
3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

### Running Tests

Run the full test suite with coverage:

```bash
uv run pytest --cov=. --cov-report=term-missing
```

Run specific test files:

```bash
uv run pytest tests/test_train.py
```

### Code Quality

This project uses several tools to maintain code quality:

#### Ruff (Linting and Formatting)

Check for linting issues:
```bash
uv run ruff check .
```

Auto-fix linting issues:
```bash
uv run ruff check . --fix
```

Format code:
```bash
uv run ruff format .
```

#### Pyright (Type Checking)

Run type checking:
```bash
uv run pyright .
```

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit. They include:

- `ruff check` - Linting
- `ruff format` - Code formatting  
- `pyright` - Type checking
- `pytest` - Running tests

To run all pre-commit hooks manually:

```bash
uv run pre-commit run --all-files
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating usage:

- `sample_usage.ipynb` - Basic usage examples and nearest neighbor queries

To run the notebooks:

```bash
uv run jupyter notebook examples/
```

## References

[Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation.](https://www.aclweb.org/anthology/D14-1162/)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
