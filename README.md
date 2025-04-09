# Morpho Tokenizer Library

Morpho is a versatile and efficient tokenizer library designed for various natural language processing tasks. It supports multiple tokenization algorithms, including BPE, WordPiece, and SentencePiece, with customizable options for pre-tokenization patterns.

## Features

- **Multiple Tokenization Algorithms**: Supports BPE, WordPiece, Unigram, and SentencePiece.
- **Customizable Pre-tokenization**: Use regex patterns for pre-tokenization, including GPT-2 style patterns.
- **Visualization Tools**: Visualize BPE merge operations and token segmentation.
- **Command-line Interface**: Easy-to-use CLI for training, encoding, and decoding.
- **Extensible Design**: Easily add new tokenization algorithms and features.

## Installation

To install the Morpho Tokenizer Library, clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/morpho.git
cd morpho
pip install -r requirements.txt
```

## Usage

### Command-line Interface

Morpho provides a CLI for training and using tokenizers. Here are some examples:

#### Train a BPE Tokenizer

```bash
morpho train --type bpe --file data.txt --vocab-size 8000 --use-gpt2-pattern
```

#### Encode Text

```bash
morpho encode --tokenizer model.json --text "hello world"
```

#### Decode Token IDs

```bash
morpho decode --tokenizer model.json --ids 12 15 18
```

### Python API

You can also use Morpho programmatically in your Python scripts:

```python
from tokenizers.bpe import BPETokenizer

# Create a tokenizer with a GPT-2 style regex pattern
gpt2_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
tokenizer = BPETokenizer(vocab_size=8000, pre_tokenize_pattern=gpt2_pattern)

# Train the tokenizer
tokenizer.train(["data.txt"], min_frequency=2)

# Tokenize text
tokens = tokenizer.tokenize("Hello, world!")
```

## Visualization

Morpho includes utilities for visualizing BPE merge operations and token segmentation. Ensure you have `matplotlib` and `networkx` installed:

```bash
pip install matplotlib networkx
```


## License

Morpho is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for more details.


## Contact

For questions or feedback, please open an issue on GitHub or contact us at [deepaksilaych@gmail.com].