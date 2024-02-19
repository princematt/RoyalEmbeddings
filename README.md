# RoyalEmbeddings
RoyalEmbeddings is a comprehensive repository designed to demonstrate the application of Word2Vec models for embedding generation and analysis using a corpus themed around royalty. 
These scripts demonstrate the process of preprocessing textual data, training word embedding models using both Gensim's Word2Vec and a custom TensorFlow implementation, and visualizing the generated embeddings.

## Overview

The scripts cover a range of operations, including:

- Text preprocessing to tokenize and clean a given corpus.
- Training word embeddings with Gensim's Word2Vec model.
- Implementing a simple neural network in TensorFlow to generate word embeddings.
- Visualizing word embeddings using Matplotlib and Seaborn.
- Saving and loading word embeddings in a format compatible with Gensim's `KeyedVectors`.
- A custom implementation of the Word2Vec algorithm.

## Files Description

The repository is structured as follows:

1. **Text Preprocessing and Word2Vec Training**: Initial steps involve preprocessing text data by tokenizing and removing stopwords, followed by training a Word2Vec model on the processed corpus.

2. **Embedding Visualization**: Scripts for visualizing the word embeddings in a 2-dimensional space to explore the relationships between words.

3. **Custom Word2Vec Implementation**: A Python class that implements a simplified version of the Word2Vec algorithm from scratch using NumPy. This includes methods for training the model, performing forward and backpropagation, and finding words similar to a given input.

4. **Embedding Saving and Loading**: Demonstrates how to save generated word embeddings to a text file and how to load them back using Gensim's `KeyedVectors`.

## Dependencies

To run the scripts, you will need the following libraries:

- NumPy
- Pandas
- NLTK
- Gensim
- TensorFlow
- Matplotlib
- Seaborn

You can install these dependencies via pip:

```bash
pip install numpy pandas nltk gensim tensorflow matplotlib seaborn
```

## Usage

Each script is designed to be run sequentially as they build upon the results of the previous steps. Here's a brief overview of how to use them:

1. **Text Preprocessing**: Prepare your text data by tokenizing and cleaning it using the provided functions.

2. **Word Embedding Training**: Train the Word2Vec model on your processed corpus with Gensim, or use the TensorFlow model for a neural network-based approach.

3. **Visualization**: Visualize the word embeddings to analyze the relationship between different words in your corpus.

4. **Custom Word2Vec**: Optionally, explore the custom Word2Vec implementation for a deeper understanding of the algorithm's workings.

5. **Saving and Loading Embeddings**: Learn how to save your word embeddings for later use and how to load them back into your environment.

## Example

Here's a quick example to get you started with preprocessing text and training a Word2Vec model:

```python
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec

# Sample text
text = ["The quick brown fox jumps over the lazy dog."]

# Preprocess the text
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text.lower())
cleaned_tokens = [token for token in tokens if token not in stopwords.words('english')]

# Train a Word2Vec model
model = Word2Vec([cleaned_tokens], vector_size=100, window=5, min_count=1, workers=4)

# Explore the model
print(model.wv.most_similar('fox'))
```

## Contribution

Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the MIT License.
