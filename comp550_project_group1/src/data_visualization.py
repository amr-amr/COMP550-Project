"""
Comp 550 - Final Project - Fall 2018
Evaluating Addition of Syntactic Information in Deep Learning Models for Sentiment Analysis
Group 1 - Andrei Romascanu (260585208) - Stefan Wapnick (id 260461342)
Implemented using Python 3, keras and tensorflow

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:
Contains utility functions for visualizing the dataset
"""
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_imdb_dataset
from caching import WordIndexCache, EmbeddingsCache
from keras.datasets import imdb


def analyze_text_lengths(text_sequences):
    """
    Graphs the frequency of sequence lengths among all text sequences
    """
    sent_word_counts = [len(x.split()) for x in text_sequences]
    _ = plt.hist(sent_word_counts, bins=[10 * x for x in range(200)])

    plt.title("Sentence length frequencies")
    plt.xlabel('Sentence length')
    plt.ylabel('Frequency')
    plt.show()

    print("Mean: %f\nStdDev: %f\nMin: %f\nMax: %f\n"
          % (np.mean(sent_word_counts),
             np.std(sent_word_counts),
             np.min(sent_word_counts),
             np.max(sent_word_counts)))


def calculate_oov(word_index, embeddings):
    """
    Calculates the out of vocabulary percentage for word embeddings
    (number of words that do not have a word embedding)
    """
    return sum([1 for w in word_index.keys() if w not in embeddings]) / len(word_index)


if __name__ == '__main__':
    (train_x, train_labels), (test_x, test_labels) = load_imdb_dataset()
    analyze_text_lengths(train_x)

    print('OOV embeddings percentage (imdb word index) = %f' %
          calculate_oov(imdb.get_word_index(), EmbeddingsCache.get_wv_embeddings()))

    print('OOV embeddings percentage = %f' %
          calculate_oov(WordIndexCache.get_word_index(), EmbeddingsCache.get_wv_embeddings()))
