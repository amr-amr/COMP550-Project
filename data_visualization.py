"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import load_imdb_dataset
from caching import WordIndexCache, EmbeddingsCache

def analyze_sentence_lengths(x_train):
    sent_word_counts = [len(x.split()) for x in x_train]
    _ = plt.hist(sent_word_counts, bins=[10 * x for x in range(200)])

    plt.title("Sentence length frequency")
    plt.xlabel('Sentence length')
    plt.ylabel('Frequency')

    print("Mean: %f\nStdDev: %f\nMin: %f\nMax: %f\n"
          % (np.mean(sent_word_counts),
             np.std(sent_word_counts),
             np.min(sent_word_counts),
             np.max(sent_word_counts)))


def calculate_oov(word_index, embeddings):
    return sum([1 for w in word_index.keys() if w not in embeddings]) / len(word_index)


if __name__ == '__main__':
    (train_x, train_labels), (test_x, test_labels) = load_imdb_dataset()
    analyze_sentence_lengths(train_x)
    calculate_oov(WordIndexCache.word_index, EmbeddingsCache.get_embeddings())