"""
Comp 550 - Final Project - Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Script Description:

"""
import matplotlib.pyplot as plt
import numpy as np


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
