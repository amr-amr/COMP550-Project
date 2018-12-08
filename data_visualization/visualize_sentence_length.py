import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
imdb = keras.datasets.imdb


def load_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data()

    # convert from integers to text
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    train_x = [decode_review(x) for x in train_data]
    test_x = [decode_review(x) for x in test_data]
    return (train_x, train_labels), (test_x, test_labels)


(train_x, train_labels), (test_x, test_labels) = load_data()

df = pd.DataFrame()
df['sent_dim'] = [len(x.split()) for x in train_x]
df['sent_dim'].value_counts().plot.bar()
bars = df['sent_dim'].value_counts().plot.bar()

sent_dims = [len(x.split()) for x in train_x]

matplotlib.interactive(True)

plt.hist(sent_dims)
plt.show()


