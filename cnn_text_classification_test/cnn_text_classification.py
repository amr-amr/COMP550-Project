from gensim.models import Word2Vec
import gensim.downloader as api
import os
import numpy as np
import pickle
import re
from keras.preprocessing.text import Tokenizer
from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# embeddings_index = {}
# f = open('data/glove.6B.100d.txt', encoding='utf8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def load_word_embeddings():
    if os.path.isfile('glove-100d.pickle'):
        print('Loading GloVe embeddings from pickle file:', 'glove-100d.pickle')
        pickle_in = open('glove-100d.pickle', 'rb')
        return pickle.load(pickle_in)

    print("Downloading GloVe embededdings:", 'glove-wiki-gigaword-100')
    embeddings_model = api.load('glove-wiki-gigaword-100')
    with open('glove-100d.pickle', 'wb') as f:
        pickle.dump(embeddings_model, f)

    return embeddings_model


def get_reviews(path, positive=True):
    label = 1 if positive else 0

    reviews = []
    with open(path, 'r', encoding='utf-8') as f:
        review_text = f.readlines()

    [reviews.append((clean_str(t), label)) for t in review_text]
    return reviews


def extract_reviews():
    pos_reviews = get_reviews("data/rt-polarity.pos", positive=True)
    neg_reviews = get_reviews("data/rt-polarity.neg", positive=False)
    return pos_reviews, neg_reviews


positive_reviews, negative_reviews = extract_reviews()
all_reviews = positive_reviews + negative_reviews

shuffle(all_reviews)

embeddings_lookup = load_word_embeddings()
print(len(embeddings_lookup.vectors))
# print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts([r[0] for r in all_reviews])
sequences = tokenizer.texts_to_sequences([r[0] for r in all_reviews])
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print(word_index)

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
unknown_counter = 0
for word, i in word_index.items():
    if word in embeddings_lookup:
        embedding_matrix[i] = embeddings_lookup[word]
    else:
        unknown_counter = unknown_counter + 1

print(unknown_counter/len(word_index))

nb_pos_validation = int(VALIDATION_SPLIT * len(positive_reviews))
nb_neg_validation = int(VALIDATION_SPLIT * len(negative_reviews))

x_train = data[:-nb_pos_validation]
y_train = [r[1] for r in all_reviews[:-nb_pos_validation]]
x_val = data[-nb_pos_validation:]
y_val = [r[1] for r in all_reviews[-nb_pos_validation:]]

# Setup CNN network
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# 128 filters per layers, 5 kernel size row-wise
l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(1, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Simplified convolutional neural network")
model.summary()
cp=ModelCheckpoint('model_cnn.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

# results=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=15, batch_size=2,callbacks=[cp])
results=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=15, batch_size=2)