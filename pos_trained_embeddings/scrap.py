from keras import Sequential
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Concatenate
import pandas as pd

import numpy as np



NLTK_POS_LENGTH = 45
SENTENCE_LENGTH = 200
model = Sequential()
# model.add(Input(batch_shape=()))
model.add(Embedding(NLTK_POS_LENGTH, NLTK_POS_LENGTH, input_length=SENTENCE_LENGTH, embeddings_initializer='truncated_normal'))

# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be
# no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

# 32 elements, 10 words tags per element
# input_array = np.random.randint(1000, size=(32, 10))
df = pd.read_pickle('nltk_pos_int_dataframe.pkl')
input_array = np.squeeze(np.array(list(df['pos'])))

model.compile('rmsprop', 'mse')

# 32 elements, 10 words tags per element, encoded as 64-element embedding
output_array = model.predict(input_array)


# Concatenate(axis=2)(embedding_layer, input_layer)
#
# assert output_array.shape == (32, 10, 64)

