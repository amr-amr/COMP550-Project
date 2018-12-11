from keras.preprocessing.text import Tokenizer
import pandas as pd
from numpy import np

pkl_train = pd.read_pickle('data_generation/df_train_new.pkl')
pkl_test = pd.read_pickle('data_generation/df_test_new.pkl')

train_text = list(pkl_train['spacy_text'])
test_text = list(pkl_test['spacy_text'])

all_text = train_text + test_text

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)

print(len())

sequences = tokenizer.texts_to_sequences([r[0] for r in all_reviews])