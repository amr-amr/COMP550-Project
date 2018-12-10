import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from data_generation.SpacyPosAndParse import SpacyPosAndParse
from time import time

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


sda = SpacyPosAndParse(spacy_model="en_core_web_md")

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()

word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_x = [decode_review(x) for x in train_data]
test_x = [decode_review(x) for x in test_data]

df = pd.DataFrame(columns=['text', 'pos', 'parse'])
df['text'] = test_x
df['label'] = train_labels

print('Starting to parse text')
start = time()
df[['pos', 'parse']] = df.apply(lambda row: sda.parse_text(row['text']), axis=1, result_type='expand')
print(time() - start)
print(df)

df.to_pickle('spacy_data_test.pkl')

# df = pd.read_pickle('test.pkl')
# df.info(memory_usage='deep')