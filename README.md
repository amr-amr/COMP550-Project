# COMP 550 Project - Group 1
### Effect of Additional Linguistic Information on Word Embeddings  

- Comp 550 - Final Project - Fall 2018    
- Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)  
- Implemented using Python 3, keras and tensorflow

## Project Overview
Sentiment analysis is an important application of Natural Language Processing (NLP) with widespread use in industry, where distilled information on opinion can be a significant source of value. Typically, this task is framed as a text classification problem whereby a text document is categorized within predefined sentiment labels. 

Over the past decade, deep learning (DL) has become a popular machine learning approach, yielding state-of-the-art results on such text classification tasks. The input features to these DL models are generally word embeddings which encode word co-occurrence information; and by extension, some degree of semantic information.

However, this approach ignores explicit syntactic information traditionally used in NLP, such as dependency parse trees and part-of-speech (POS) tags. There is conflicting information as to whether or not this information can improve performance of DL models on NLP tasks such as text classification for sentiment analysis. To address this, we propose a novel approach for adding POS tag and dependency parse information to the input features of three different DL models (CNN, BiLSTM and simple feed-forward). We then isolate and evaluate the impact of this approach on polarity classification in the IMDB review dataset.
