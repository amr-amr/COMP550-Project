"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
import os
import pickle


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def load_pickle(file_path):
    return pickle.load(open(file_path, 'rb')) if os.path.isfile(file_path) else None


def save_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
