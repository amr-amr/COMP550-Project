"""
Comp 550 - Final Project - Fall 2018
Effects of Additional Linguistic Information on Word Embeddings
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)
Implemented using Python 3, keras and tensorflow

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:
Script containing miscellaneous helper functions related to interacting with the file system
"""
import os
import pickle


def ensure_folder_exists(folder_path):
    """
    Creates the specified folder if it does not already exist
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def load_pickle(file_path):
    """
    Loads a pickle from the file system
    """
    return pickle.load(open(file_path, 'rb')) if os.path.isfile(file_path) else None


def save_pickle(file_path, data):
    """
    Saves the specified object as a pickle to the file system
    """
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
