"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:         https://github.com/amr-amr/COMP550-Project
Data folder:    https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
from __future__ import print_function
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import copy
import pandas as pd
from constants import DATA_DIRECTORY
from model import ModelFactory, TextSequence
from matplotlib import pyplot as plt
from dtos import ExperimentData, ExperimentParameters
from helpers import ensure_folder_exists
from results_analysis import ModelResultsAnalyzer, TestResultsManager
from data_preprocessing import load_data

import numpy as np
from caching import WordIndexCache


def train_dev_split(df_train_dev, train_percent=0.9):
    nb_train = int(len(df_train_dev) * train_percent)
    return df_train_dev[:nb_train], df_train_dev[nb_train:]


class ExperimentWrapper:

    def __init__(self):
        self.model_factory = ModelFactory()
        self.test_results_manager = TestResultsManager()

    def run(self, train_data: ExperimentData, dev_data: ExperimentData,
            test_data: ExperimentData, params: ExperimentParameters):
        results_folder = os.path.join(DATA_DIRECTORY, 'results', params.file_name())
        ensure_folder_exists(results_folder)

        # Build model
        model = self.model_factory.create(params)
        model.summary()
        with open(os.path.join(results_folder, 'architecture.txt'), 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Run training and validation
        training_generator = TextSequence(train_data, params)
        validation_params = copy.deepcopy(params)
        validation_params.batch_size = len(dev_data.x)  # Set not batch size for validation text sequence
        validation_generator = TextSequence(dev_data, validation_params)

        print("Running experiment:")
        print(params)

        check_pointer = ModelCheckpoint(filepath=os.path.join(results_folder, 'model.hdf5'), save_best_only=True,
                                        verbose=1)
        hist = model.fit_generator(training_generator, epochs=params.epochs, validation_data=validation_generator,
                                   verbose=2, callbacks=[check_pointer])

        # Save model training history
        history = pd.DataFrame(hist.history)
        history.to_csv(os.path.join(results_folder, 'history.csv'), encoding='utf-8', index=False)

        # Plot training and validation accuracy
        fig = plt.figure(figsize=(12, 12))
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.title('%s\nTraining and Validation Accuracy' % params.__str__())
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.show()
        fig.savefig(os.path.join(results_folder, 'accuracy_plot'))

        # Evaluate test set
        model.load_weights(check_pointer.filepath)
        self.evaluate_test_set(model, test_data, params, results_folder)

    def evaluate_test_set(self, model, test_data: ExperimentData, params: ExperimentParameters, results_folder):
        test_generator = TextSequence(test_data, params)
        test_df = test_data.df

        y_pred = np.round(model.predict_generator(test_generator))
        labels = ['good', 'bad']
        test_df['y_pred_label'] = ['good' if i == 1 else 'bad' for i in y_pred]
        test_df['true_label'] = ['good' if i == 1 else 'bad' for i in test_df['label']]

        mra = ModelResultsAnalyzer(test_df, labels, "true_label", ["y_pred_label"])
        metrics = mra.get_metrics("y_pred_label")
        confusion_matrix = mra.get_cm("y_pred_label")
        print(metrics)
        print(confusion_matrix)
        self.test_results_manager.save_result(params, metrics)

        with open(os.path.join(results_folder, 'metrics.txt'), 'w') as f:
            print(metrics, file=f)
            print(confusion_matrix, file=f)


if __name__ == '__main__':

    # Sample of all possible settings:
    # ---------------------------------
    # dropout = 0.5
    # epochs = 20
    # pos_dim = 10
    # batch_size = 256
    # sent_dim = [200, 400]
    # nn_models = ['cnn', 'lstm', 'ff']
    # train_wv = [False, True]
    # use_pos = [None, 'embed', 'one_hot']
    # use_parse = [None, 'filt', 'concat']
    # ---------------------------------
    dropout = 0.5
    epochs = 20
    pos_dim = 10
    batch_size = 256
    sent_dim = [300]
    nn_models = ['ff']
    train_wv = [False]
    use_pos = [None]
    use_parse = [None]

    df_train_val, df_test = load_data('df_train.pkl', 'df_test.pkl')

    if not WordIndexCache.is_initialized():
        WordIndexCache.initialize(list(df_train_val['spacy_text']) + list(df_test['spacy_text']))

    df_train, df_dev = train_dev_split(df_train_val)

    experiment_wrapper = ExperimentWrapper()
    exp_params = [ExperimentParameters(nn_model=nn, dropout=dropout, epochs=epochs, sent_dim=sd,
                                       batch_size=batch_size, train_wv=wvt, use_pos=upos, use_parse=uparse)
                  for nn in nn_models for sd in sent_dim for wvt in train_wv for upos in use_pos for uparse in use_parse]

    for ep in exp_params:
        experiment_wrapper.run(ExperimentData.from_df(df_train),
                               ExperimentData.from_df(df_dev),
                               ExperimentData.from_df(df_test), ep)
