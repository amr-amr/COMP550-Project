import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import copy
import pandas as pd
from constants import DATA_DIRECTORY
from model import ModelFactory, TextSequence
from matplotlib import pyplot as plt
from dtos import ExperimentData, ExperimentParameters
from helpers import ensure_folder_exists
import numpy as np


def train_dev_split(df_train_dev, train_percent=0.9):
    nb_train = int(len(df_train_dev) * train_percent)
    return df_train_dev[:nb_train], df_train_dev[nb_train:]


class ExperimentWrapper:

    def __init__(self):
        self.model_factory = ModelFactory()

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
        validation_generator = TextSequence(dev_data, params, validation=True)

        print("Running experiment:")
        print(params)

        check_pointer = ModelCheckpoint(filepath=resultsFiles.model_path(), save_best_only=True, verbose=1)
        hist = model.fit_generator(training_generator, epochs=params.epochs, validation_data=validation_generator,
                                   verbose=2, callbacks=[check_pointer])

        history = pd.DataFrame(hist.history)
        history.to_csv(resultsFiles.history_path(), encoding='utf-8', index=False)

        fig = plt.figure(figsize=(12, 12))
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.title('%s\nTraining and Validation Accuracy)' % params.__str__())
        plt.legend(['Training Accuracy', 'Validation Accuracy'])
        plt.show()
        fig.savefig(resultsFiles.plot_path())

        # Evaluate test set
        test_generator = TextSequence(test_data, params)
        model.load_weights(check_pointer.filepath)
        test_df = test_data.df
        y_pred_score = model.predict_generator(test_generator)
        y_pred = np.round(y_pred_score)

        # convert labels
        labels = ['good', 'bad']
        test_df['y_pred_label'] = ['good' if i == 1 else 'bad' for i in y_pred]
        test_df['true_label'] = ['good' if i == 1 else 'bad' for i in test_df['label']]

        mra = ModelResultsAnalyzer(test_df, labels, "true_label", ["y_pred_label"])

        metrics = mra.get_metrics("y_pred_label")
        cm = mra.get_cm("y_pred_label")
        print(metrics)
        print(cm)

        with open(os.path.join(resultsFiles.results_folder, 'metrics.txt'), 'w') as fh:
            with contextlib.redirect_stdout(fh):
                print(metrics)
                print("\n")
                print(cm)


if __name__ == '__main__':
    df_train_dev = pd.read_pickle('df_train.pkl')
    df_test = pd.read_pickle('df_test.pkl')

    df_train, df_dev = train_dev_split(df_train_dev, 0.9)

    experiment_wrapper = ExperimentWrapper()
    exp_params = ExperimentParameters(dropout=0.5, epochs=10, batch_size=128)

    train_data = ExperimentData.from_df(df_train)
    dev_data = ExperimentData.from_df(df_dev)
    test_data = ExperimentData.from_df(df_test)

    experiment_wrapper.run(train_data, dev_data, test_data, exp_params)
