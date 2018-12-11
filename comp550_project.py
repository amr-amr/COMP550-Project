import os
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import copy
import pandas as pd

from data_generation.pos_dicts import PosDictionary
from keras_extensions import TextSequence, ExperimentParameters, ExperimentData
from model import ModelFactory
from matplotlib import pyplot as plt
from helpers import ensure_folder_exists

DATA_DIRECTORY = os.path.join('drive', 'My Drive', 'Comp550data')


def train_dev_split(df_train_dev, train_percent=0.9):
    nb_train = int(len(df_train_dev) * train_percent)
    return df_train_dev[:nb_train], df_train_dev[nb_train:]


class ExperimentWrapper:

    def __init__(self):
        self.model_factory = ModelFactory()

    def run(self, train_data: ExperimentData, dev_data: ExperimentData,
            test_data: ExperimentData, params: ExperimentParameters):
        model = self.model_factory.create(params)
        model.summary()
        training_generator = TextSequence(train_data, params)
        validation_params = copy.deepcopy(params)
        validation_params.batch_size = len(dev_data.x)
        validation_generator = TextSequence(dev_data, validation_params)
        test_generator = TextSequence(test_data, params)

        print("Running experiment:")
        print(params)

        tensor_board = TensorBoard(os.path.join(DATA_DIRECTORY, 'logs', params.file_name()))
        model_folder = os.path.join(DATA_DIRECTORY, 'models', params.file_name())
        ensure_folder_exists(model_folder)

        early_stopper = EarlyStopping(monitor='val_acc', patience=7, mode='max')
        check_pointer = ModelCheckpoint(filepath=os.path.join(model_folder, params.timestamp), save_best_only=True)

        hist = model.fit_generator(training_generator, epochs=params.epochs, validation_data=validation_generator,
                                   verbose=2, callbacks=[check_pointer, tensor_board, early_stopper])

        history = pd.DataFrame(hist.history)
        plt.figure(figsize=(12, 12))
        plt.plot(history["acc"])
        plt.plot(history["val_acc"])
        plt.title("Accuracy with pretrained word vectors")
        plt.show()

        #         model.save(best_model_path)
        model.load_weights(check_pointer.filepath)

        loss, acc = model.evaluate_generator(test_generator)
        print('Test accuracy = %f' % acc)


if __name__ == '__main__':
    df_train_dev = pd.read_pickle('df_train.pkl')
    df_test = pd.read_pickle('df_test.pkl')

    df_train, df_dev = train_dev_split(df_train_dev, 0.9)

    experiment_wrapper = ExperimentWrapper()
    exp_params = ExperimentParameters(use_pos='embed')

    train_data = ExperimentData.from_df(df_train)
    dev_data = ExperimentData.from_df(df_dev)
    test_data = ExperimentData.from_df(df_test)

    experiment_wrapper.run(train_data, dev_data, test_data, exp_params)
