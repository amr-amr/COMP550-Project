"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:
Script containing classes related to the analysis of prediction results reported from trained models
"""
import pandas as pd
import numpy as np
import os
from constants import DATA_DIRECTORY
from dtos import ExperimentParameters


class ModelResultsAnalyzer:

    def __init__(self, pred_df, labels, true_y_col, model_pred_cols):
        self.df = pred_df
        self.labels = labels
        self.true_y_col = true_y_col
        self.model_pred_cols = model_pred_cols
        self.df_counts = self._create_df_counts()
        self.df_cm_dict = self._create_df_cm_dict()

    def _create_df_counts(self):
        # create confusion matrix multi index columns
        multi_index_cols = []
        for model_pred in self.model_pred_cols:
            for l_true in self.labels:
                col_true = "true_" + str(l_true)
                for l_pred in self.labels:
                    col_pred = "pred_" + str(l_pred)
                    multi_index_cols.append((model_pred, col_true, col_pred))

        # create confusion matrix dataframe
        df_cm = pd.DataFrame(np.zeros((len(self.df), len(multi_index_cols))),
                             columns=pd.MultiIndex.from_tuples(multi_index_cols))

        # fill confusion matrix dataframe
        Y = self.true_y_col
        for y in self.model_pred_cols:
            for i, r in self.df.iterrows():
                cm_col = (y,
                          "true_" + str(r[Y]),
                          "pred_" + str(r[y]))
                df_cm.loc[i, cm_col] = 1

        return df_cm

    def _create_df_cm(self, model):
        # create rows and columns of confusion matrix
        rows = ["true_" + str(l) for l in self.labels]
        rows.append("total")
        cols = ["pred_" + str(l) for l in self.labels]
        cols.append("total")

        # create confusion matrix
        cm = pd.DataFrame(0, index=rows, columns=cols)
        for l_true in self.labels:
            row = "true_" + str(l_true)
            for l_pred in self.labels:
                col = "pred_" + str(l_pred)
                count = self.df_counts[model][row][col].sum()
                cm.loc[row, col] = count

                # increment total
                cm.loc["total", col] += count
                cm.loc[row, "total"] += count

        return cm

    def _create_df_cm_dict(self):
        df_cm_dict = {}
        for model in self.model_pred_cols:
            df_cm_dict[model] = self._create_df_cm(model)

        return df_cm_dict

    def get_cm(self, model):
        return self.df_cm_dict[model]

    def get_metrics(self, model):
        metrics = {}
        metrics["accuracy"] = 0
        cm = self.df_cm_dict[model]
        for l in self.labels:
            true = "true_" + str(l)
            pred = "pred_" + str(l)

            metrics["accuracy"] += cm.loc[true, pred]

            metrics[l] = {}
            p = cm.loc[true, pred] / cm.loc["total", pred]
            metrics[l]["precision"] = p
            r = cm.loc[true, pred] / cm.loc[true, "total"]
            metrics[l]["recall"] = r
            metrics[l]["f1-score"] = 2 / (1 / p + 1 / r)
        metrics["accuracy"] /= cm.loc["total", :].sum()

        return metrics

    def get_samples(self, text_col, model, true_y, pred_y, baseline_model=None):
        true_l = "true_" + true_y
        pred_l = "pred_" + pred_y
        df = self.df_counts
        if baseline_model is not None:
            _ = self._create_comparison_df(baseline_model, model)
            df = self.df_counts
            df_to_return = self.df[text_col].loc[(df[model][true_l][pred_l] == 1) &
                                                 (df['same_pred'] == False)]
            return df_to_return.tolist()

        else:
            df = self.df_counts
            return self.df[text_col].loc[df[model][true_l][pred_l] == 1].tolist()

    def _create_comparison_df(self, baseline_model, model):
        self.df_counts['same_pred'] = True
        for i, r in self.df_counts.iterrows():
            if r[baseline_model].tolist() != r[model].tolist():
                self.df_counts.loc[i, 'same_pred'] = False
        df = self.df_counts.loc[self.df_counts['same_pred'] == False]
        return df

    def _create_comparison_df_cm(self, baseline_model, model):
        # create rows and columns of confusion matrix
        rows = ["true_" + str(l) for l in self.labels]
        rows.append("total")
        cols = ["pred_" + str(l) for l in self.labels]
        cols.append("total")

        # get comparison df
        df = self._create_comparison_df(baseline_model, model)

        # create confusion matrix
        cm = pd.DataFrame(0, index=rows, columns=cols)
        for l_true in self.labels:
            row = "true_" + l_true
            for l_pred in self.labels:
                col = "pred_" + l_pred
                count = df[model][row][col].sum()
                cm.loc[row, col] = count

                # increment total
                cm.loc["total", col] += count
                cm.loc[row, "total"] += count

        return cm


class TestResultsManager:

    def __init__(self):

        self._df_lookup = {}
        models = ['cnn', 'lstm', 'ff']

        for nn_model in models:
            file_path = os.path.join(DATA_DIRECTORY, 'results', '%s_test_results.csv' % nn_model)
            if os.path.exists(file_path):
                print('Loading existing results summary %s' % file_path)
                df = pd.read_csv(file_path)
            else:
                df = pd.DataFrame(columns=['model', 'sent_dim', 'train_wv', 'use_pos', 'use_parse',
                                           'baseline', 'accuracy', 'acc_vs_baseline', 'f1-score', 'f1-score_vs_baseline',
                                           'precision', 'precision_vs_baseline', 'recall', 'recall_vs_baseline'])

            self._df_lookup[nn_model] = (df, file_path)

    def save_result(self, params: ExperimentParameters, metrics):

        baseline = params.get_baseline()
        model_name = baseline if params.is_baseline() else params.get_name()
        (df, results_path) = self._df_lookup[params.nn_model]
        existing_entry = df[df['model'] == model_name]

        accuracy = metrics['accuracy']
        f1score = metrics['good']['f1-score']
        precision = metrics['good']['precision']
        recall = metrics['good']['recall']

        if existing_entry.empty:
            # Add new entry
            df = df.append(
                {'model': model_name,
                 'sent_dim': params.sent_dim,
                 'train_wv': params.train_wv,
                 'use_pos': str(params.use_pos),
                 'use_parse': str(params.use_parse),
                 'baseline': baseline,
                 'accuracy': accuracy,
                 'f1-score': f1score,
                 'precision': precision,
                 'recall': recall,
                 'acc_vs_baseline': 0}, ignore_index=True)
        else:
            # Update accuracy of existing entry
            df.loc[df['model'] == model_name, 'accuracy'] = accuracy
            df.loc[df['model'] == model_name, 'f1-score'] = f1score
            df.loc[df['model'] == model_name, 'precision'] = precision
            df.loc[df['model'] == model_name, 'recall'] = recall

        # Update accuracies versus baseline
        baseline_model = df.loc[df['model'] == baseline]

        if not baseline_model.empty:
            baseline_metrics = baseline_model.iloc[0]
            df['acc_vs_baseline'] = df.loc[df['baseline'] == baseline] \
                .apply(lambda x: x['accuracy'] - baseline_metrics['accuracy'], axis=1)
            df['f1-score_vs_baseline'] = df.loc[df['baseline'] == baseline] \
                .apply(lambda x: x['f1-score'] - baseline_metrics['f1-score'], axis=1)
            df['precision_vs_baseline'] = df.loc[df['baseline'] == baseline] \
                .apply(lambda x: x['precision'] - baseline_metrics['precision'], axis=1)
            df['recall_vs_baseline'] = df.loc[df['baseline'] == baseline] \
                .apply(lambda x: x['recall'] - baseline_metrics['recall'], axis=1)

        self._df_lookup[params.nn_model] = (df, results_path)
        df.to_csv(results_path, index=False)


if __name__ == '__main__':
    trm = TestResultsManager()
    trm.save_result(ExperimentParameters(), {'accuracy': 0.9, 'good': {'f1-score': 0.9, 'precision': 0.9, 'recall': 0.9}})
    trm.save_result(ExperimentParameters(use_pos='embed'), {'accuracy': 0.8, 'good': {'f1-score': 0.7, 'precision': 0.7, 'recall': 0.7}})
    trm.save_result(ExperimentParameters(use_pos='one_hot'), {'accuracy': 0.7, 'good': {'f1-score': 0.2, 'precision': 0.1, 'recall': 0.4}})
    trm.save_result(ExperimentParameters(nn_model='cnn'), {'accuracy': 0.4, 'good': {'f1-score': 0.5, 'precision': 0.2, 'recall': 0.4}})
    trm.save_result(ExperimentParameters(nn_model='cnn', use_pos='one_hot'), {'accuracy': 0.3, 'good': {'f1-score': 0.4, 'precision': 0.2, 'recall': 0.5}})
