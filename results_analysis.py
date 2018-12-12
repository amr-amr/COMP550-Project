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
    _results_path = os.path.join(DATA_DIRECTORY, 'test_results_summary.csv')

    def __init__(self):
        self._df_results = pd.DataFrame(columns=['model', 'baseline', 'accuracy', 'acc_vs_baseline'])
        if os.path.exists(TestResultsManager._results_path):
            self._df_results = pd.read_csv(self._results_path)

    def save_result(self, params: ExperimentParameters, accuracy):

        baseline = params.get_baseline()
        existing_entry = self._df_results[self._df_results['model'] == params.get_name()]

        if existing_entry.empty:
            self._df_results = self._df_results.append({'model': params.get_name(), 'baseline': baseline.get_name(),
                                                        'accuracy': accuracy, 'acc_vs_baseline': 0}, ignore_index=True)
        else:
            self._df_results.loc[self._df_results['model'] == params.get_name(), 'accuracy'] = accuracy

        baseline_accuracy_set = self._df_results[self._df_results['model'] == baseline.get_name()]['accuracy']
        baseline_accuracy = None if baseline_accuracy_set.empty else baseline_accuracy_set.values[0]

        # Update accuracies versus baseline
        if baseline_accuracy is not None:
            print('%s: %.3f versus baseline %s %.3f (%.3f)' % (params.get_name(), accuracy, baseline.get_name(),
                                                               baseline_accuracy, accuracy - baseline_accuracy))

            self._df_results['acc_vs_baseline'] = self._df_results \
                .apply(lambda x: x['accuracy'] - baseline_accuracy if x['baseline'] == baseline.get_name() else x[
                'acc_vs_baseline'], axis=1)

        self._df_results.to_csv(TestResultsManager._results_path)
