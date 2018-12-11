import pandas as pd
import numpy as np


class ModelResultsAnalyzer:
    def __init__(self, pred_df, labels, true_y_col, model_pred_cols):
        self.df = pred_df
        self.labels = labels
        self.true_y_col = true_y_col
        self.model_pred_cols = model_pred_cols
        self.df_counts = self._create_df_counts()
        self.df_cm_dict = self._create_df_cm_dict()
        # TODO: prevent recalculation of comparison df each time?


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
                          "true_"+str(r[Y]),
                          "pred_"+str(r[y]))
                df_cm.loc[i, cm_col] = 1

        return df_cm

    def _create_df_cm(self, model):
        # create rows and columns of confusion matrix
        rows = ["true_"+str(l) for l in self.labels]
        rows.append("total")
        cols = ["pred_"+str(l) for l in self.labels]
        cols.append("total")

        # create confusion matrix
        cm = pd.DataFrame(0, index=rows, columns=cols)
        for l_true in self.labels:
            row = "true_"+str(l_true)
            for l_pred in self.labels:
                col = "pred_"+str(l_pred)
                count = self.df_counts[model][row][col].sum()
                cm.loc[row, col] = count

                # increment total
                cm.loc["total", col] += count
                cm.loc[row,"total"] += count

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
        cm = self.df_cm_dict[model]
        for l in self.labels:
            true = "true_"+str(l)
            pred = "pred_"+str(l)

            metrics[l] = {}
            p = cm.loc[true, pred] / cm.loc["total", pred]
            metrics[l]["precision"] = p
            r = cm.loc[true, pred]/cm.loc[true, "total"]
            metrics[l]["recall"] = r
            metrics[l]["f1-score"] = 2/(1/p + 1/r)

        return metrics

    def get_samples(self, text_col, model, true_y, pred_y, baseline_model = None):
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
        for i,r in self.df_counts.iterrows():
            if r[baseline_model].tolist() != r[model].tolist():
                self.df_counts.loc[i, 'same_pred'] = False
        df = self.df_counts.loc[self.df_counts['same_pred'] == False]
        return df

    def _create_comparison_df_cm(self, baseline_model, model):
        # create rows and columns of confusion matrix
        rows = ["true_"+str(l) for l in self.labels]
        rows.append("total")
        cols = ["pred_"+str(l) for l in self.labels]
        cols.append("total")

        # get comparison df
        df = self._create_comparison_df(baseline_model, model)

        # create confusion matrix
        cm = pd.DataFrame(0, index=rows, columns=cols)
        for l_true in self.labels:
            row = "true_"+l_true
            for l_pred in self.labels:
                col = "pred_"+l_pred
                count = df[model][row][col].sum()
                cm.loc[row, col] = count

                # increment total
                cm.loc["total", col] += count
                cm.loc[row,"total"] += count

        return cm


test_df = pd.read_pickle('df_test_new.pkl')

# test_data = ExperimentData.from_df(df_test, pos_col='nltk_pos')
# test_df = test_data.df

# from keras.models import load_model
# model = load_model("drive/My Drive/Comp550data/results/nn_model=lstmbatch_size=512use_pos=embeduse_parse=falsesent_dim=200pos_dim=20dropout=0.50train_wv=true/models/2018-12-11_20-34-41.mdl")
# test_generator = TextSequence(test_data, exp_params)
# test_df['ypreds'] = np.round(model.predict_generator(test_generator))

np.random.seed(8008135)

# test_df['y_pred'] = ['good' if i == 1 else 'bad' for i in np.round(np.random.random(len(test_df)))]
# test_df['y_pred2'] = ['good' if i == 1 else 'bad' for i in np.round(np.random.random(len(test_df)))]
test_df['y_pred'] = ['good' if i == 1 else 'bad' for i in np.round(np.random.random(len(test_df)))]
test_df['y_pred2'] = ['good' if i == 1 else 'bad' for i in np.round(np.random.random(len(test_df)))]
test_df['label'] = ['good' if i == 1 else 'bad' for i in test_df['label']]
labels = ['good', 'bad']
test_df['label'].unique().tolist()

# df = pd.get_dummies(df, 'label')
mra = ModelResultsAnalyzer(test_df[:100], labels, "label", ["y_pred", 'y_pred2'])

samples = mra.get_samples('text', 'y_pred', true_y='good', pred_y='bad', baseline_model='y_pred2')
[print(s) for s in samples]

print(len(samples))

cm = mra._create_comparison_df_cm("y_pred2", "y_pred")
print(cm)

print(mra._create_comparison_df("y_pred2", "y_pred"))