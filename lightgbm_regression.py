import warnings
from datetime import timedelta
from itertools import chain
from time import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import seaborn as sns
from bayes_opt import BayesianOptimization
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

warnings.filterwarnings("ignore")


def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


data_preprocessed = pd.read_json("data/owi-covid-values_imputed.json")
x_data = data_preprocessed.loc[:, data_preprocessed.columns != "new_deaths_smoothed"]
y_data = data_preprocessed.loc[:, data_preprocessed.columns == "new_deaths_smoothed"]

# Convert the pivot columns for the location back to location column i.o. to speed up the execution on GPU
eu_countries = ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
                "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Luxembourg",
                "Lithuania", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovenia", "Slovakia",
                "Spain", "Sweden", "United Kingdom"]
eu_countries = list(map(lambda x: "location_" + str(x), eu_countries))
original_back = list(data_preprocessed[eu_countries].idxmax(axis=1))
original_back = list(map(lambda x: x.replace("location_", ""), original_back))

data_preprocessed = data_preprocessed.drop(eu_countries, axis=1)
data_preprocessed["location"] = original_back
data_preprocessed["location"] = data_preprocessed["location"].astype('category')


# Activate with Multiprocessing, params, and 5 fold CV
# Objective Function
def lgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds), True


dtrain = lgb.Dataset(data=x_data, label=y_data)


def bayesion_opt_lgbm(X, y, init_iter=3, n_iters=7, random_state=11, seed=101, num_iterations=100):
    dtrain = lgb.Dataset(data=X, label=y)

    def lgb_r2_score(preds, dtrain):
        labels = dtrain.get_label()
        return 'r2', r2_score(labels, preds), True

    # Objective Function
    def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight):
        params = {'application': 'regression', 'num_iterations': num_iterations,
                  'learning_rate': 0.05, 'early_stopping_round': 50,
                  'metric': 'lgb_r2_score'}  # Default parameters
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['verbose'] = -1
        cv_results = lgb.cv(params, dtrain, nfold=5, seed=seed, categorical_feature=[], stratified=False,
                            verbose_eval=None, feval=lgb_r2_score)
        # print(cv_results)
        return np.max(cv_results['r2-mean'])

    # Domain space-- Range of hyperparameters
    pds = {'num_leaves': (80, 100),
           'feature_fraction': (0.1, 0.9),
           'bagging_fraction': (0.8, 1),
           'max_depth': (17, 25),
           'min_split_gain': (0.001, 0.1),
           'min_child_weight': (10, 25)
           }

    # Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)

    # Optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)


# t1 = time()
# print("\n\nLightGBM:")
# print("\nBayesian Optimization:")
# bayesion_opt_lgbm(x_data, y_data, init_iter=10, n_iters=200, random_state=77, seed=101, num_iterations=400)
# t2 = time()
# print(f"\n\nExecution Time {timedelta(seconds=t2 - t1)}")
# data = pd.read_excel("data/lightgbm_bayesian_optimization.xlsx")
# fig = make_subplots(rows=1, cols=6,
#                     subplot_titles=["bagging_fraction", "feature_fraction", "max_depth", "min_child_weight",
#                                     "min_split_gain", "num_leaves"])
# fig.add_trace(
#     go.Scatter(x=data["bagging_fraction"], y=data["target"], mode='markers', name="bagging_fraction"),
#     row=1, col=1
# )
# fig.add_trace(
#     go.Scatter(x=data["feature_fraction"], y=data["target"], mode='markers', name="feature_fraction"),
#     row=1, col=2
# )
# fig.add_trace(
#     go.Scatter(x=data["max_depth"], y=data["target"], mode='markers', name="max_depth"),
#     row=1, col=3
# )
# fig.add_trace(
#     go.Scatter(x=data["min_child_weight"], y=data["target"], mode='markers', name="min_child_weight"),
#     row=1, col=4
# )
# fig.add_trace(
#     go.Scatter(x=data["min_split_gain"], y=data["target"], mode='markers', name="min_split_gain"),
#     row=1, col=5
# )
# fig.add_trace(
#     go.Scatter(x=data["num_leaves"], y=data["target"], mode='markers', name="num_leaves"),
#     row=1, col=6
# )
# fig.update_layout(height=600, width=1500, title_text="Hyperparameter for Target Variable R\u00b2")
# plotly.offline.plot(fig, filename='data/lightgbm_bayesian_optimization_result.html', auto_open=True)
mean_result = []
# max_bin=63 add below if device is GPU
lm = lgb.LGBMRegressor(bagging_fraction=0.9133, feature_fraction=0.5429, max_depth=int(24.99),
                       min_child_weight=11.66, min_split_gain=0.008908, num_leaves=int(84.88),
                       application="regression", num_iterations=200, learning_rate=0.05, metric='lgb_r2_score',
                       device="cpu", n_jobs=-1, gpu_use_dp=False, categorical_column=24)
predicted = []
true_vals = []
feature_imp = dict()
for i in tqdm(range(1200)):
    cv_result = []
    indices = []
    s_split = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8)
    for train_index, test_index in s_split.split(x_data):
        indices.append([train_index, test_index])
        X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        t1 = time()
        lm.fit(X_train, y_train)
        coeff = np.abs(lm.feature_importances_)
        rel_func = lambda x: x / np.sum(coeff)
        coeff = rel_func(coeff)

        for counter, column in enumerate(x_data.columns):
            if column in feature_imp.keys():
                feature_imp[column].append(coeff[counter])
            else:
                feature_imp.update({column: [coeff[counter]]})
        y_pred = lm.predict(X_test)
        t2 = time()
        predicted.append(y_pred.tolist())
        true_vals.append(y_test["new_deaths_smoothed"].tolist())
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        cv_result.append([r2, mse, rmse, mae, mape, t2 - t1])
    means = list(np.mean(np.array(cv_result), axis=0))
    mean_result.append(means)

for key, value in feature_imp.items():
    feature_imp[key] = np.mean(feature_imp[key])
imp_coef = pd.Series(feature_imp)
imp_coef = pd.DataFrame(imp_coef).reset_index()
imp_coef.columns = ["Feature", "Value"]
imp_coef = imp_coef.sort_values(by="Value", ascending=False)
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=imp_coef)
plt.title('Relative LightGBM Feature Importance (mean over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png', dpi=200)
plt.show()

pd.DataFrame(mean_result, columns=["R2", "MSE", "RMSE", "MAE", "MAPE", "Execution Time"]).to_excel(
    "data/lightgbm_cv_run.xlsx")

predicted = list(chain.from_iterable(predicted))
true_vals = list(chain.from_iterable(true_vals))
regression_res_collected = dict()

for counter, true_val in enumerate(true_vals):
    if true_val in regression_res_collected.keys():
        regression_res_collected[true_val].append(predicted[counter])
    else:
        regression_res_collected.update({true_val: [predicted[counter]]})

regression_res_collected = dict(sorted(regression_res_collected.items()))

for key, value in regression_res_collected.items():
    regression_res_collected[key] = [min(regression_res_collected[key]), max(regression_res_collected[key])]

max_pred = []
min_pred = []
for key, value in regression_res_collected.items():
    max_pred.append(value[1])
    min_pred.append(value[0])

fig, ax = plt.subplots()
real_vals = list(regression_res_collected.keys())

fig, ax = plt.subplots()
ax.fill_between(real_vals, min_pred, max_pred, alpha=1.0, interpolate=True, color="red", label='Prediction Interval')
ax.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], color="blue", linestyle='--', lw=4,
        label="Ideal Prediction")
ax.set_xlabel("Real new_deaths_smoothed")
ax.set_ylabel("Predicted new_deaths_smoothed")
fig.suptitle('LightGBM Regression', fontsize=16)
plt.legend(loc="upper left", frameon=False)
plt.savefig("data/lightgbm_cv_results.png", dpi=250)
plt.show()
