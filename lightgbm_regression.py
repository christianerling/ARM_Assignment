import warnings
from time import time

import lightgbm as lgb
import numpy as np
import pandas as pd
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
                "Lithuania", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovenia",
                "Spain", "Sweden", "United Kingdom"]
eu_countries = list(map(lambda x: "location_" + str(x), eu_countries))
original_back = list(data_preprocessed[eu_countries].idxmax(axis=1))
original_back = list(map(lambda x: x.replace("location_", ""), original_back))

data_preprocessed = data_preprocessed.drop(eu_countries, axis=1)
data_preprocessed["location"] = original_back
data_preprocessed["location"] = data_preprocessed["location"].astype('category')
# t1 = time()
#
#
# # Activate with Multiprocessing, params, and 5 fold CV
# # Objective Function
# def lgb_r2_score(preds, dtrain):
#     labels = dtrain.get_label()
#     return 'r2', r2_score(labels, preds), True
#
#
# dtrain = lgb.Dataset(data=x_data, label=y_data)
#
#
# def bayesion_opt_lgbm(X, y, init_iter=3, n_iters=7, random_state=11, seed=101, num_iterations=100):
#     dtrain = lgb.Dataset(data=X, label=y)
#
#     def lgb_r2_score(preds, dtrain):
#         labels = dtrain.get_label()
#         return 'r2', r2_score(labels, preds), True
#
#     # Objective Function
#     def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight):
#         params = {'application': 'regression', 'num_iterations': num_iterations,
#                   'learning_rate': 0.05, 'early_stopping_round': 50,
#                   'metric': 'lgb_r2_score'}  # Default parameters
#         params["num_leaves"] = int(round(num_leaves))
#         params['feature_fraction'] = max(min(feature_fraction, 1), 0)
#         params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
#         params['max_depth'] = int(round(max_depth))
#         params['min_split_gain'] = min_split_gain
#         params['min_child_weight'] = min_child_weight
#         params['verbose'] = -1
#         cv_results = lgb.cv(params, dtrain, nfold=5, seed=seed, categorical_feature=[], stratified=False,
#                             verbose_eval=None, feval=lgb_r2_score)
#         # print(cv_results)
#         return np.max(cv_results['r2-mean'])
#
#     # Domain space-- Range of hyperparameters
#     pds = {'num_leaves': (80, 100),
#            'feature_fraction': (0.1, 0.9),
#            'bagging_fraction': (0.8, 1),
#            'max_depth': (17, 25),
#            'min_split_gain': (0.001, 0.1),
#            'min_child_weight': (10, 25)
#            }
#
#     # Surrogate model
#     optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)
#
#     # Optimize
#     optimizer.maximize(init_points=init_iter, n_iter=n_iters)
#
#
# t1 = time()
# print("\n\nLightGBM:")
# print("\nBayesian Optimization:")
# bayesion_opt_lgbm(x_data, y_data, init_iter=5, n_iters=100, random_state=77, seed=101, num_iterations=200)
# t2 = time()
# print(f"\n\nExecution Time {timedelta(seconds=t2 - t1)}")

mean_result = []
# max_bin=63 add below if device is GPU
lm = lgb.LGBMRegressor(bagging_fraction=0.9404, feature_fraction=0.3161, max_depth=int(18.21),
                       min_child_weight=17.38, min_split_gain=0.03066, num_leaves=int(94.24),
                       application="regression", num_iterations=200, learning_rate=0.05, metric='lgb_r2_score',
                       device="cpu", n_jobs=-1, gpu_use_dp=False, categorical_column=24)
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
        y_pred = lm.predict(X_test)
        t2 = time()
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        cv_result.append([r2, mse, rmse, mae, mape, t2 - t1])
    means = list(np.mean(np.array(cv_result), axis=0))
    mean_result.append(means)

pd.DataFrame(mean_result, columns=["R2", "MSE", "RMSE", "MAE", "MAPE", "Execution Time"]).to_excel(
    "data/lightgbm_cv_run.xlsx")
