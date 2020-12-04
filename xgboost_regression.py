from time import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm


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


# # Generate alpha score list between 0.01 and 1
xgboost_params = {
    'eta': [0.01, 0.015, 0.025, 0.05, 0.1],
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    'reg_lambda': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0],
    'gamma': [i / 10.0 for i in range(0, 5)],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'min_child_weight': [6, 8, 10, 12],
    'verbosity': [0]}
data_preprocessed = pd.read_json("data/owi-covid-values_imputed.json")
x_data = data_preprocessed.loc[:, data_preprocessed.columns != "new_deaths_smoothed"]
y_data = data_preprocessed.loc[:, data_preprocessed.columns == "new_deaths_smoothed"]

# t1 = time()
# # Activate with Multiprocessing, params, and 5 fold CV
#
# xgboost_grid_search_cv = RandomizedSearchCV(xgb.XGBRegressor(predictor="auto", tree_method="gpu_hist", nthread=-1),
#                                             param_distributions=xgboost_params,
#                                             cv=5,
#                                             verbose=1, n_jobs=-1, n_iter=1000)
# xgboost_grid_search_cv.fit(x_data, y_data)
# t2 = time()
# print("\n\n")
# print(f"XGBoost Regression: Best Score {xgboost_grid_search_cv.best_score_}")
# print(f"XGBoost Regression: Best Parameter {xgboost_grid_search_cv.best_params_}")
# print(f"\nExecution Time: {timedelta(seconds=(t2 - t1))}")
# mean_times = xgboost_grid_search_cv.cv_results_["mean_fit_time"]
# std_times = xgboost_grid_search_cv.cv_results_["std_fit_time"]
# mean_score = xgboost_grid_search_cv.cv_results_["mean_test_score"]
# std_score = xgboost_grid_search_cv.cv_results_["std_test_score"]
# subsamples = np.array(xgboost_grid_search_cv.cv_results_["param_subsample"], dtype=float)
# min_child_weights = np.array(xgboost_grid_search_cv.cv_results_["param_min_child_weight"], dtype=float)
# max_depths = np.array(xgboost_grid_search_cv.cv_results_["param_max_depth"], dtype=float)
# lambdas = np.array(xgboost_grid_search_cv.cv_results_["param_reg_lambda"], dtype=float)
# gammas = np.array(xgboost_grid_search_cv.cv_results_["param_gamma"], dtype=float)
# etas = np.array(xgboost_grid_search_cv.cv_results_["param_eta"], dtype=float)
# colsample_bytrees = np.array(xgboost_grid_search_cv.cv_results_["param_colsample_bytree"], dtype=float)
# alphas = np.array(xgboost_grid_search_cv.cv_results_["param_reg_alpha"], dtype=float)
#
# grid_search_scores_xgboost = pd.DataFrame(
#     {"mean_times": mean_times, "std_times": std_times, "mean_score": mean_score, "std_score": std_score,
#      "param_subsample": subsamples, "param_min_child_weight": min_child_weights, "param_max_depth": max_depths,
#      "param_lambda": lambdas, "param_gamma": gammas, "param_eta": etas, "colsample_bytrees": colsample_bytrees,
#      "param_alpha": alphas}
# )
# grid_search_scores_xgboost.to_excel("data/xgboost_grid_search_results.xlsx")

grid_search_scores_xgboost = pd.read_excel("data/xgboost_grid_search_results.xlsx")

# plt.plot(grid_search_scores_xgboost["param_alphas"], grid_search_scores_xgboost["mean_score"],
#          label="LASSO Regression")
# plt.xlabel("Alpha")
# plt.ylim(0, 1)
# plt.ylabel("Mean R\u00b2 Score")
# plt.legend(loc="upper right", frameon=False)
# plt.savefig("data/xgboost_grid_search_results.png")
# plt.show()

mean_result = []
lm = xgb.XGBRegressor(predictor="auto", tree_method="gpu_hist", nthread=-1, verbosity=0, subsample=0.6,
                      min_child_weight=6,
                      max_depth=3, reg_lambda=0.09, gamma=0.3, eta=0.05, colsample_bytree=0.9, reg_alpha=0.1)

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
    "data/xgboost_cv_run.xlsx")
