from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold


# TODO correct the error for zero division
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# # Generate alpha score list between 0.01 and 1
alpha_scores = list(np.linspace(0.01, 1, 100))
# Add scores between 1 and 25
alpha_scores.extend(list(np.linspace(1, 100, 100000)))
lasso_params = {'alpha': alpha_scores}
data_preprocessed = pd.read_json("data/owi-covid-values_imputed.json")
x_data = data_preprocessed.loc[:, data_preprocessed.columns != "new_deaths_smoothed"]
y_data = data_preprocessed.loc[:, data_preprocessed.columns == "new_deaths_smoothed"]
#
# t1 = time()
# # Activate with Multiprocessing, params, and 5 fold CV
# lasso_grid_search_cv = GridSearchCV(linear_model.Lasso(), param_grid=lasso_params, n_jobs=-1, cv=5, verbose=1)
# lasso_grid_search_cv.fit(x_data, y_data)
# t2 = time()
# print("\n\n")
# print(f"LASSO Regression: Best Score {lasso_grid_search_cv.best_score_}")
# print(f"LASSO Regression: Best Parameter {lasso_grid_search_cv.best_params_}")
# print(f"\nExecution Time: {timedelta(seconds=(t2 - t1))}")
# mean_times = lasso_grid_search_cv.cv_results_["mean_fit_time"]
# std_times = lasso_grid_search_cv.cv_results_["std_fit_time"]
# mean_score = lasso_grid_search_cv.cv_results_["mean_test_score"]
# std_score = lasso_grid_search_cv.cv_results_["std_test_score"]
# param_alphas = np.array(lasso_grid_search_cv.cv_results_["param_alpha"], dtype=float)
#
# grid_search_scores_lasso = pd.DataFrame(
#     {"mean_times": mean_times, "std_times": std_times, "mean_score": mean_score, "std_score": std_score,
#      "param_alphas": param_alphas}
# )
# grid_search_scores_lasso.to_excel("data/lasso_grid_search_results.xlsx")

grid_search_scores_lasso = pd.read_excel("data/lasso_grid_search_results.xlsx")

grid_search_scores_lasso_filtered = grid_search_scores_lasso[(grid_search_scores_lasso["param_alphas"] >= 60)]
plt.plot(grid_search_scores_lasso_filtered["param_alphas"], grid_search_scores_lasso_filtered["mean_score"],
         label="LASSO Regression")
plt.xlabel("Alpha")
plt.ylim(0.869, 0.8725)
plt.ylabel("Mean R\u00b2 Score")
plt.legend(loc="upper right", frameon=False)
plt.savefig("data/lasso_grid_search_results.png")
plt.show()

kf = KFold(n_splits=25, random_state=True, shuffle=True)
eval_results = []
for train_index, test_index in kf.split(x_data):
    t1 = time()
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    lm = linear_model.Lasso(alpha=86.65367653676537)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    t2 = time()
    eval_results.append([r2, mse, rmse, mae, mape, t2 - t1])

pd.DataFrame(eval_results, columns=["R2", "MSE", "RMSE", "MAE", "MAPE", "Execution Time"]).to_excel(
    "data/lasso_cv_run.xlsx")
