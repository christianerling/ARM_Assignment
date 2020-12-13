import warnings
from itertools import chain
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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


# # # Generate alpha score list between 0.01 and 1
alpha_scores = []  # list(np.linspace(0.01, 1, 100))
# # Add scores between 1 and 25
alpha_scores.extend(list(np.linspace(0, 200, 20000)))
lasso_params = {'alpha': alpha_scores}
data_preprocessed = pd.read_json("data/owi-covid-values_imputed.json")
x_data = data_preprocessed.loc[:, data_preprocessed.columns != "new_deaths_smoothed"]
y_data = data_preprocessed.loc[:, data_preprocessed.columns == "new_deaths_smoothed"]

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

# grid_search_scores_lasso = pd.read_excel("data/lasso_grid_search_results.xlsx")
#
# grid_search_scores_lasso_filtered = grid_search_scores_lasso  # grid_search_scores_lasso[(grid_search_scores_lasso["param_alphas"] >= 60)]
# plt.plot(grid_search_scores_lasso_filtered["param_alphas"], grid_search_scores_lasso_filtered["mean_score"],
#          label="LASSO Regression")
# plt.xlabel("Alpha")
# plt.ylim(0.25, 1.0)
# plt.xlim(0, 10)
# plt.ylabel("Mean R\u00b2 Score")
# plt.legend(loc="upper right", frameon=False)
# plt.savefig("data/lasso_grid_search_results.png", dpi=250)
# plt.show()

lm = linear_model.Lasso(alpha=0.01000050002500125)
mean_result = []
predicted = []
true_vals = []
for i in tqdm(range(1200)):
    cv_result = []
    indices = []
    s_split = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.8)
    for train_index, test_index in s_split.split(x_data):
        indices.append([train_index, test_index])
        X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
        y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
        t1 = time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            lm.fit(X_train, y_train)
        # coef = pd.Series(lm.coef_, index=x_data.columns)
        # imp_coef = pd.DataFrame(coef).reset_index()
        # imp_coef.columns = ["Feature", "Value"]
        # imp_coef["Value"] = imp_coef["Value"].abs()
        # imp_coef = imp_coef.sort_values(by="Value", ascending=False)
        # plt.figure(figsize=(20, 10))
        # sns.barplot(x="Value", y="Feature", data=imp_coef)
        # plt.title('LASSO Features (avg over folds)')
        # plt.tight_layout()
        # plt.savefig('lasso_importances-01.png', dpi=200)
        # plt.show()
        y_pred = lm.predict(X_test)
        # for i, v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i, v))
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

pd.DataFrame(mean_result, columns=["R2", "MSE", "RMSE", "MAE", "MAPE", "Execution Time"]).to_excel(
    "data/lasso_cv_run.xlsx")

predicted = list(chain.from_iterable(predicted))
true_vals = list(chain.from_iterable(true_vals))

fig, ax = plt.subplots()
ax.scatter(true_vals, predicted, edgecolors=(0, 0, 0))
ax.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 'k--', lw=4, label="Real Values")
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
fig.suptitle('LASSO Regression', fontsize=16)
plt.legend(loc="upper right", frameon=False)
plt.savefig("data/lasso_cv_results.png", dpi=250)
plt.show()
