from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
# Generate alpha score list between 0.01 and 1
alpha_scores = list(np.linspace(0.01, 1, 100))
# Add scores between 1 and 25
alpha_scores.extend(list(np.linspace(1, 100,100000)))
lasso_params = {'alpha': alpha_scores}
data_preprocessed = pd.read_json("data/owi-covid-values_imputed.json")
x_data = data_preprocessed.loc[:, data_preprocessed.columns != "new_deaths_smoothed"]
y_data = data_preprocessed.loc[:, data_preprocessed.columns == "new_deaths_smoothed"]

t1 = time()
# Activate with Multiprocessing, params, and 5 fold CV
lasso_grid_search_cv = GridSearchCV(linear_model.Lasso(), param_grid=lasso_params, n_jobs=-1, cv=5, verbose=1)
lasso_grid_search_cv.fit(x_data, y_data)
t2 = time()
print("\n\n")
print(f"LASSO Regression: Best Score {lasso_grid_search_cv.best_score_}")
print(f"LASSO Regression: Best Parameter {lasso_grid_search_cv.best_params_}")
print(f"\nExecution Time: {timedelta(seconds=(t2 - t1))}")
mean_times = lasso_grid_search_cv.cv_results_["mean_fit_time"]
std_times = lasso_grid_search_cv.cv_results_["std_fit_time"]
mean_score = lasso_grid_search_cv.cv_results_["mean_test_score"]
std_score = lasso_grid_search_cv.cv_results_["std_test_score"]
param_alphas = np.array(lasso_grid_search_cv.cv_results_["param_alpha"], dtype=float)

grid_search_scores_lasso = pd.DataFrame(
    {"mean_times": mean_times, "std_times": std_times, "mean_score": mean_score, "std_score": std_score,
     "param_alphas": param_alphas}
)
grid_search_scores_lasso.to_excel("data/lasso_grid_search_results.xlsx")

plt.plot(param_alphas, mean_score)
plt.xlabel("Alpha")
plt.ylim(0, 1)
plt.ylabel("Mean R^2 Score")
plt.savefig("data/lasso_grid_search_results.png")
plt.show()