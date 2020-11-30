import datetime as dt

import pandas as pd
# noinspection PyUnresolvedReferences
from nltk.corpus import stopwords

# Download the owid_covid-data CSV from https://ourworldindata.org/coronavirus-source-data and place it in
# the data folder
covid_19_raw = pd.read_csv("data/owid-covid-data.csv", encoding="utf-8")

col_selection = ["location"
    , "date"
    , "total_cases"
    , "new_cases"
    , "new_cases_smoothed"
    , "total_deaths"
    , "new_deaths"
    , "new_deaths_smoothed"
    , "reproduction_rate"
    , "icu_patients"
    , "hosp_patients"
    , "new_tests"
    , "new_tests_smoothed"
    , "new_tests_per_thousand"
    , "new_tests_smoothed_per_thousand"
    , "tests_per_case"
    , "positive_rate"
    , "stringency_index"
    , "population"
    , "population_density"
    , "median_age"
    , "aged_65_older"
    , "aged_70_older"
    , "gdp_per_capita"
    , "extreme_poverty"
    , "cardiovasc_death_rate"
    , "diabetes_prevalence"
    , "female_smokers"
    , "male_smokers"
    , "hospital_beds_per_thousand"
    , "life_expectancy"
    , "human_development_index"]
# Perform Selection of Columns
covid_19_column_selection = covid_19_raw[col_selection]
covid_19_column_selection.to_excel("data/owi-covid-data_cols_filtered.xlsx")
# Drops Rows with empty values
na_columns = ["icu_patients", "hosp_patients", "tests_per_case", "positive_rate", "stringency_index",
              "population_density", "aged_70_older", "gdp_per_capita", "extreme_poverty", "female_smokers",
              "hospital_beds_per_thousand", "total_deaths", "reproduction_rate", "new_tests", "new_tests_per_thousand",
              "total_cases",
              "new_cases", "new_deaths"]
covid_19_empty_vals_deleted = covid_19_column_selection.dropna(subset=na_columns)
# Replace empty values with zero
covid_19_empty_vals_deleted.to_excel("data/owi-covid-data_empty_vals_deleted.xlsx")
# Change datatype of columns
covid_19_empty_vals_deleted["date"] = pd.to_datetime(covid_19_empty_vals_deleted["date"])
covid_19_empty_vals_deleted['date'] = covid_19_empty_vals_deleted['date'].map(dt.datetime.toordinal)
# Hot encode the data -> Transform the location column into a pivot column
covid_19_preprocessed = pd.get_dummies(covid_19_empty_vals_deleted)
# Export To Excel
covid_19_preprocessed.to_excel("data/owi-covid-data_preprocessed.xlsx")
dtypes_covid_19 = covid_19_empty_vals_deleted.dtypes
# Check on datatypes
print(dtypes_covid_19)
covid_19_preprocessed.describe()
