#   this file is used to run an interactive webapp
#   for the model discussed in model.ipynb file

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

from fft_model import FFTModel
from regression_model import RegressionModel


def add_features(df_, pivot_date):
    df_["day"] = df_.index.day
    df_["day_of_week"] = df_.index.day_of_week
    df_["day_of_year"] = df_.index.day_of_year
    df_["isweekend"] = np.int0(df_.index.weekday > 4)
    df_["week"] = df_.index.weekofyear
    df_["month"] = df_.index.month
    df_["quarter"] = df_.index.quarter
    df_["date_id"] = (df_.index-pivot_date).days
    return df_


pivot_date = None
unique_items = None


def run_to_serialize():
    #   load data

    file_path = "./streamlit_files/train-sales.csv"
    df = pd.read_csv(file_path, parse_dates=["date"])

    #   drop null redundant columns

    df.drop("store", axis=1, inplace=True)

    #   grouping data and date index

    df = df.groupby(["date", "item"]).sum().reset_index()
    df.set_index("date", inplace=True)

    #   drop < 2014
    mask = df.index.year < 2014
    df = df[~mask]

    #   Add new features
    pivot_date = df.index.min()
    unique_items = df['item'].unique().astype(str).tolist()
    df = add_features(df, pivot_date)

    #   Split to features and target

    X, y = df.drop("sales", axis=1), df["sales"]

    max_id = df['item'].unique().max()
    items = df['item'].unique().tolist()

    #   Regression Model
    reg_models = np.empty((max_id+1), dtype=np.poly1d)

    for item in items:
        mask = X['item'] == item
        X_ = X[mask]
        y_ = y[mask]
        reg_model = RegressionModel()
        reg_model.fit(X_["date_id"], y_)
        reg_models[item] = reg_model.get_reg_poly()
        #   DeTrend
        y.loc[mask] = y.loc[mask] - \
            reg_models[item](y.loc[mask])
    print("Reg models ready!")

    #   Fourier Model
    fft_models = np.zeros((max_id+1, 366))

    for item in items:
        mask = X['item'] == item
        X_ = X[mask]
        y_ = y[mask]
        fft_model = FFTModel(slice_=[0, 366])
        fft_model.fit(None, y_)
        fft_models[item] = fft_model.get_fit_array()
        #   DeTrend
        y.loc[mask] = y.loc[mask] - \
            fft_models[item, X.loc[mask, "day_of_year"]-1]
    print("FFT models ready!")

    #   Poly Model
    poly_models = np.zeros((max_id+1, 7))

    for item in items:
        mask = X['item'] == item
        X_ = X[mask]
        y_ = y[mask]
        poly_model = RegressionModel(4)
        poly_model.fit(X_["day_of_week"], y_)
        poly_models[item] = \
            poly_model.get_reg_poly()(np.arange(7))
        #   DeTrend
        y.loc[mask] = y.loc[mask] - \
            poly_models[item, X.loc[mask, "day_of_week"]]
    print("Poly models ready!")

    #   RandomForest Model

    rf_model = RandomForestRegressor(max_depth=10, n_estimators=50)
    rf_model.fit(X.drop("date_id", axis=1), y)
    print("RF model ready!")
    data = {}
    data["models"] = [rf_model, poly_models, fft_models, reg_models, ]
    data["variables"] = [pivot_date, unique_items]
    return data


rf_model, poly_models, fft_models, reg_models = [None]*4
try:
    data = load('./streamlit_files/model.joblib')
    rf_model, poly_models, fft_models, reg_models = data['models']
    pivot_date, unique_items = data["variables"]
except:
    print("Model dump loading failed!")
    print("Refitting Model!")
    data = run_to_serialize()
    rf_model, poly_models, fft_models, reg_models = data['models']
    pivot_date, unique_items = data["variables"]
    print('Dumping Model!')
    dump(data, './streamlit_files/model.joblib')


def run_models(df_: pd.DataFrame):
    items = df_['item'].unique().tolist()
    result = pd.DataFrame(df_[["item"]])
    result[['rf', 'poly', 'fft', 'reg']] = None
    for item in items:
        mask = df_['item'] == item
        #   Random Forest
        result.loc[mask, 'rf'] = rf_model.predict(
            df_.loc[mask].drop("date_id", axis=1))
        #   Polynomial
        result.loc[mask, "poly"] =\
            poly_models[item, df_.loc[mask, "day_of_week"]]
        #   Fourier Transform
        result.loc[mask, "fft"] = \
            fft_models[item, df_.loc[mask, "day_of_year"]-1]
        #   Linear Regression
        result.loc[mask, "reg"] = \
            reg_models[item](df_.loc[mask, "date_id"])
    result["total"] = \
        result[['rf', "poly", "fft", "reg"]].sum(axis=1)
    return result


def predict(start, period, items_to_predict):
    predict_df = pd.DataFrame()
    for item in items_to_predict:
        predict_df_ = pd.DataFrame()
        index = pd.date_range(start=start, periods=period)
        predict_df_['date'] = index
        predict_df_['item'] = item
        predict_df = pd.concat([predict_df, predict_df_])
    predict_df.set_index("date", inplace=True)
    predict_df = add_features(predict_df, pivot_date)
    result_df = run_models(predict_df)
    return result_df
