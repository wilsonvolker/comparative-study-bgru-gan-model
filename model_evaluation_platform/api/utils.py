# common packages
import base64
import io

import numpy as np
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# data denoising
import pywt
import copy

# feature extraction
from ta.volume import AccDistIndexIndicator
from ta.trend import MACD
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands

# feature normalization
from sklearn.preprocessing import MinMaxScaler

# dimensionality reduction
from sklearn.decomposition import PCA

# model evaluation
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


#%%
# Common variables
EVALUATION_DIAGRAM_PATH = "../../diagrams/model/evaluation"

# stocks model checkpoint paths
HK_MODELS_CHECKPOINT_PATH = "../../model/hk"
US_MODELS_CHECKPOINT_PATH = "../../model/us"

hk_bgru_file_path = "{}/bgru.h5".format(HK_MODELS_CHECKPOINT_PATH)
hk_bgru_train_history_file_path = "{}/bgru_history/bgru_training_history.npy".format(HK_MODELS_CHECKPOINT_PATH)

us_bgru_file_path = "{}/bgru.h5".format(US_MODELS_CHECKPOINT_PATH)
us_bgru_train_history_file_path = "{}/bgru_history/bgru_training_history.npy".format(US_MODELS_CHECKPOINT_PATH)

# gan
hk_gan_file_path = "{}/gan.h5".format(HK_MODELS_CHECKPOINT_PATH)
hk_gan_train_history_file_path = "{}/gan_training_history.npy".format(HK_MODELS_CHECKPOINT_PATH)

us_gan_file_path = "{}/gan.h5".format(US_MODELS_CHECKPOINT_PATH)
us_gan_train_history_file_path = "{}/gan_training_history.npy".format(US_MODELS_CHECKPOINT_PATH)

evaluation_fields_name = [
    "loss (mean_squared_error)",
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error"
]

#%%
# Functions

def parse_string_to_datetime(x):
    return datetime.strptime(x, '%Y-%m-%d')


def is_stock_hk(stock_name):
    return ".HK" in stock_name


def create_dir_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

def split_stock_names(stock_names: str) -> list[str]:
    """
    Split the stock names by comma
    :param stock_names: str, stock names joined with comma, e.g. "1038.HK,1299.HK,2888.HK,AAPL,MSFT,TEAM"
    :return: list[str]
    """
    return stock_names.split(",")

def reformat_model_names(model_name: str) -> str:
    """
    Reformat the model names
    :param model_name: str, patten like: "HK BGRU"
    :return: "BGRU (HK)"
    """
    found_words = None
    search_words = ["HK", "US"]
    for i in range(len(search_words)):
        if search_words[i] in model_name:
            found_words = search_words[i]

    model_name = model_name.replace(found_words, "").strip()
    model_name = "{} ({})".format(model_name, found_words)

    return model_name

def fetch_data(stock_symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch the stock data from yahoo finance api
    """

    stock = yf.Ticker(stock_symbol)
    hist = stock.history(start=start_date, end=end_date)

    return hist


def preprocess_data(stock_data: pd.DataFrame):
    """
    Preprocess the data
    :param stock_data: pandas.DataFrame object, stock data fetched from Yahoo Fianace API
    :return: X_value, y_value, y_scaler
    """
    # clone a new object
    raw_data = copy.deepcopy(stock_data)

    # data denoise
    tmp_close_price = raw_data["Close"]
    coeff = wavelet_denoise(tmp_close_price)
    raw_data["Close"] = np.sum(coeff, axis=0)


    # Feature extraction
    # ACD
    adi = AccDistIndexIndicator(
        high=raw_data["High"],
        low=raw_data["Low"],
        close=raw_data["Close"],
        volume=raw_data["Volume"],
        fillna=True
    )
    raw_data["ACD"] = adi.acc_dist_index()

    # MACD
    macd_instance = MACD(
        close=raw_data["Close"],
        fillna=True
    )
    # MACD line
    raw_data["MACD"] = macd_instance.macd()

    # MACD signal line
    raw_data["MACD_SIGNAL"] = macd_instance.macd_signal()

    # MACD Diff
    raw_data["MACD_DIFF"] = macd_instance.macd_diff()

    # OBV
    obv_instance = OnBalanceVolumeIndicator(
        close=raw_data["Close"],
        volume=raw_data["Volume"],
        fillna=True
    )
    raw_data["OBV"] = obv_instance.on_balance_volume()

    # Bollinger bands
    bb_instance = BollingerBands(
        close=raw_data[["High", "Close", "Low"]].mean(axis=1),  # we use mean of High, low, close
        fillna=True
    )
    # calc bollinger bands
    raw_data["MB"] = bb_instance.bollinger_mavg()
    raw_data["UB"] = bb_instance.bollinger_hband()
    raw_data["LB"] = bb_instance.bollinger_lband()


    # Feature Normalization
    # split as x and y values
    d_norm_X = copy.deepcopy(raw_data)
    # d_norm_X['Date'] = pd.to_datetime(d_norm_X['Date'])
    # d_norm_X = d_norm_X.set_index('Date')

    d_norm_y = pd.DataFrame(d_norm_X["Close"])

    # do min max normalization
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    X_scaler.fit(d_norm_X)
    y_scaler.fit(d_norm_y)

    d_norm_X = X_scaler.fit_transform(d_norm_X)
    d_norm_y = y_scaler.fit_transform(d_norm_y)

    # assert to check the shape after normalisation
    assert d_norm_X.shape == (len(raw_data), 15)
    assert d_norm_y.shape == (len(raw_data), 1)

    # Dimensionality Reduction
    pca_instance = PCA(
        n_components=2
    )
    # reduce the dimensionality of the extracted features
    tmp_pcaed = pca_instance.fit_transform(d_norm_X[:, 7:])
    pca_X = np.c_[copy.deepcopy(d_norm_X[:, :7]), tmp_pcaed]


    # Data Organization
    time_lag = 30  # days
    to_organize_x = copy.deepcopy(pca_X)
    # to_organize_x = copy.deepcopy(d_norm_x)
    to_organize_y = copy.deepcopy(d_norm_y)

    organized_X = []
    organized_y = []

    for t in range(time_lag, len(to_organize_x)):  # loop on the time
        organized_X.append(to_organize_x[t - time_lag: t][:, :])  # get features from t-30 to t
        organized_y.append(to_organize_y[t])

    organized_X = np.array(organized_X)
    organized_y = np.array(organized_y)

    # assert to check the shape after organisation
    assert organized_X.shape == (len(raw_data) - time_lag, 30, 9)
    assert organized_y.shape == (len(raw_data) - time_lag, 1)

    X_value = organized_X
    y_value = organized_y

    return X_value, y_value, y_scaler

def wavelet_denoise(index_list, wavefunc='haar', lv=2, m=1, n=2, plot=False):
    '''
    *** Reference **
    Obtained from: https://github.com/SunHao95/PHBS_TQFML-StockIndex-Wavelet-Transformation-ARIMA-ML-Model/blob/c1c3c11e80568663448bdc30f87dc378db0538d2/Project/model.py#L19-L68
    Edited to fit the project's requirements
    *** END ***

    WT: Wavelet Transformation Function
    index_list: Input Sequence;

    lv: Decomposing Level；

    wavefunc: Function of Wavelet, 'db4' default；

    m, n: Level of Threshold Processing

    '''

    # Decomposing
    coeff = pywt.wavedec(index_list, wavefunc, mode='sym',
                         level=lv)  #  Decomposing by levels，cD is the details coefficient
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0  # sgn function

    # Denoising
    # Soft Threshold Processing Method
    for i in range(m,
                   n + 1):  #  Select m~n Levels of the wavelet coefficients，and no need to dispose the cA coefficients(approximation coefficients)
        cD = coeff[i]
        Tr = np.sqrt(2 * np.log2(len(cD)))  # Compute Threshold
        for j in range(len(cD)):
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) * (np.abs(cD[j]) - Tr)  # Shrink to zero
            else:
                coeff[i][j] = 0  # Set to zero if smaller than threshold

    # Reconstructing
    coeffs = {}
    for i in range(len(coeff)):
        coeffs[i] = copy.deepcopy(coeff)
        for j in range(len(coeff)):
            if j != i:
                coeffs[i][j] = np.zeros_like(coeff[j])

    for i in range(len(coeff)):
        coeff[i] = pywt.waverec(coeffs[i], wavefunc)
        if len(coeff[i]) > len(index_list):
            coeff[i] = coeff[i][:-1]

    if plot:
        denoised_index = np.sum(coeff, axis=0)
        data = pd.DataFrame({'CLOSE': index_list, 'denoised': denoised_index})
        data.plot(figsize=(10, 10), subplots=(2, 1))
        data.plot(figsize=(10, 5))
        print(data)

    return coeff

def evaluate_model(stock_symbol: str, X_value: np.array, y_value: np.array, y_scaler) -> list:
    """
    Evaluate the model with the data provided
    :param stock_symbol: the symbol of the stock, e.g. AAPL
    :param X_value: X value of the stock data
    :param y_value: y value of the stock data
    :param y_scaler: MinMaxScaler for the y value
    :return:
    """
    # load stock full names (Get the stocks' name from raw data)
    stock_name_hk = pd.read_csv("../../data/raw/HSI Constituents list_filtered.csv")
    stock_name_hk = stock_name_hk[["Name", "Symbol"]]

    stock_name_us = pd.read_csv("../../data/raw/S&P 500 Constituents_filtered.csv")
    stock_name_us = stock_name_us[["Name", "Symbol"]]

    stock_name_eva = pd.read_csv("../../data/raw/Stocks for evaluation.csv")
    stock_name_eva = stock_name_eva[["Name", "Symbol"]]

    stock_names = pd.concat([stock_name_us, stock_name_hk, stock_name_eva], ignore_index=True)

    # load models
    hk_bgru = load_model(hk_bgru_file_path)
    hk_gan = load_model(hk_gan_file_path)
    us_bgru = load_model(us_bgru_file_path)
    us_gan = load_model(us_gan_file_path)

    hk_gan.compile()
    us_gan.compile()

    models = [hk_bgru, hk_gan, us_bgru, us_gan]
    model_names = ["HK BGRU", "HK GAN", "US BGRU", "US GAN"]

    y_true = y_scaler.inverse_transform(
        y_value
    )

    df_eva_metrics_list = []

    for i in range(len(models)):
        tmp_predict_result = models[i].predict(x=X_value)

        y_predicted = y_scaler.inverse_transform(
            tmp_predict_result
        )

        tmp_base64_img_str = plot_predicted_price(
            predicted_y=y_predicted,
            actual_y=y_true,
            title="{} ({}) Stock Price Prediction with {}".format(
                stock_names[stock_names["Symbol"] == stock_symbol]["Name"].item(),
                stock_symbol,
                model_names[i]
            ),
            predicted_y_legend_label="{} Predicted Closing Price".format(model_names[i]),
            actual_y_legend_label="Actual Closing Price",
            price_currency="HKD" if is_stock_hk(stock_symbol) else "USD"
        )

        if "GAN" in model_names[i].upper():
            # calculate metrics for GAN models
            tmp_mse = mean_squared_error(
                y_true=y_true,
                y_pred=y_predicted
            )
            tmp_mae = mean_absolute_error(
                y_true=y_true,
                y_pred=y_predicted,
            )
            tmp_rmse = mean_squared_error(
                y_true=y_true,
                y_pred=y_predicted,
                squared=False
            )
            tmp_mape = mean_absolute_percentage_error(
                y_true=y_true,
                y_pred=y_predicted
            )

            df_eva_metrics_list.append([
                "{}".format(reformat_model_names(model_names[i])),
                stock_symbol,
                tmp_mse,
                tmp_mae,
                tmp_rmse,
                tmp_mape,
                tmp_base64_img_str,
            ])

        elif "BGRU" in model_names[i].upper():
            # update MAPE value
            tmp_eva_result = models[i].evaluate(x=X_value, y=y_value)
            tmp_eva_result[-1] = mean_absolute_percentage_error(
                y_true=y_true,
                y_pred=y_predicted,
            )
            df_eva_metrics_list.append(
                ["{}".format(reformat_model_names(model_names[i])), stock_symbol] + tmp_eva_result + [tmp_base64_img_str])

    df_eva_metrics = pd.DataFrame(
        columns=["model", "stock"] + evaluation_fields_name + ["plot"],
        data=df_eva_metrics_list
    )

    return df_eva_metrics.to_dict("records")

# plot diagrams function
def plot_predicted_price(predicted_y, actual_y, title, predicted_y_legend_label, actual_y_legend_label,
                                   price_currency) -> str:

    plt.figure(figsize=(14, 5), dpi=500, facecolor="white")
    plt.plot(actual_y, label=actual_y_legend_label)
    plt.plot(predicted_y, label=predicted_y_legend_label)
    plt.xlabel('Trading Day')
    plt.ylabel(price_currency)
    plt.title(title)
    plt.legend()

    # create_dir_if_not_exist(EVALUATION_DIAGRAM_PATH)
    # plt.savefig('{}/{}.png'.format(EVALUATION_DIAGRAM_PATH, plt.gca().get_title()))
    # plt.show()

    # convert plt to base64 string and return. Ref: https://stackoverflow.com/a/38061400/9500852
    plt_IO_bytes = io.BytesIO()
    plt.savefig(plt_IO_bytes, format='png')
    plt_IO_bytes.seek(0)
    base64_png_str = base64.b64encode(plt_IO_bytes.read())

    return base64_png_str.decode('utf-8')

