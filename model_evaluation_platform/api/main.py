from typing import Optional
from fastapi import FastAPI
from utils import *
from datetime import datetime
# import numpy as np
# import joblib
import multiprocessing as mp
import time
import itertools

# for debugging purpose
import uvicorn

app = FastAPI()

# evaluation_stocks_path = "../../data/processed/stocks_for_evaluate/"
# SCALER_PATH = "../../scaler"
# template_filename_test_x = "{}/{}_test_X.npy"
# template_filename_test_y = "{}/{}_test_y.npy"

default_evaluation_stocks = split_stock_names("1038.HK,1299.HK,2888.HK,AAPL,MSFT,TEAM")

@app.get("/")
def api_root():
    return "Hello world! The Model Evaluation Platform API is operating normally."

@app.get(
    "/evaluate/",
    summary="Evaluate the stocks",
    description="Evaluate the BGRU and GAN model with stocks provided, stock list should be separated by comma (\",\"), " +
                "e.g.  1038.HK,1299.HK,2888.HK,AAPL,MSFT,TEAM. Also, the date should be YYYY-mm-dd ",
         )
def evaluate(stocks: str = "1038.HK,1299.HK,2888.HK,AAPL,MSFT,TEAM",
             start_date: Optional[datetime] = datetime(2019, 10, 1),
             end_date: Optional[datetime] = datetime(2021, 10, 1)
             ):
    start_calc_time = time.time()
    stock_names = split_stock_names(stocks)
    # evaluation_result = []

    # do the calculation in with multiprocessing
    with mp.Pool(mp.cpu_count()) as pp:
        evaluation_result = pp.starmap(do_evaluate, [(stock, start_date, end_date, default_evaluation_stocks) for stock in stock_names])

    # check whether the stock data exists or not, if it is default evaluation stock, it should exists
    # for stock in stock_names:

        # X_value = None
        # y_value = None
        # y_scaler = None
        # tmp_stock_short_name = None
        #
        # if stock in default_evaluation_stocks:
        #     X_value= np.load(
        #             template_filename_test_x.format(
        #                 evaluation_stocks_path,
        #                 stock
        #             )
        #         )
        #     y_value = np.load(
        #             template_filename_test_y.format(
        #                 evaluation_stocks_path,
        #                 stock
        #             )
        #         )
        #     y_scaler = joblib.load(SCALER_PATH + "/" + "{}.y_scaler.joblib".format(stock))
        #
        # else:
        #     # no existing dataset found, proceed to fetch data and data preprocessing
        #     tmp_stock, tmp_stock_short_name = fetch_data(stock, start_date, end_date)
        #     X_value, y_value, y_scaler = preprocess_data(tmp_stock)
        #
        #
        # tmp_evaluate_result = evaluate_model(stock, tmp_stock_short_name, X_value, y_value, y_scaler)

        # if not evaluation_result: # if evaluation_result is empty list
        #     evaluation_result = tmp_evaluate_result
        # else:
        #     evaluation_result = evaluation_result + tmp_evaluate_result

    end_calc_time = time.time()

    print("Used time: {}".format(end_calc_time - start_calc_time))
    # return evaluation_result
    return list(itertools.chain.from_iterable(evaluation_result))

# for debugging purpose
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)