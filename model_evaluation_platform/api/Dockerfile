# Define working env (python version & os)
FROM python:3.9.12-slim
#FROM tensorflow/tensorflow:2.8.0-gpu
# https://hub.docker.com/r/tensorflow/tensorflow Might need nvidia-docker to run

# Define working dir
WORKDIR /api

# Copy necessary files
# API requirements.txt
COPY ./model_evaluation_platform/api/requirements.txt /api/requirements.txt

# Project files (codes)
COPY ./model_evaluation_platform/api/* /api/

# models, scalers & default datasets
COPY ./model/hk/* /api/model/hk/
COPY ./model/us/* /api/model/us/
COPY ./data/processed/stocks_for_evaluate/* /api/data/processed/stocks_for_evaluate/
COPY ./scaler/* /api/scaler

# Install packages
RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN pip install -r /api/requirements.txt \
    && rm -rf /root/.cache/pip

# Run the app
#CMD ["uvicorn", "main:app", "--port", "8000"]