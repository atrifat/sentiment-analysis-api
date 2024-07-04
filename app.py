import os
from dotenv import load_dotenv
import pandas as pd
from flask import Flask, request, jsonify
import datetime
import logging
import torch
from transformers import pipeline

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "production")
LISTEN_HOST = os.getenv("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = os.getenv("LISTEN_PORT", "5000")
SENTIMENT_ANALYSIS_MODEL = os.getenv(
    "SENTIMENT_ANALYSIS_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

APP_VERSION = "0.0.1"

# Setup logging configuration
LOGGING_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
if APP_ENV == "production":
    logging.basicConfig(
        level=logging.INFO,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        datefmt=LOGGING_DATE_FORMAT,
        format=LOGGING_FORMAT,
    )

app = Flask(__name__)

sentiment_task = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_ANALYSIS_MODEL,
    tokenizer=SENTIMENT_ANALYSIS_MODEL,
)


def perform_sentiment_analysis(query):
    result = []
    temp_result = []
    default_result = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    result = default_result

    try:
        temp_result = sentiment_task(query, top_k=3)

        for i, item in enumerate(temp_result):
            result[item["label"]] = item["score"]

    except Exception as e:
        logging.error(e)

    return result


@app.errorhandler(Exception)
def handle_exception(error):
    res = {"error": str(error)}
    return jsonify(res)


@app.route("/detect", methods=["POST"])
def predict():
    data = request.json
    q = data["q"]
    start_time = datetime.datetime.now()
    result = perform_sentiment_analysis(q)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    logging.debug("elapsed detection time: %s", str(elapsed_time))
    return jsonify(result)


@app.route("/", methods=["GET"])
def index():
    response = {"message": "Use /detect route to get detection result"}
    return jsonify(response)


@app.route("/app_version", methods=["GET"])
def app_version():
    response = {"message": "This app version is ".APP_VERSION}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host=LISTEN_HOST, port=LISTEN_PORT)
