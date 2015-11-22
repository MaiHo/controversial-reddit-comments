from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import json
import numpy as np
import pandas as pd
import sqlite3

pd.options.mode.chained_assignment = None  # default='warn'

FEATURES = ["body", "score", "controversiality", "gilded", "edited", "subreddit"]

def read_comments():
    sql_conn = sqlite3.connect('data/sample.sqlite')
    dataframe = pd.read_sql('SELECT * FROM May2015', sql_conn)
    
    # Extract the relevant features and turn them into the format we want.
    relevant_dataframe = dataframe[FEATURES]
    relevant_dataframe["edited"] = relevant_dataframe["edited"].apply(lambda x: int(x > 0))

    return relevant_dataframe

def preprocess():
    comments_dataframe = read_comments()
    vectorizer = CountVectorizer(binary=True, stop_words="english", max_features=20)
    X = vectorizer.fit_transform(comments_dataframe["body"])
    y = comments_dataframe["controversiality"]
    
    numerical_features = ["score", "gilded", "edited"]
    for feature in numerical_features:
        X_2 = csr_matrix(comments_dataframe[feature].values)
        X = hstack([X, X_2.transpose()])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess()
