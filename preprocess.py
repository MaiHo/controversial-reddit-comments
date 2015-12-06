from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import json
import numpy as np
import pandas as pd
import pickle
import sqlite3

pd.options.mode.chained_assignment = None  # default='warn'

FEATURES = ["body", "score", "controversiality", "gilded", "edited", "subreddit"]
PERCENTAGE_OF_TEST_SET = 0.33

def read_comments(controversial_only=False):
    if controversial_only:
        sql_conn = sqlite3.connect('data/sample_controversial.sqlite')
    else:
        sql_conn = sqlite3.connect('data/sample.sqlite')

    dataframe = pd.read_sql('SELECT * FROM May2015', sql_conn)

    # Extract the relevant features and turn them into the format we want.
    relevant_dataframe = dataframe[FEATURES]
    relevant_dataframe["edited"] = relevant_dataframe["edited"].apply(lambda x: int(x > 0))

    all_subreddits = relevant_dataframe["subreddit"].unique()
    all_subreddits_dict = {all_subreddits[i]: i for i in range(len(all_subreddits))}
    relevant_dataframe["subreddit"] = relevant_dataframe["subreddit"].apply(lambda x: all_subreddits_dict[x])

    return relevant_dataframe

def preprocess_subreddit_baseline():
	filenames = ["X_train_baseline",
		"X_test_baseline",
		"y_train_baseline",
		"y_test_baseline"]

	return save_or_load_training_testing_sets(filenames,
		lambda: create_baseline_training_testing_sets())

def preprocess(max_features=5, force_load=False):
	filenames = ["X_train_normal",
		"X_test_normal",
		"y_train_normal",
		"y_test_normal"]

	return save_or_load_training_testing_sets(filenames,
		lambda: create_training_testing_sets(max_features), force_load)

def save_or_load_training_testing_sets(filenames, create_fcn, force_load):
	files = [(filename, read_pickle(filename)) for filename in filenames]
	if any([matrix is None for _, matrix in files]) or force_load:
		print "Preprocessed files not found, preprocessing from scratch..."
		training_testing_sets = create_fcn()

		files = zip(filenames, list(training_testing_sets))
		for filename, matrix in files:
			save_pickle(filename, matrix)

		return training_testing_sets

	else:
		print "Preprocessed files found, no further preprocessing necessary..."
		return tuple([x for _, x in files])


def create_baseline_training_testing_sets():
    # Subreddit baseline model is a decision tree that only considers a subreddit
    # Therefore, subreddit is the only feature.
    comments_dataframe = read_comments()
    X = comments_dataframe[["subreddit"]]
    y = comments_dataframe["controversiality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=PERCENTAGE_OF_TEST_SET, random_state=42)

    return X_train, X_test, y_train, y_test


def create_training_testing_sets(max_features):
    controversial_df = read_comments(controversial_only=True)
    vectorizer = CountVectorizer(binary=True, stop_words="english", max_features=max_features)
    vectorizer.fit(controversial_df["body"])

    # Now get bag of words vector for each comment
    comments_dataframe = read_comments()
    X = vectorizer.transform(comments_dataframe["body"])
    y = comments_dataframe["controversiality"].as_matrix()

    numerical_features = ["score", "gilded", "edited", "subreddit"]
    for feature in numerical_features:
        X_2 = csr_matrix(comments_dataframe[feature].values)
        X = hstack([X, X_2.transpose()])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=PERCENTAGE_OF_TEST_SET, random_state=42)

    return X_train, X_test, y_train, y_test

def read_pickle(prefix):
    try:
        with open(get_pickle_filename(prefix), "rb") as f:
            var = pickle.load(f)
            return var
    except StandardError:
        return None

def save_pickle(prefix, var):
	with open(get_pickle_filename(prefix), "wb") as f:
		pickle.dump(var, f)
		return var

def get_pickle_filename(prefix):
	return prefix + ".pickle"

