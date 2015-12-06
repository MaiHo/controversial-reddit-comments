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
PERCENTAGE_OF_TEST_SET = 0.20

def read_comments():
    sql_conn = sqlite3.connect('data/sample_1mil.sqlite')

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

    return save_or_load_training_testing_sets(filenames,
        lambda: create_baseline_training_testing_sets())

def preprocess(max_features=5):
    filenames = ["X_train_normal",
        "X_test_normal",
        "y_train_normal",
        "y_test_normal"]

    return save_or_load_training_testing_sets(filenames,
        lambda: create_training_testing_sets(max_features))

def save_or_load_training_testing_sets(filenames, create_fcn):
    files = [(filename, read_pickle(filename)) for filename in filenames]
    if any([matrix is None for _, matrix in files]):
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
    return create_training_testing_sets(is_baseline=True)

def create_training_testing_sets(max_features=0, is_baseline=False):
    # Ensure determinism.
    np.random.seed(42)

    comments_dataframe = read_comments()
    num_comments, _ = comments_dataframe.shape

    train_indices, test_indices, _, _ =\
        train_test_split(range(num_comments), range(num_comments),
        test_size=PERCENTAGE_OF_TEST_SET, random_state=42)

    # Change test indices to represent correct distribution of ~2.4% controversial comments.
    test_set_comments = comments_dataframe.iloc[test_indices]

    test_set_uncontroversial = test_set_comments[comments_dataframe["controversiality"] == 0].index.tolist()

    test_set_controversial = test_set_comments[comments_dataframe["controversiality"] == 1].index.tolist()
    test_set_controversial = np.random.choice(test_set_controversial, len(test_set_controversial) - 97524)

    test_indices = np.append(test_set_uncontroversial, test_set_controversial)

    if is_baseline:
        X = comments_dataframe[["subreddit"]].as_matrix()
    else:
        controversial_df = comments_dataframe.iloc[train_indices][comments_dataframe["controversiality"] == 1]
        vectorizer = CountVectorizer(binary=True, stop_words="english", max_features=max_features)
        vectorizer.fit(controversial_df["body"])

        # Now get bag of words vector for each comment
        X = vectorizer.transform(comments_dataframe["body"])
    
        numerical_features = ["score", "gilded", "edited", "subreddit"]
        for feature in numerical_features:
            X_2 = csr_matrix(comments_dataframe[feature].values)
            X = hstack([X, X_2.transpose()])

        # Convert to another sparse array format that supports advanced indexing.
        X = X.tocsc()

    y = comments_dataframe["controversiality"].as_matrix()

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

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
