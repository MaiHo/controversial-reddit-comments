import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3

DATASET_SIZE = 100000

def percentage_edited():
    sql_conn = sqlite3.connect('data/sample_controversial.sqlite')
    dataframe = pd.read_sql('SELECT * FROM May2015 WHERE edited=1', sql_conn)

    print "Number of edited controversial comments: ", len(dataframe.index)
    print "Percentage: ", 1.0 * len(dataframe.index) / DATASET_SIZE

def percentage_gilded():
    sql_conn = sqlite3.connect('data/sample_controversial.sqlite')
    dataframe = pd.read_sql('SELECT * FROM May2015 WHERE gilded=1', sql_conn)

    print "Number of gilded controversial comments: ",len(dataframe.index)
    print "Percentage: ", 1.0 * len(dataframe.index) / DATASET_SIZE

def plot_scores(controversial=True):
    # Grab comments
    if controversial:
        db = 'data/sample_controversial.sqlite'
    else:
        db = 'data/sample_uncontroversial.sqlite'
    sql_conn = sqlite3.connect(db)
    dataframe = pd.read_sql('SELECT score FROM May2015', sql_conn)

    plt.figure()
    
    if controversial:
        plt.title("Scores of Controversial Comments")
    else:
        plt.title("Scores of Uncontroversial Comments")

    plt.ylabel("Score (net upvotes)")

    # Plot data
    plt.boxplot(dataframe.score.values)
    plt.ylim(-100, 200)
    plt.show()

    # Print mean, median, mode, 25th and 75th percentiles
    print "Average score: ", np.mean(dataframe.score.values)
    print "Median: ", np.median(dataframe.score.values)
    print "25th Percentile: ", np.percentile(dataframe.score.values, 25)
    print "75th Percentile: ", np.percentile(dataframe.score.values, 75)
    print "Minimum score: ", min(dataframe.score.values)
    print "Maximum score: ", max(dataframe.score.values)

if __name__ == "__main__":
    plot_scores(controversial=False)
