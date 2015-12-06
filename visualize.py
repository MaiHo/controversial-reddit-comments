from collections import defaultdict

import itertools
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
    plt.style.use('ggplot')

    if controversial:
        plt.title("Scores of Controversial Comments")
    else:
        plt.title("Scores of Uncontroversial Comments")

    plt.ylabel("Score (net upvotes)")

    # Plot data
    plt.boxplot(dataframe.score.values)

    if not controversial:
        plt.ylim(-100, 200)
    plt.show()

    # Print mean, median, mode, 25th and 75th percentiles
    print "Average score: ", np.mean(dataframe.score.values)
    print "Median: ", np.median(dataframe.score.values)
    print "25th Percentile: ", np.percentile(dataframe.score.values, 25)
    print "75th Percentile: ", np.percentile(dataframe.score.values, 75)
    print "Minimum score: ", min(dataframe.score.values)
    print "Maximum score: ", max(dataframe.score.values)


def plot_most_controversial_subreddits():
    sql_conn = sqlite3.connect('data/sample_controversial.sqlite')
    dataframe = pd.read_sql('SELECT subreddit FROM May2015', sql_conn)

    counts = pd.DataFrame({'count' : dataframe.groupby("subreddit").size()}).reset_index().values

    # [::-1] reverses the list.
    sorted_counts = counts[:, 1].argsort(axis=False)[::-1]
    sorted_counts = sorted_counts[:20]

    plt.figure()
    plt.style.use('ggplot')
    plt.title("Subreddits With Most Controversial Comments")
    plt.ylabel("Subreddit")
    plt.xlabel("Number of Controversial Comments")
    plt.yticks(range(20), counts[sorted_counts, 0])
    plt.barh(range(20), counts[sorted_counts, 1], align="center")
    plt.tight_layout()
    plt.show()

def plot_alt_most_controversial_subreddits():
    sql_conn = sqlite3.connect('../data/sample_2.sqlite')
    df = pd.read_sql('SELECT subreddit, controversiality FROM May2015 WHERE subreddit in (SELECT subreddit FROM May2015 GROUP BY subreddit HAVING COUNT(*) > 3000)', sql_conn)

    counts = df.groupby(['subreddit', 'controversiality'])['subreddit'].count().unstack('controversiality').fillna(0)
    plt.style.use('ggplot')
    counts.plot(kind='barh', stacked=True)
    plt.title("Subreddits with Most Controversial Comments")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_most_controversial_subreddits()
    plot_alt_most_controversial_subreddits()
