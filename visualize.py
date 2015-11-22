from collections import defaultdict

import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

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
    plt.xlabel("Subreddit")
    plt.ylabel("Number of Controversial Comments")
    plt.yticks(range(20), counts[sorted_counts, 0])
    plt.barh(range(20), counts[sorted_counts, 1], align="center")
    plt.tight_layout()
    plt.show()

def plot_ups_and_downs():
    # Grab controversial comments
    sql_conn = sqlite3.connect('../input/database.sqlite')
    dataframe = pd.read_sql('SELECT ups, downs FROM May2015 WHERE CONTROVERSIAL=1 ORDER BY Random() LIMIT 5000000', sql_conn)

    plt.figure()
    plt.title("Upvotes vs Downvotes of Controversial Comments")
    plt.xlabel("Number of upvotes")
    plt.ylabel("Number of downvotes")

    # Plot data
    plt.plot(dataframe.ups.values, dataframe.downs.values)

    # Plot y=x line
    plt.plot(plt.get_xlim(), plt.get_ylim(), ls="--", c=".3")
    plt.show()

if __name__ == "__main__":
    plot_most_controversial_subreddits()
