import json
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3

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
	plot_ups_and_downs()
