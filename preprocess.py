from collections import defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import json
import numpy as np

COMMENTS_FILENAME = "data/data_20_comments"
FEATURES = set(["body", "downs", "ups", "score", "controversiality", "gilded", "edited", "subreddit"])

def read_comments():
	with open(COMMENTS_FILENAME) as f:
		for line in f:
			comment = json.loads(line)
			yield {feature: comment[feature] for feature in FEATURES}

def find_percentage_controverisial_comments(output_filename):
	num_controversial = 0
	total = 0
	comments = read_comments()

	with open(output_filename, 'wr') as f:
		for comment in comments:
			total += 1

			if comment["controversiality"] == 1:
				num_controversial += 1
				f.write(json.dumps(comment) + "\n")

		pct = 100.0 * num_controversial / total
		print "% of controversial comments", pct

		return pct

def find_most_controversial_subreddits():
	subreddits = defaultdict(int)
	comments = read_comments()

	for comment in comments:
		if comment["controversiality"] == 1:
			subreddits[comment["subreddit"]] += 1

	sorted_subreddits =\
		reversed(sorted(subreddits.keys(), key=lambda x: subreddits[x]))
	return sorted_subreddits[:20]


def preprocess():
	comments = read_comments()
	vectorizer = CountVectorizer(binary=True, stop_words="english")
	X = vectorizer.fit_transform((comment["body"] for comment in comments))

	comments = read_comments()
	y = np.array([[comment["controversiality"]] for comment in comments])
	
	numerical_features = ["downs", "ups", "score", "gilded"]
	for feature in numerical_features:
		comments = read_comments()
		X_2 = csr_matrix([[comment[feature]] for comment in comments])
		X = hstack([X, X_2])

	boolean_features = ["edited"]
	for feature in boolean_features:
		comments = read_comments()
		X_2 = csr_matrix([[int(comment[feature])] for comment in comments])
		X = hstack([X, X_2])

	X_train, X_test, y_train, y_test = train_test_split(X, y,
		test_size=0.33, random_state=42)

	return X_train, X_test, y_train, y_test


if __name__ == "__main__":
	pass
