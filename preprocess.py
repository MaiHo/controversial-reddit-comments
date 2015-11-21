from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import json

COMMENTS_FILENAME = "data/data_538238"
FEATURES = set(["body", "downs", "ups", "score", "controversiality", "gilded", "edited", "subreddit"])

def read_comments():
	with open(COMMENTS_FILENAME) as f:
		for line in f:
			comment = json.loads(line)
			yield {feature: comment[feature] for feature in FEATURES}

def find_percentage_controverisial_comments():
	num_controversial = 0
	total = 0
	comments = read_comments()
	for comment in comments:
		total += 1

		if comment["controversiality"] == 1:
			num_controversial += 1

	print num_controversial
	print total
	pct = 100.0 * num_controversial / total
	print "% of controversial comments", pct
	return pct

def preprocess():
	comments = read_comments()
	vectorizer = CountVectorizer(binary=True, stop_words="english")
	X = vectorizer.fit_transform((comment["body"] for comment in comments))


if __name__ == "__main__":
	find_percentage_controverisial_comments()
	preprocess()
