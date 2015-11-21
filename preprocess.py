from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

import json

FEATURES = set(["body", "downs", "ups", "score", "controversiality", "gilded", "edited", "subreddit"])

def read_file(filename):
	data = []
	with open(filename) as f:
		for line in f:
			comment = json.loads(line)
			data.append({feature: comment[feature] for feature in FEATURES})

	return data

def find_percentage_controverisial_comments(data):
	


def preprocess(data):
	vectorizer = CountVectorizer(binary=True, stop_words="english")
	X = vectorizer.fit_transform((d["body"] for d in data))


if __name__ == "__main__":
	data = read_file("data/data_20_comments")
	preprocess(data)
