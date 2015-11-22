from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import preprocess as pp

def run_models():
	print "Beginning preprocessing..."
	train_test_sets = pp.preprocess(max_features=10)
	print "...finished preprocessing"

	print "Beginning training..."
	test_subreddit_baseline_classifier()
	test_decision_tree_classifier(train_test_sets)
	print "...finished training"
	
def test_subreddit_baseline_classifier():
	X_train, X_test, y_train, y_test = pp.preprocess_subreddit_baseline()
	clf = DecisionTreeClassifier(criterion="entropy")

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_train)
	print "SUBREDDIT BASELINE CLASSIFIER RESULTS"
	print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

	y_pred = clf.predict(X_test)
	print_metrics(y_test, y_pred)


def test_decision_tree_classifier(train_test_sets, criterion="entropy"):
	X_train, X_test, y_train, y_test = train_test_sets
	clf = DecisionTreeClassifier(criterion="entropy")

	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_train)
	print "DECISION TREE CLASSIFIER RESULTS"
	print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

	y_pred = clf.predict(X_test)
	print_metrics(y_test, y_pred)

	return y_pred

def print_metrics(y_test, y_pred):
	print
	print "\tMetrics"
	print "\t\tTesting accuracy is ", metrics.accuracy_score(y_test, y_pred, normalize=True)
	print "\t\tPrecision score is ", metrics.precision_score(y_test, y_pred)
	print "\t\tRecall score is ", metrics.recall_score(y_test, y_pred)
	print "\t\tF1 score is ", metrics.f1_score(y_test, y_pred)



if __name__ == "__main__":
	run_models()