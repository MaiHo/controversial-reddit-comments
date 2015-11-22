from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import preprocess as pp

def run_models():
	print "Beginning preprocessing..."
	X_train, X_test, y_train, y_test = pp.preprocess()
	print "...finished preprocessing"

	print "Beginning training..."
	clf = DecisionTreeClassifier(criterion="entropy")
	clf.fit(X_train, y_train)
	print "...finished training"

	y_pred = clf.predict(X_train)
	print "Training accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

	print_metrics(clf, X_test, y_test)

def print_metrics(clf, X_test, y_test):
	y_pred = clf.predict(X_test)

	print
	print "Metrics"
	print
	print "\tTesting accuracy is ", metrics.accuracy_score(y_test, y_pred, normalize=True)
	print "\tPrecision score is ", metrics.precision_score(y_test, y_pred)
	print "\tRecall score is ", metrics.recall_score(y_test, y_pred)
	print "\tF1 score is ", metrics.f1_score(y_test, y_pred)



if __name__ == "__main__":
	run_models()