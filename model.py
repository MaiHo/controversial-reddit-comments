from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

import preprocess as pp

def run_models():
	X_train, X_test, y_train, y_test = pp.preprocess()

	clf = DecisionTreeClassifier(criterion="entropy")
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_train)

	train_error = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
	print "Training error is ", train_error

if __name__ == "__main__":
	run_models()