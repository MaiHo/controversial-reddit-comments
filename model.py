# ML Final Project
# Controversial Reddit Comments
#
# Authors: Mai Ho and Maury Quijada

import numpy as np
import preprocess as pp

from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def run_models():
    """ Run all classification models. """
    # TODO: Vary max_features.
    # feature_range = range(1, 31)
    # scores = {}
    # for i in feature_range:
    #     print "Beginning preprocessing..."
    #     train_test_sets = pp.preprocess(max_features=i, force_load=True)
    #     print "...finished preprocessing"

    #     print "Depth-limited Decision Tree Classifier..."
    #     scores[i] = test_decision_tree_classifier(
    #         train_test_sets, depth_limited=True)

    # max_features = max(feature_range, key=lambda x: scores[x])
    # print "Best max_features: ", max_features

    train_test_sets = pp.preprocess(max_features=max_features, force_load=True)

    # print "Beginning training..."

    # TODO: Change all metrics to F1-score once we have imbalanced data set.
    # Also change number of folds if necessary.

    # print "Baseline Classifier..."
    # # test_subreddit_baseline_classifier()

    # print "Decision Tree Classifier..."
    # # test_decision_tree_classifier(train_test_sets)

    # print "Depth-limited Decision Tree Classifier..."
    # test_decision_tree_classifier(train_test_sets, depth_limited=True)

    # print "Logistic Regression Classifier..."
    test_logistic_regression_classifier(train_test_sets)

    # print "Adaboosting with Decision Tree Stumps..."
    # test_adaboost_classifier(train_test_sets)


def performance(y_true, y_pred, metric="accuracy"):
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Args:
        y_true: true labels
        y_pred: predicted labels
        metric: type of metric used

    Returns:
        score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1

    # part 2a: compute classifier performance
    handlers = {
        "accuracy": metrics.accuracy_score,
        "f1_score": metrics.f1_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score}

    return handlers[metric](y_true, y_pred)


def cv_performance(clf, X, y, kf, metric="accuracy"):
    """
    Does k-fold cross validation.

    Args:
        clf: classifier
        X: feature vectors
        y: labels
        kf: cross_validation.StratifiedKFold
        metric: performance measure

    Returns:
        Average cross-validation performance across all k folds.
    """

    scores = np.zeros(kf.n_folds)

    i = 0
    for train_index, test_index in kf:
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores[i] = performance(y_test, y_pred, metric=metric)
        i += 1

    return np.average(scores)


def select_dt_depth(X, y, kf, metric="accuracy"):
    """
    Finds the best depth for the Decision Tree Classifier.

    Args:
        X: feature vectors
        y: labels
        kf: cross_validation.StratifiedKFold
        metric: performance measure

    Returns:
        Depth that maximizes performance on k-fold cross validation.
    """
    depths = range(1, 30)
    depth_scores = {}
    for d in depths:
        score = cv_performance(DecisionTreeClassifier(
            criterion="entropy", max_depth=d), X, y, kf, metric=metric)
        depth_scores[d] = score

    return max(depth_scores, key=lambda d: depth_scores[d])


def select_regularization(X, y, kf, classifier="logreg", metric="accuracy"):
    """
    Finds the best regularization constant for LogisticRegression.

    Args:
        X: feature vectors
        y: labels
        classifier: type of classifier, can be logreg or svm.
        kf: cross_validation.StratifiedKFold
        metric: performance measure

    Returns:
        Depth that maximizes performance on k-fold cross validation.
    """
    C = 10.0 ** np.arange(-3, 3)
    C_scores = {}
    for c in C:
        if classifier == "logreg":
            clf = LogisticRegression(C=c)
        else:
            clf = SVC(kernel='linear', C=c)
        score = cv_performance(clf, X, y, kf, metric=metric)
        C_scores[c] = score

    return max(C_scores, key=lambda c: C_scores[c])


def test_decision_tree_classifier(train_test_sets, criterion="entropy", depth_limited=False):
    """ Decision Tree Classifier with optional depth-limit.

    Args:
        train_test_sets: array of training and testing sets
        criterion: parameter for Decision Tree
        depth_limited: whether or not to prune to best depth
    """
    X_train, X_test, y_train, y_test = train_test_sets

    if depth_limited:
        # TODO: Change number of folds?
        kf = StratifiedKFold(y_train, n_folds=5, shuffle=True)
        depth = select_dt_depth(X_train, y_train, kf, metric="accuracy")
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    else:
        clf = DecisionTreeClassifier(criterion="entropy")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print "DECISION TREE CLASSIFIER RESULTS"
    print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)

    return metrics.f1_score(y_test, y_pred)


def test_logistic_regression_classifier(train_test_sets):
    """ Logistic Regression Classifier.

    Does k-fold cross validation to determine the regularization term.
    """
    X_train, X_test, y_train, y_test = train_test_sets

    # TODO: Change number of folds?
    kf = StratifiedKFold(y_train, n_folds=5, shuffle=True)
    best_C = select_regularization(
        X_train, y_train, kf, classifier="logreg", metric="accuracy")
    clf = LogisticRegression(C=best_C)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print "LOGISTIC REGRESSION CLASSIFIER RESULTS"
    print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)


def test_svm_classifier(train_test_sets):
    """ Support Vector Machine Classifier.
    """
    X_train, X_test, y_train, y_test = train_test_sets
    kf = StratifiedKFold(y_train, n_folds=5, shuffle=True)
    best_C = select_regularization(
        X_train, y_train, kf, classifier="svm", metric="accuracy")
    clf = SVC(C=best_C)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print "SVM CLASSIFIER RESULTS"
    print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)


def test_adaboost_classifier(train_test_sets):
    """ Adaboost Classifier with Decision Tree Stumps. """
    X_train, X_test, y_train, y_test = train_test_sets
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print "ADABOOST CLASSIFIER RESULTS"
    print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)


def test_subreddit_baseline_classifier():
    """ Runs baseline classifier.

    Baseline Classifier is just a decision tree with one node: subreddit.
    """
    X_train, X_test, y_train, y_test = pp.preprocess_subreddit_baseline()
    clf = DecisionTreeClassifier(criterion="entropy")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    print "SUBREDDIT BASELINE CLASSIFIER RESULTS"
    print "\tTraining accuracy is ", metrics.accuracy_score(y_train, y_pred, normalize=True)

    y_pred = clf.predict(X_test)
    print_metrics(y_test, y_pred)


def print_metrics(y_test, y_pred):
    print
    print "\tMetrics"
    print "\t\tTesting accuracy is ", metrics.accuracy_score(y_test, y_pred, normalize=True)
    print "\t\tPrecision score is ", metrics.precision_score(y_test, y_pred)
    print "\t\tRecall score is ", metrics.recall_score(y_test, y_pred)
    print "\t\tF1 score is ", metrics.f1_score(y_test, y_pred)


if __name__ == "__main__":
    run_models()
