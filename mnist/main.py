import pickle
import os
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib as mpl
import matplotlib.pyplot as plot

def load_pickle(file_name):
    if os.path.isfile(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)

def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def train_binary(x_train, x_test, y_train, y_test):
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix
    from sklearn.base import clone

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    classifier = SGDClassifier(random_state=42)
    classifier.fit(x_train, y_train_5)

    # # Automatic cross validation
    # validation = cross_val_score(classifier, x_train, y_train_5, cv=3, scoring="accuracy")
    # print(validation)
    # # Manual cross validation
    # skfolds = StratifiedKFold(n_splits=3, random_state=42)
    # for train_index, test_index in skfolds.split(x_train, y_train_5):
    #     clone_clf = clone(classifier)
    #     x_train_folds = x_train[train_index]
    #     y_train_folds = y_train_5[train_index]

    #     x_test_fold = x_train[test_index]
    #     y_test_fold = y_train_5[test_index]

    #     clone_clf.fit(x_train_folds, y_train_folds)
    #     y_pred = clone_clf.predict(x_test_fold)
    #     n_correct = sum(y_pred == y_test_fold)
    #     print(n_correct / len(y_pred))

    # Confusion matrix
    # Rows: a class from the classifier, in this case, true or false
    # Cols: a predicited class from the classifier.
    # [negatives TN, false positives FP]
    # [false negatives FN, positives TP]
    # y_train_pred = cross_val_predict(classifier, x_train, y_train_5, cv=3)
    # cmatrix = confusion_matrix(y_train_5, y_train_pred)
    # print(cmatrix)

    # from sklearn.metrics import precision_score, recall_score, f1_score

    # # Accuracy of positive predictions: Precision = TP / (TP + FP)
    # precision = precision_score(y_train_5, y_train_pred)
    # print(f'Precision: {round(precision*100, 4)}% correctly classified a 5 correctly')

    # # Recall, Sensitivity, True Positive Rate (TPR) = TP / (TP + FN)
    # sensitivity = recall_score(y_train_5, y_train_pred)
    # print(f'Sensitivity: {round(sensitivity*100, 4)}% of the 5s were detected')

    # F1 Score (Harmonic Mean) = 2 ((P * S) / (P + S))
    # useful when comparing two classifiers
    # A high F1 score means both P and S have to both be high
    # favors a similar P and S, which is not always desireable
    # harmonic_mean = f1_score(y_train_5, y_train_pred)
    # print(f'F1 Score: {round(harmonic_mean*100, 4)}%')
    # An increase in P means a decrease in S. And vise-versa. Precision/Recall tradeoff

    # getting the actual y scores instead of just the validation score
    # y_scores = cross_val_predict(classifier, x_train, y_train_5, cv=3, method="decision_function")

    # from sklearn.metrics import precision_recall_curve
    # precisions, sensitivies, thresholds = precision_recall_curve(y_train_5, y_scores)
    # # Plot precision and recall againt threshold
    # plot.plot(thresholds, precisions[:-1], 'b--', label="Precision")
    # plot.plot(thresholds, sensitivies[:-1], 'g-', label="Sensitivity")
    # plot.xlabel("Threshold")
    # plot.legend(loc="center left")
    # plot.ylim([0, 1])
    # plot.show()
    # # Plot precision against recall
    # plot.plot(sensitivies[:-1], precisions[:-1], 'b--')
    # plot.xlabel("Sensitivity")
    # plot.ylabel("Precision")
    # plot.legend(loc="center left")
    # plot.show()

    # Plots the ROC Curve. FPR against TPR. The best classifier stays
    # as far away from the line (0,1) -> (1,0) as possible. Which in this
    # graph will be the top left corner (the curve of the FPR vs TPR)
    # from sklearn.metrics import roc_curve
    # FPR, TPR, thresholds = roc_curve(y_train_5, y_scores)
    # plot.plot(FPR, TPR)
    # plot.plot([0, 1], [0, 1], 'k--')
    # plot.axis([0, 1, 0, 1])
    # plot.xlabel("False Positive Rate")
    # plot.ylabel("True Positive Rate")
    # plot.show()

    # ROC AUC is the area under the FPR vs TPR to (0,1)->(1,0)
    # A perfect classifier will have 100% ROCAUC. A purely
    # random classifier will have 50% ROCAUC
    # from sklearn.metrics import roc_auc_score
    # rocauc = roc_auc_score(y_train_5, y_scores)
    # print(f'ROCAUC: {round(rocauc*100, 4)}')

def train_knearest(x_train, x_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix

    # print('Training classifier')
    # classifier = KNeighborsClassifier(
    #     n_neighbors=5,
    #     weights="uniform",
    #     algorithm="auto",
    #     leaf_size=30,
    #     p=2,
    #     metric="minkowski",
    #     n_jobs=8
    # )
    # classifier.fit(x_train, y_train)

    #save_pickle('knearest', classifier)

    print('Loading classifier')
    classifier = load_pickle('knearest')

    # print('Cross Validating Three Folds')
    # y_train_pred = cross_val_predict(
    #     classifier,
    #     x_train,
    #     y_train,
    #     cv=3,
    #     n_jobs=8,
    #     method="predict_proba"
    # )

    # save_pickle('knearest_val', y_train_pred)

    # print('Loading y_train_pred')
    y_train_pred = load_pickle('knearest_val')

    # print('Prediciting test set')
    # y_pred = classifier.predict(x_test)
    # from sklearn.metrics import accuracy_score
    # from sklearn.metrics import mean_squared_error
    # print(accuracy_score(y_test, y_pred))

    from sklearn.metrics import roc_curve
    for col_index in range(y_train_pred.shape[1]):
        y_s = y_train_pred[:, col_index]
        y_t = (y_train == (col_index))
        FPR, TPR, thresholds = roc_curve(y_t, y_s)
        plot.plot(FPR, TPR)
    plot.plot([0, 1], [0, 1], 'k--')
    plot.axis([0, 1, 0, 1])
    plot.xlabel("False Positive Rate")
    plot.ylabel("True Positive Rate")
    plot.show()

def main():
    print('Loading dataset')
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist['data'], mnist['target']
    # in string format. Making sure is a number
    y = y.astype(np.uint8)

    # some_digit = x[0]
    # some_digit_image = some_digit.reshape(28, 28)
    # plot.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
    # plot.axis("off")
    # plot.show()

    x_train, x_test, y_train, y_test = x[:60_000], x[60_000:], y[:60_000], y[60_000:]

    #train_binary(x_train, x_test, y_train, y_test)
    #train_multi(x_train, x_test, y_train, y_test)
    train_knearest(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    main()