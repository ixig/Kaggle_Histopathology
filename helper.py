import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask import delayed, compute
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


def load_npz(file):
    npdict = np.load(file)
    X_train = npdict['X_train']
    y_train = npdict['y_train']
    X_test = npdict['X_test']
    y_test = npdict['y_test']
    npdict.close()
    print(
        (X_train.shape, X_train.dtype),
        (y_train.shape, y_train.dtype))
    print(
        (X_test.shape, X_test.dtype),
        (y_test.shape, y_test.dtype))
    return (X_train, y_train), (X_test, y_test)


def load2x_npz(file):
    npdict = np.load(file)
    X_train = npdict['X_train']
    X2_train = npdict['X2_train']
    y_train = npdict['y_train']
    X_test = npdict['X_test']
    X2_test = npdict['X2_test']
    y_test = npdict['y_test']
    npdict.close()
    print(
        (X_train.shape, X_train.dtype),
        (X2_train.shape, X2_train.dtype),
        (y_train.shape, y_train.dtype))
    print(
        (X_test.shape, X_test.dtype),
        (X2_test.shape, X2_test.dtype),
        (y_test.shape, y_test.dtype))
    return (X_train, X2_train, y_train), (X_test, X2_test, y_test)


def GBT(X_train, X_test, y_train, y_test,
        n_ests = [50, 100, 200, 400, 600, 800]):
    clfs = [(GradientBoostingClassifier(n_estimators=n_est), n_est)
            for n_est in (n_ests[:3] if len(X_train) <= 4000 else n_ests[3:])]
    clfs_d = [delayed(clf[0].fit)(X_train, y_train) for clf in clfs]
    clfs_c = compute(clfs_d)[0]
    print('n_est: Train, Test')
    for i, clf in enumerate(clfs_c):
        print(f'{clfs[i][1]:<5}', end=': ')
        print(f'{clf.score(X_train, y_train)*100:.1f}', end=',  ')
        print(f'{clf.score(X_test, y_test)*100:.1f}')


def CM(y_true, y_hat):
    cm = metrics.confusion_matrix(y_true, y_hat)
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap='gray', cbar=False, square = True);
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    accuracy = np.trace(cm) / np.sum(cm)
    plt.title(f'Accuracy: {accuracy*100:.1f}%')
    plt.tick_params(labelsize= 12)


def return_feature_rank_from_RF(X_train,y_train):
# Build a forest and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%3d. feature %3d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    return (indices,importances)

