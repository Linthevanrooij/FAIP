from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def compute_feature_curve(X,y,model,repeats,train_size=0.5):    
    
    performance = np.empty((repeats,X.shape[1]))
    test_size = 1 - train_size

    for i in range(0,repeats):
        for features in range(1, X.shape[1]+1):
            X_selected = X[:,:features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=i)
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            performance[i,features-1] = mean_squared_error(y_test, y_predict)
            
    return np.mean(performance,axis=0)


def plot_learning_curve(estimator, model_name, X, y, c, only_test=False, show_std=True, metric='MSE',
                        cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plots a learning curve for the estimator to the current active axis of pyplot.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    model_name : str
        Name of the model.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    c : color representing this model.

    only_test : if True, only plots the test performance and hides the training curve.

    show_std : if True, indicates the width of the standard deviation across the folds.

    metric : 'MSE' for mean squared error, 'accuracy' for accuracy.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    ax = plt.gca()
    ax.set_xlabel("Training set size")

    modifier = 1
    if metric == 'MSE':
        scoring = 'neg_mean_squared_error'
        modifier = -1
        modifier_bias = 0
        ax.set_ylabel("MSE")
        scoring_legend = 'MSE'
    if metric == 'accuracy':
        scoring = 'accuracy'
        modifier = -1
        modifier_bias = 1
        scoring_legend = 'error'
        ax.set_ylabel("Error rate")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=scoring)

    train_scores_mean = modifier * np.mean(train_scores, axis=1) + modifier_bias
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = modifier * np.mean(test_scores, axis=1) + modifier_bias
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    if show_std:
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color=c)
        if not only_test:
            ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color=c)

    if not only_test:
        ax.plot(train_sizes, train_scores_mean, ':o', color=c,
                label="%s Train %s" % (model_name, scoring_legend))
    ax.plot(train_sizes, test_scores_mean, '-o', color=c,
            label="%s Test %s" % (model_name, scoring_legend))
    ax.legend(loc="best")

    return plt
