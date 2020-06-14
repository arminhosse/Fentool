# Licensed under the MIT License.
""" Wrapper for different models"""

import abc
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


class Model(object):
    """ Class containing different models

    Parameters
    ----------
    model_type: String
                  Sets the type of "regression" or "classification" model.
                  Currently only regression is implemented, the values of the model
                  type should be 'linreg', 'lasso', 'lassocv', 'ridge', 'ridgecv',
                  'rfr', 'svr'.

    test_size: Float, Default=0.3
          Sets the size of the training set for evaluatingn the model.

    time_series: bool, Default=False
            Flag evaluating if the problem is a time series problem.
            Currently Fentool does not have support for time-series.

    """
    def __init__(self,
                 model_type='linreg',
                 test_size=0.3,
                 time_series=False,
                 **kwargs):

        self.model_type = model_type
        self.test_size = test_size
        self.time_series = time_series

        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self._model = []

        self.setup_model(**kwargs)

    def setup_model(self, **kwargs):
        """

        """

        if self.model_type is 'linreg':
            self._model = LinearRegression(**kwargs)
        elif self.model_type is 'lasso':
            self._model = Lasso(**kwargs)
        elif self.model_type is 'lassocv':
            self._model = LassoCV(**kwargs)
        elif self.model_type is 'ridge':
            self._model = Ridge(**kwargs)
        elif self.model_type is 'ridgecv':
            self._model = RidgeCV(**kwargs)
        elif self.model_type is 'rfr':
            self._model = RandomForestRegressor(**kwargs)
        elif self.model_type is 'svr':
            self._model = SVR(**kwargs)
        else:
            raise ValueError(
                "The model type {} is not supported".format(self.model_type))

    def train_test_split_(self):
        """ Wrapper for splitting the training and test method
        """
        # cross validation for time series not supported.
        if self.time_series is not True:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self.x, self.y, test_size=self.test_size)
        else:
            raise ValueError("Unrecodnized flag for input signal"
                             " (timehistory, etc)")

    def setup_feature_target(self, x, y):
        """ Set up the feature set for the model

        Parameters
        ----------
        x: pd.DataFrame
           Contains the feature set

        y: pd.DataFrame
           Contains the target variable
        """

        # update the feature and target set
        self.x = x.copy()
        self.y = y.copy()

    @abc.abstractmethod
    def _check_fit(self):
        """ Function to check if a model is already fitted
        """
        pass

    def fit(self, x, y):
        """ Wrapper for model fit method

        Parameters
        ----------
        x: pd.DataFrame
           Contains the feature set

        y: pd.DataFrame
           Contains the target variable

        """
        # check for previous fits
        self._check_fit()

        # setup the feature and target sets
        self.setup_feature_target(x, y)

        # split the to training and test sets
        self.train_test_split_()

        # fit the model
        self._model.fit(self.x, self.y.values.reshape(-1))

    def predict(self, x):
        """ Wrapper for model fit method

        Parameters
        ----------
        x: pd.DataFrame,
           Contains the feature set for prediction

        Returns
        -------

        """
        return self._model.predict(x)

    def score(self, set_type='test'):
        """ Wrapper for score method

        Parameters
        ----------
        set_type: String, Default='test
                 Determines on what set, training or test set the score should
                 be reported.

        Returns
        -------
        score: double,
               Returns the score value on either training or test sets

        """

        if set_type is 'train':
            score = self._model.score(self.x_train, self.y_train)
        elif set_type is 'test':
            score = self._model.score(self.x_test, self.y_test)
        return score

    def evaluate_model(self, x, y, n_splits=10, metric='r2', shuffle=True):
        """

        Parameters
        ----------
        x: pd.DataFrame
           Contains the feature set

        y: pd.DataFrame
           Contains the target variable

        n_splits: Integer, Default=10
          Determines the number of fold for cross-validation

        metric: String, Defaul='r2'
                The metric to evaluate the model using cross-validation.

        shuffle: bool, default=True
                Determines if the training data needs to be shuffled before
                training.

        Returns
        -------
        score: double,
              contains a dictionary with the different scores of the model from
              different cross validation folds.
        """

        # create the folds
        kfold = KFold(n_splits=n_splits, random_state=7, shuffle=shuffle)

        # evaluate scores on the different folds
        scores = cross_val_score(self._model, x, y.values.reshape(-1),
                                 cv=kfold, scoring=metric)

        return scores



