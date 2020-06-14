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
from xgboost import XGBRegressor


class Model(object):
    """ Class for defining the model
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

    def validate_inputs(self):
        """

        Returns
        -------

        """

        if self.time_series is not False:
            raise ValueError('Time series support is not included yet')

        if (self.test_size > 1.0) or (self.test_size < .0):
            raise ValueError('The test size value should be between 0 and 1')

    def setup_model(self, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

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
        elif self.model_type is 'xgb':
            self._model = XGBRegressor(**kwargs)
        elif self.model_type is 'svr':
            self._model = SVR(**kwargs)
        else:
            raise ValueError(
                "The model type {} is not supported".format(self.model_type))

    def train_test_split_(self):
        """

        Returns
        -------

        """

        if self.time_series is not True:
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(self.x, self.y, test_size=self.test_size)
        else:
            raise ValueError("Unrecodnized flag for input signal"
                             " (timehistory, etc)")
        # reshape for use in linreg models
        #self.y_train = self.y_train.values.reshape(-1)
        #self.y_test = self.y_test.values.reshape(-1)

    def setup_feature_target(self, x, y):

        # update the feature and target set
        self.x = x.copy()
        self.y = y.copy()

    @abc.abstractmethod
    def _check_fit(self):
        pass

    def fit(self, x, y):
        """

        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        # check for previous fits
        self._check_fit()

        # setup the feature and target sets
        self.setup_feature_target(x, y)

        # split the to training and test sets
        self.train_test_split_()

        # fit the model
        self._model.fit(self.x, self.y)

    def predict(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        return self._model.predict(x)

    def score(self, set_type='test'):

        if set_type is 'train':
            score = self._model.score(self.x_train, self.y_train)
        elif set_type is 'test':
            score = self._model.score(self.x_test, self.y_test)
        return score

    def evaluate_model(self, x, y, n_splits=10, metric='r2', shuffle=True):

        # create the folds
        kfold = KFold(n_splits=n_splits, random_state=7, shuffle=shuffle)

        # evalute scores on the different folds
        scores = cross_val_score(self._model, x, y,
                                 cv=kfold, scoring=metric)

        return scores



