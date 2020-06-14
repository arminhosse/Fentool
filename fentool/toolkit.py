# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
import abc

import pandas as pd
import numpy as np

from fentool.pre_process.transformers import Minmax, Standard
from fentool.pre_process.encoders import Encoder
from fentool.models import Model

logger = logging.getLogger('fentool')
logger.setLevel(logging.INFO)


class Fentool(object):
    """ Fentool feature engineering tool

    Parameters
    ----------
    sup_learning_type: String
                     Determines the type of supervised learning, should be
                     regression or classification

    model_type: String
              Sets the type of "regression" or "classification" model.
              Currently only regression is implemented, the values of the model
              type should be 'linreg', 'lasso', 'lassocv', 'ridge', 'ridgecv',
              'rfr', 'svr'.

    encoder_type: String
                 Sets the type of encoding for the data sets. Currenlty
                 one-hot encoding and ordinal encoding is available. The values
                 should be 'one-hot' or 'Ordinal'

    input_treatment: String
                    Sets the type of treatment for the input(feature set)
                    Currenlty only normalization (minmax) and
                    standardization (mean and standard deviation) is implemented.
                    The values should be 'normalize' or 'standardize'.

    output_treatment: String
                     Set the type of treatment for the output(target).
                      Currenlty only normalization (minmax) and
                    standardization (mean and standard deviation) is implemented.
                    The values should be 'normalize' or 'standardize'.

    time_series: bool, Default=False
                Flag evaluating if the problem is a time series problem.
                Currently Fentool does not have support for time-series.

    fillna: String
            Method to remove or replace nans, nulls, etc. Should be "None",
            "drop", "mean", "zeros"

    test_size: Float, Default=0.3
              Sets the size of the training set for evaluatingn the model.

    null_tol_ratio: Float, Default=0.8
                    A value that determines  the maximum tolerance for
                    fentool to handle datasets with many null values. Must be
                    between 0 and 1.

    null_warn_ratio: Float, Default=0.3
                    A value that determines the lower threshold for
                    fentool to give warnings with datasets containig
                    many null values. Must be between 0 and 1.
    """

    def __init__(self,
                 sup_learning_type='regression',
                 model_type='linreg',
                 encoder_type=None,
                 input_treatment=None,
                 output_treatment=None,
                 time_series=False,
                 fillna='drop',
                 test_size=0.3,
                 null_tol_ratio=0.8,
                 null_warn_ratio=0.3,
                 **kwargs):
        self.sup_learning_type = sup_learning_type
        self.model_type = model_type
        self.encoder_type = encoder_type
        self.input_treatment = input_treatment
        self.output_treatment = output_treatment
        self.time_series = time_series
        self.fillna = fillna
        self.test_size = test_size
        self.null_tol_ratio = null_tol_ratio
        self.null_warn_ratio = null_warn_ratio

        self.target = []
        self.model = []
        self.df = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.x_trans = pd.DataFrame()
        self.y_trans = pd.DataFrame()
        self.validate_inputs()
        self.setup_model(**kwargs)

    def validate_inputs(self):
        """ This function validates the inputs given to prophet
        """
        # check for the type of supervised learning algorithm
        if self.sup_learning_type not in 'regression':
            raise ValueError('Currently supporting only "classification and '
                             '"regression models.')
        # check for the supported model types for assessing feature eng
        # effectiveness
        if self.model_type not in ('linreg', 'lasso', 'lassocv',
                                   'ridge', 'ridgecv', 'rfr', 'xgb'
                                   'svr'):
            raise ValueError('Not supported model type {} '
                             .format(self.model_type))

        # check for encoder types
        if self.encoder_type is not None:
            if self.encoder_type not in ('one-hot', 'ordinal'):
                raise ValueError('Not supported encoder type {} '
                                 .format(self.encoder_type))

        # validate  the input treatment
        if self.input_treatment is not None:
            if self.input_treatment not in ('normalize', 'standardize'):
                raise ValueError('Input treatment not supported!')

        if self.output_treatment is not None:
            if self.output_treatment not in ('normalize', 'standardize'):
                raise ValueError('Output treatment not supported!')

        if self.time_series is not False:
            raise ValueError('Time series support is not included yet')

        if self.fillna is not None:
            if self.fillna not in ('drop', 'zeros', 'mean'):
                raise ValueError('Not supported fill null method {} '
                                 .format(self.model_type))

        if (self.test_size > 1.0) or (self.test_size < .0):
            raise ValueError('The test size value should be between 0 and 1')

        if (self.null_tol_ratio > 1.0) or (self.null_tol_ratio < .0):
            raise ValueError('The null_tol_ratio should be between 0 and 1')

        if (self.null_warn_ratio > 1.0) or (self.null_warn_ratio < .0):
            raise ValueError('The null_warn_ratio should be between 0 and 1')

    def clean_nans(self):
        """ Function to clean up nan values
        """
        # drop nans
        if self.fillna is 'drop':
            self.df.dropna(inplace=True)
        # TODO One include methods to better handle nans for categorical vars
        # replace nans by zeros
        elif self.fillna is 'zeros':
            self.df.fillna(0)
        # replace nans by the mean value of each column
        elif self.fillna is 'mean':
            self.df.fillna(self.df.mean())

    def setup_dataframe(self, df, target):
        """ Function to setup the dataframe

        Parameters
        ----------
        df: pd.DataFrame
            The complete dataframe before.

        target: String
                Determines the column name of the target variable in the given
                dataframe.

        """

        # create a copy
        self.df = df.copy()

        # calculate the number rows with nulls
        num_null = sum(self.df.isnull().values.ravel())
        ratio_null = num_null/self.df.shape[0]

        # raise an error if the ratio of nulls is above a certain threshold
        if ratio_null > self.null_tol_ratio:
            raise ValueError("Dataframe has a high null to"
                             " value ratio: {}".format(round(ratio_null, 2)))
        # raise a warning regardless if there are some nulls
        elif ratio_null > self.null_warn_ratio:
            warnings.warn("Data set has a null "
                          "ratio of {} , treating with given method "
                          "'{}'".format(round(ratio_null, 2), self.fillna),
                          Warning)

        # run clean nans function
        if self.fillna is not None:
            self.clean_nans()

        # set up feature and target sets
        self.x = self.df.drop(columns=target)
        self.y = pd.DataFrame(self.df[target])

    def setup_model(self, **kwargs):
        """ Function to call the model class wrapper Model

        Parameters
        ----------
        kwargs: Used to pas exclusive model inputs specific to each model.

        """
        # build the model attr
        self.model = Model(model_type=self.model_type,
                           test_size=self.test_size,
                           time_series=self.time_series
                           , **kwargs)

    def feature_encoder(self):
        """ Function to encode the feature and target sets
        """
        # encode the feature set
        if self.encoder_type is not None:
            enc = Encoder(encoder_type=self.encoder_type)
            self.x = enc.fit_transform(self.x)

        # check to see if the target is a category variable
        if self.y.dtypes.name == 'category':
            self.y = pd.DataFrame(self.y.cat.codes)

    def feature_transform(self):
        """ Function to transform the feature sets with available transformer
        classes.
        """

        # check fot the give feature engineering setting
        if self.input_treatment == 'normalize':
            trans_input = Minmax()
            trans_input.fit(self.x)
            self.x_trans = trans_input.transform(self.x)

        elif self.input_treatment == 'standardize':
            trans_input = Standard()
            trans_input.fit(self.x)
            self.x_trans = trans_input.transform(self.x)
        else:
            self.x_trans = self.x.copy()

        # check for the required target transformation setting
        if self.output_treatment == 'normalize':
            trans_output = Minmax()
            trans_output.fit(self.y)
            self.y_trans = trans_output.transform(self.y)

        elif self.input_treatment == 'standardize':
            trans_output = Standard()
            trans_output.fit(self.y)
            self.y_trans = trans_output.transform(self.y)

        else:
            self.y_trans = self.y.copy()

    def prepar_fit_data(self, df, target):
        """ wrapper for preparing data sets for fit method

        Parameters
        ----------
        df: pd.DataFrame
            The complete dataframe before.

        target: String
                Determines the column name of the target variable in the given
                dataframe.
        """
        # clean the data and setup feature and target sets
        self.setup_dataframe(df, target)

        # encode the feature and target sets
        self.feature_encoder()

        # transform the feature and target sets.
        self.feature_transform()

    def fit(self, df, target):
        """ Wrapper for model fit method

        Parameters
        ----------
        df: pd.DataFrame
            The complete dataframe before.

        target: String
                Determines the column name of the target variable in the given
                dataframe.
        """

        # prepare the data for the fit method
        self.prepar_fit_data(df.copy(), target)

        # fit the model with the transformed data
        self.model.fit(self.x_trans, self.y_trans)

    def evaluate_model(self, df, target, n_splits=10,
                       metric='r2', shuffle=True):
        """ Function to evaluate the model using cross validation

        Parameters
        ----------
        df: pd.DataFrame
            The complete dataframe before.

        target: String
                Determines the column name of the target variable in the given
                dataframe.

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

        self.prepar_fit_data(df.copy(), target)

        score = self.model.evaluate_model(self.x_trans, self.y_trans,
                                          n_splits=n_splits,
                                          metric=metric, shuffle=shuffle)

        return score

    @staticmethod
    def model_compare(models, input_trans, output_trans, encoder_types,
                      df, target, n_splits=10,
                      metric='r2'):
        """ Wrapper to run a number of feature comninations

        Parameters
        ----------
        models: String, list/tuple
                Contains the models where the user needs to evaluate the best
                feature settings.

        input_trans: String, list/tuple
                    Contains the user settings for the required feature set
                    transformation

        output_trans: String, list/tuple
                    Contains the user settings for the required target set
                    transformation

        encoder_types: String, list/tuple
                    Contains the user settings for the required encoding method
                    of the non-numerical varaibles.

        df: pd.DataFrame
            The complete dataframe before.

        target: String
                Determines the column name of the target variable in the given
                dataframe.

        n_splits: Integer, Default=10
                  Determines the number of fold for cross-validation

        metric: String, Defaul='r2'
                The metric to evaluate the model using cross-validation.

        Returns
        -------
        Scores: pd.DataFrame
                A dataframe with different user cases and their corresponding
                scoring.

        """
        # loop over case names
        scores = pd.DataFrame()
        for modl in models:
            for enc in encoder_types:
                for in_trans in input_trans:
                    for out_trans in output_trans:
                        case = Fentool(encoder_type=enc,
                                       model_type=modl,
                                       input_treatment=in_trans,
                                       output_treatment=out_trans)

                        case_name = modl + '_' + enc + '_in' + \
                                    str(in_trans)[:4] + '_out' + str(out_trans)[:4]

                        scores[case_name] = case.evaluate_model(df=df, target=target,
                                                                n_splits=n_splits,
                                                                metric=metric)
            
        return scores
