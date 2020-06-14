# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict, defaultdict
from copy import deepcopy
import logging
import warnings

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
    sup_learnin_type: String
                     Determines the type of supervised learning, should be
                     regression or classification

    model_type: String
              Sets the type of "regression" or "classification" model

    input_treatment: String
                    Sets the type of treatment for the input(feature set)
                    such as normalization or standardization.

    output_treatment: String
                     Set the type of treatment for the output(target) such as
                     normalization or standardization.

    time_series: Bool
                Flag evaluating if the problem is a time series problem

    fillna: String
            Method to remove or replace nans, nulls, etc. Should be "None",
            "drop", "mean", "zeros"

    test_size: Float
              Sets the size of the training set for evaluation the model and
              selected treatment.

    null_tol_ratio: Float
                    A value that determines on the maximum tolerance for
                    fentool to handle datasets with many null values. Must be
                    between 0 and 1.

    target: String
           The column name of the target variable

    df: pd.DataFrame
       Contains the orginal datafram given by the user

    df_trans pd.DataFrame:
            The encoded and transformed dataframe
    """

    def __init__(self,
                 sup_learning_type='regression',
                 model_type='linreg',
                 encoder_type=None,
                 input_treatment=None,
                 output_treatment=None,
                 time_series=False,
                 fillna = 'drop',
                 test_size=0.3,
                 null_tol_ratio=0.8,
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

        self.target = []
        self.model = []
        self.df = pd.DataFrame()
        self.df_trans = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.validate_inputs()
        self.setup_model(**kwargs)

    def validate_inputs(self):
        """ This function validates the inputs given to prophet

        Returns
        -------

        """
        # check for the type of supervised learning algorithm
        if self.sup_learning_type not in 'regression':
            raise ValueError('Currently supporting only "classification and '
                             '"regression models.')
        # check for the supported model types for assessing feature eng
        # effectiveness
        if self.model_type not in ('linreg', 'lasso'):
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

        if self.output_treatment is not None:
            if self.fillna not in ('drop', 'zeros', 'mean'):
                raise ValueError('Not supported fill null method {} '
                                 .format(self.model_type))

        if (self.test_size > 1.0) or (self.test_size < .0):
            raise ValueError('The test size value should be between 0 and 1')

    def clean_nans(self):
        """

        Returns
        -------

        """

        if self.fillna is 'drop':
            self.df.dropna(inplace=True)
        elif self.fillna is 'zeros':
            self.df.fillna(0)
        elif self.fillna is 'mean':
            self.df.fillna(self.df.mean())

    def setup_dataframe(self, df, target):
        """

        Parameters
        ----------
        df
        target

        Returns
        -------

        """

        # create a copy
        self.df = df.copy()

        # calculate the number rows with nulls
        num_null=sum(self.df.isnull().values.ravel())
        ratio_null= num_null/self.df.shape[0]

        # raise an error if the ratio of nulls is above a certain threshold
        if ratio_null > self.null_tol_ratio:
            raise ValueError("Dataframe has a high null to"
                             " value ratio: {}".format(round(ratio_null, 2)))
        # raise a warning regardless if there are some nulls
        elif ratio_null > 0:
            warnings.warn("Data set has a null "
                  "ratio of {} , treating with given method "
                  "'{}'".format(round(ratio_null, 2), self.fillna),
                          Warning)

        # run clean nans function
        if self.fillna is not None:
            self.clean_nans()

        # set up feature and target sets
        self.x = self.df.drop(columns=target)
        self.y = self.df[target]

    def setup_model(self,**kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

        """

        self.model = Model(model_type=self.model_type, **kwargs)

    def feature_transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        if self.encoder_type is not None:
            enc = Encoder(encoder_type=self.encoder_type)

    def fit(self, df, target):
        """

        Parameters
        ----------
        df
        target

        Returns
        -------

        """

        self.setup_datafram(df, target)

    @abc.abstractmethod
    def evaluate_model(self):
        """

        Returns
        -------

        """
        pass

    @abc.abstractmethod
    def model_compare(self):
        """

        Returns
        -------

        """
        pass