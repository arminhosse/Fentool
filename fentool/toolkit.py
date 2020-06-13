# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict, defaultdict
from copy import deepcopy
import logging

import pandas as pd
import numpy as np

from fentool.pre_process import Minmax, Standard

logger = logging.getLogger('fentool')
logger.setLevel(logging.INFO)


class Fentool(object):
    """ Fentool feature engineering tool
    Parameters
    ----------
    sup_learnin_type: String
                     Determines the type of supervised learning, regression or
                      classification
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
    test_size: Float
              Sets the size of the training set for evaluation the model and
              selected treatment.
    """

    def __init__(self,
                 sup_learning_type='regression',
                 model_type='linreg',
                 input_treatment=None,
                 output_treatment=None,
                 time_series = False,
                 test_size=0.3,
                 **kwargs):
        self.model_type = model_type
        self.sup_learning_type = sup_learning_type
        self.input_treatment = input_treatment
        self.output_treatment = output_treatment
        self.time_series = time_series
        self.test_size = test_size
        self.validate_inputs()

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
            raise ValueError('Not supprted model type {} '
                             .format(self.model_type))
        # validate  the input treatment
        if self.input_treatment is not None:
            if self.input_treatment not in ('normalize', 'standardize'):
                raise ValueError('Input treatment not supported!')

        if self.output_treatment is not None:
            if self.output_treatment not in ('normalize', 'standardize'):
                raise ValueError('Output treatment not supported!')

        if self.time_series is not False:
            raise ValueError('Time series support is not supported yet')

        if (self.test_size > 1.0) or (self.test_size < .0):
            raise ValueError('The test size value should be between 0 and 1')

    def feature_transform(self):
        pass