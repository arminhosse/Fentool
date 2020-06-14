# Licensed under the MIT License.

""" Implemented methods for feature engineering """

import abc
import pandas as pd


class Minmax(object):
    """ Minmax class to transform and fit each column of a data frame
    to its corresponding minimum and maximum.

    Parameters
    ----------
    input_range: Float, Default: (0,1)
                 The target range with with the minmax class
                 transforms the columns.
    """

    def __init__(self,
                 input_range=(0, 1)
                 ):
        self.input_range = input_range
        self.data_min = []
        self.data_range = []
        self.transform_ratio = []
        self.new_data_min = []
        self.df = pd.DataFrame()

        self.validate_input()

    def validate_input(self):
        """ validate inputs to the class
        """

        if self.input_range[0] > self.input_range[1]:
            raise ValueError('The first number should be lower than the second')

    def fit(self, df):
        """ Function to fit to the min max and range of data

        Parameters
        ----------
        df: pd.DataFrame
           Input dataframe for the fit method

        """
        # check if the input is a dataframe
        if not isinstance(df, pd.DataFrame):
            raise ValueError('The input to the fit must be a dataframe')
        elif self.df.empty:
            self.df = df.copy()
        else:
            raise ValueError('The model is already fit!')

        # determine the max an min for each column
        self.data_min = self.df.min()
        self.data_range = self.df.max()-self.df.min()

        # define the transform scale ratio
        self.transform_ratio = (self.input_range[1] -
                                self.input_range[0])/self.data_range

        # calculate the new mimimum range for the data
        self.new_data_min = (self.input_range[0] -
                             self.data_min * self.transform_ratio)

    def transform(self, df):
        """

        Parameters
        ----------
        df: pd.DataFrame
           Input dataframe for the transformation using the previously used fit
           method

        Returns
        -------
        df_scaled: pd.DataFrame
                   Normalized dataframe with minmax method

        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError('The input to the transform must be a dataframe')

        if len(self.data_range) == 0:
            raise ValueError('Empty fit method variables, '
                             'Use the fit method first')

        df_scaled = df.copy()

        df_scaled = (df_scaled * self.transform_ratio) + self.new_data_min

        return df_scaled

    def inverse_transform(self, df):
        """

        Parameters
        ----------
        df: pd.DataFrame
           Input dataframe for the inverse transform using the already fitted
           params

        Returns
        -------
        df_inv: pd.DataFrame
                A dataframe using inverse minmax transform
        """

        df_inv = df.copy()

        df_inv = (df_inv - self.new_data_min) / self.transform_ratio

        return df_inv


class Standard(object):
    """ Class method for standardization with mean and standard deviation
    """

    def __init__(self,
                 ):
        self.mean = []
        self.std = []
        self.df = pd.DataFrame()

    def fit(self, df):
        """ Method to compute the mean and std for the given dataframe

        df: pd.DataFrame
           Input dataframe for the fit method
        """

        self.df = df.copy()

        self.mean = df.mean()
        self.std = df.std()

    def transform(self, df):
        """

        df: pd.DataFrame
           Input dataframe for the transformation using the previously used fit
           method

        df_scaled: pd.DataFrame
                   Normalized dataframe with standardization method

        """

        df_scaled = df.copy()

        df_scaled = (df_scaled - self.mean)/self.std

        return df_scaled

    def inverse_transform(self, df):
        """

        Parameters
        ----------
        df: pd.DataFrame
           Input dataframe for the inverse transform using the already fitted
           params

        Returns
        -------
        df_inv: pd.DataFrame
                A dataframe using inverse standardization transform
        """

        df_inv = df.copy()

        df_inv = (df_inv * self.std) + self.mean

        return df_inv
