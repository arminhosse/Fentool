""" Method for normlizing the """

import abc
import pandas as pd


class Minmax(object):
    """ Minmax class to transform and fit each column of a data frame
    to its corresponding minimum and maximum.

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
        """

        Returns
        -------

        """

        if self.input_range[0] > self.input_range[1]:
            raise ValueError('The first number should be lower than the second')

    @abc.abstractmethod
    def clean_for_fit(self):
        pass

    def fit(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError('The input to the fit must be a dataframe')
        elif self.df.empty:
            self.df = df.copy()
        else:
            raise ValueError('The model is already fit!')

        # determine the max an min for each column
        self.data_min = self.df.min()
        self.data_range = self.df.max()-self.df.min()

        self.transform_ratio = (self.input_range[1] -
                                self.input_range[0])/self.data_range

        self.new_data_min = (self.input_range[0] -
                             self.data_min * self.transform_ratio)

    def transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        if not isinstance(df, pd.DataFrame):
            raise ValueError('The input to the transform must be a dataframe')

        df_scaled = df.copy()

        df_scaled = (df_scaled * self.transform_ratio) + self.new_data_min

        return df_scaled

    def inverse_transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

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

        self.validate_input()

    @abc.abstractmethod
    def validate_input(self):
        """

        Returns
        -------

        """
        pass

    def fit(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        self.df = df.copy()

        self.mean = df.mean()
        self.std = df.std()

    def transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        df_scaled = df.copy()

        df_scaled = (df_scaled - self.mean)/self.std

        return df_scaled

    def inverse_transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        df_inv = df.copy()

        df_inv = (df_inv * self.std) + self.mean

        return df_inv
