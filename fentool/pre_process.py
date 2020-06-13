"""A wrapper for different pre processing method classes"""

import pandas as pd
import numpy as np


class Encoder(object):
    """ Encoder class for numerical and categorical data sets.

    """
    def __init__(self,
                 encoder_type='one-hot'):
        self.encoder_type = encoder_type
        self.cat_cols = []

    def verify_input(self):
        """ verify the input settings for the encoder.

        Returns
        -------

        """
        if self.encoder_type not in ('one-hot','ordinal'):
            raise ValueError("The encoder must be one-hot or ordinal")

    @staticmethod
    def auto_detect_categorical(df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        categorical_cols = df.select_dtypes(exclude=['int', 'float',
                                                     'double']).columns
        return categorical_cols

    def fit_transform(self, df):
        """

        Parameters
        ----------
        df

        Returns
        -------

        """

        df_enc = df.copy()

        if self.encoder_type == 'ordinal':
            self.cat_cols = self.auto_detect_categorical(df_enc)
            for col in self.cat_cols:
                df_enc[col] = df_enc[col].astype('category')
                df_enc[col] = df_enc[col].cat.codes

        elif self.encoder_type == 'one-hot':
            df_enc = pd.get_dummies(df_enc)

        else:
            raise ValueError("Unrecognized encoder type {}".
                             format(self.encoder_type))

        return df_enc


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

    def __init__(self):
        pass

    def fit(self, df):
        pass

    def transform(self, df):
        pass

    def inverse_transform(self, df):
        pass