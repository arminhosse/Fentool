# Licensed under the MIT License.
""" Implemented encoder methods for non-numerical variables"""

import pandas as pd


class Encoder():
    """ Encoder class for numerical and categorical data sets.

    Parameters
    ----------
    encoder_type: String
                 Sets the type of encoding for the data sets. Currently
                 one-hot encoding and ordinal encoding is available. The values
                 should be 'one-hot' or 'Ordinal'

    """
    def __init__(self,
                 encoder_type='one-hot'):
        self.encoder_type = encoder_type
        self.cat_cols = []

    @staticmethod
    def auto_detect_categorical(df):
        """ detect non-numerical columns

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the feature and target set

        Returns
        -------
        categorical_cols: String
                        Contains the name of the columns that are not numerical

        """

        categorical_cols = df.select_dtypes(exclude=['int', 'float', 'float64',
                                                     'uint8', 'double']).columns
        return categorical_cols

    def fit_transform(self, df):
        """ Wrapper for fit transform function

        Parameters
        ----------
        df: pd.DataFrame
            The dataframe containing the feature and target set

        """

        df_enc = df.copy()

        # perform ordinal transformation
        if self.encoder_type == 'ordinal':
            self.cat_cols = self.auto_detect_categorical(df_enc)
            for col in self.cat_cols:
                df_enc[col] = df_enc[col].astype('category')
                df_enc[col] = df_enc[col].cat.codes

        # one-hot encoding
        elif self.encoder_type == 'one-hot':
            df_enc = pd.get_dummies(df_enc)
            for col in df_enc.columns:
                df_enc[col] = pd.to_numeric(df_enc[col])

        else:
            raise ValueError("Unrecognized encoder type {}".
                             format(self.encoder_type))

        return df_enc
