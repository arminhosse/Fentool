import pandas as pd



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
            for col in df_enc.columns:
                df_enc[col] = pd.to_numeric(df_enc[col])

        else:
            raise ValueError("Unrecognized encoder type {}".
                             format(self.encoder_type))

        return df_enc
