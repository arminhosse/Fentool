from unittest import TestCase, skipUnless
import pytest
import os
import pandas as pd

from fentool.pre_process.encoders import Encoder

RESOURCE_PATH = '%s/resources' % os.path.dirname(os.path.realpath(__file__))


class TestEncoder(TestCase):
    """ Unittest for the encoder method """
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')

    def test_auto_detect_categorical(self):

        enc = Encoder()

        col = enc.auto_detect_categorical(TestEncoder.df)

        self.assertTrue(col == 'ocean_proximity',
                    msg="auto category detection failed")

    def test_fit_transform(self):

        # test the ordinal
        enc = Encoder('ordinal')
        df_enc = enc.fit_transform(TestEncoder.df.copy())

        self.assertTrue((TestEncoder.df.columns == df_enc.columns).all()
                        , msg="Mismatch in df columns after ordinal encoder")
        self.assertTrue(enc.cat_cols == 'ocean_proximity',
                        msg="Unexpected categorical column")

        self.assertTrue(df_enc[enc.cat_cols].min().values == 0,
                        msg="Unexpected categorical code")
        self.assertTrue(df_enc[enc.cat_cols].max().values == 3,
                        msg="Unexpected categorical code")

        # test the one-hot encoder
        enc = Encoder('one-hot')
        df_enc = enc.fit_transform(TestEncoder.df)
        self.assertTrue(df_enc.columns.shape == (15,),
                        msg="Mismatch in df columns after ordinal encoder")

