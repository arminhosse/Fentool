"""Unittest wrapper for different feature engineering methods"""

from unittest import TestCase, skipUnless
import pytest
import os
import pandas as pd

from fentool.pre_process.encoders import Encoder
from fentool.pre_process.transformers import Minmax, Standard


RESOURCE_PATH = '%s/resources' % os.path.dirname(os.path.realpath(__file__))


class TestMinMax(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')
        enc = Encoder()
        cls.df_enc = enc.fit_transform(cls.df)

    def test_validate_inputs(self):

        prep = Minmax(input_range=(0, 1))

        self.assertTrue(prep.input_range == (0, 1),
                        msg="desired range not assigned properly")

    def test_fit(self):
        # build the minmax function
        prep = Minmax()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestMinMax.df_enc.copy()))

        # check to see if the fit function has updated the
        self.assertTrue(prep.df.empty!=True,  msg="minmax fit method failed")

    def test_transform(self):
        # build the minmax function
        prep = Minmax()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestMinMax.df_enc.copy()))

        # use the minmax transform with the fit parameters
        df_norm = prep.transform(TestMinMax.df_enc.copy())

        # assert to see if the min and max match the given default range
        self.assertTrue((df_norm.max() == prep.input_range[1]).all,
                        msg="maximum normalized value deviates from default "
                            "range")

        self.assertTrue((df_norm.min() == prep.input_range[0]).all,
                        msg="minimum normalized value deviates from default "
                            "range")

    def test_inverse_transform(self):

        # build the minmax function
        prep = Minmax()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestMinMax.df_enc.copy()))

        # use the minmax transform with the fit parameters
        df_norm = prep.transform(TestMinMax.df_enc.copy())

        # inverse the transformation
        df_org = prep.inverse_transform(df_norm)

        # check to see if the inverse gives the same result
        self.assertAlmostEqual(df_org.values.all(),
                               TestMinMax.df_enc.values.all(),
                               msg="inverse minmax gives different results")


class TestStandard(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')
        cls.df = cls.df.dropna()

        enc = Encoder()
        cls.df_enc = enc.fit_transform(cls.df)

    def test_fit(self):
        # build the minmax function
        prep = Standard()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestStandard.df_enc.copy()))

        # check to see if the fit function has updated the
        self.assertTrue(prep.df.empty!=True,  msg="Standard fit method failed")

    def test_transform(self):
        # build the minmax function
        prep = Standard()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestStandard.df_enc.copy()))

        # use the minmax transform with the fit parameters
        df_scaled = prep.transform(TestStandard.df_enc.copy())

        # assert to see if the min and max match the given default range
        self.assertAlmostEqual(max(round(df_scaled.mean())), 0.0, 1,
                               msg="Standardized data mean deviates from 0")


    def test_inverse_transform(self):

        # build the minmax function
        prep = Standard()

        # fit to the sample dataframe
        prep.fit(pd.get_dummies(TestStandard.df_enc.copy()))

        # use the minmax transform with the fit parameters
        df_norm = prep.transform(TestStandard.df_enc.copy())

        # inverse the transformation
        df_org = prep.inverse_transform(df_norm)

        # check to see if the inverse gives the same result
        self.assertAlmostEqual(df_org.values.all(),
                               TestStandard.df_enc.values.all(),
                               msg="inverse standard gives different results")