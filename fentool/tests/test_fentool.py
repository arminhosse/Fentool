"""Unittest function for fentool functios """

from unittest import TestCase, skipUnless
import warnings
from fentool.toolkit import Fentool

import pytest
import os
import pandas as pd

from fentool.pre_process.encoders import Encoder
from fentool.pre_process.transformers import Minmax


RESOURCE_PATH = '%s/resources' % os.path.dirname(os.path.realpath(__file__))


class TestFentool(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')

    def test_validate_inputs(self):

        fent = Fentool(sup_learning_type='regression',
                       input_treatment='normalize',
                       output_treatment='normalize',
                       test_size= 0.4,
                       model_type='linreg')

        self.assertTrue(fent.sup_learning_type == 'regression',
                        msg="learning type mismatch")
        self.assertTrue(fent.input_treatment == 'normalize',
                        msg="input treatment type mismatch")
        self.assertTrue(fent.output_treatment == 'normalize',
                        msg="output treatment type mismatch")
        self.assertTrue(fent.test_size == 0.4,
                        msg="test size set mismatch")
        self.assertTrue(fent.model_type == 'linreg',
                        msg="model type mismatch")

    def test_clean_nans(self):

        fent = Fentool()

        fent.df = TestFentool.df.copy()
        fent.clean_nans()

        self.assertEqual(fent.df.shape,(2046,12),
                         msg='Unexpected dataframe shape after nan removals')

        fent.df = TestFentool.df
        fent.fillna = 'zeros'
        fent.clean_nans()

        self.assertEqual(fent.df.shape, TestFentool.df.shape,
                         msg='Unexpected dataframe shape after nan replacement')

        fent.df = TestFentool.df
        fent.fillna = 'mean'
        fent.clean_nans()

        self.assertEqual(fent.df.shape, TestFentool.df.shape,
                         msg='Unexpected dataframe shape after nan replacement')

    def test_setup_dataframe(self):

        fent = Fentool()

        # test if the setup data frame issues a warning for nulls
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fent.setup_dataframe(TestFentool.df, target='median_house_value')
            self.assertEqual(len(w), 1)

        # test the shape of feature set
        self.assertEqual(fent.x.shape[1], TestFentool.df.shape[1]-1,
                         msg='Unexpected feature set shape after df setup')

        self.assertEqual(fent.y.shape[0],
                         2046,
                         msg='Unexpected target set shape after df setup')

        self.assertEqual(fent.x.shape[0],
                         fent.y.shape[0],
                         msg='Mismatchin number of rows between feature '
                             'and target set')

    def test_setup_model(self):

        fent = Fentool()

        self.assertTrue(hasattr(fent, 'model'),
                        msg='Missing model attribute')

    def test_feature_transform(self):

        pass
