"""Unittest function for fentool functios """

from unittest import TestCase, skipUnless
import warnings
from fentool.toolkit import Fentool

import pytest
import os
import pandas as pd
import seaborn as sns

from fentool.pre_process.encoders import Encoder
from fentool.pre_process.transformers import Minmax


RESOURCE_PATH = '%s/resources' % os.path.dirname(os.path.realpath(__file__))


class TestFentool(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')
        cls.df_mod = cls.df.dropna()

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

        self.assertEqual(fent.df.shape, (2046,12),
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
                         msg='Mismatch in number of rows between feature '
                             'and target set')

    def test_setup_model(self):

        fent = Fentool()

        self.assertTrue(hasattr(fent, 'model'),
                        msg='Missing model attribute')

    def test_feature_encoder(self):

        fent = Fentool(encoder_type='one-hot')

        # test if the setup data frame issues a warning for nulls
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fent.setup_dataframe(TestFentool.df, target='median_house_value')

        fent.feature_encoder()

        # TODO Needs to verify if other numerical formats should be included.
        num_cols_x = fent.x.select_dtypes(exclude=['int', 'float', 'float64',
                                                   'uint8', 'double']).columns

        self.assertEqual(num_cols_x.shape[0], 0,
                         msg="Detected un-encoded columns")

        self.assertTrue(fent.y.dtypes[0] in ['int', 'float', 'float64', 'uint8',
                                             'double'],
                        msg="Detected categorical values for target")

    def test_feature_transform(self):

        fent = Fentool(encoder_type='one-hot', input_treatment='normalize',
                       output_treatment='normalize')

        # test if the setup data frame issues a warning for nulls
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fent.setup_dataframe(TestFentool.df, target='median_house_value')

        fent.feature_encoder()

        self.assertTrue((fent.x_trans.max()==1).all(),
                        msg="Detected max values outside of"
                            " normalization range")

        self.assertTrue((fent.x_trans.min()==0).all(),
                        msg="Detected min values outside of "
                            "normalization range")

    def test_fit(self):

        # test the fit with minmax normalization
        fent = Fentool(encoder_type='one-hot', input_treatment='normalize',
                       output_treatment='normalize')

        fent.fit(TestFentool.df_mod, target='median_house_value')
        self.assertAlmostEqual(fent.model._model.coef_.max(), 1.45, 2,
                               msg="Unexpected fit coefficient with "
                                   "sample data")

        self.assertAlmostEqual(fent.model.score(), 0.66, 1,
                               msg="Unexpected model score on test date")

        # test the fit with standard scaler
        fent = Fentool(encoder_type='one-hot', input_treatment='standardize',
                       output_treatment='standardize')

        fent.fit(TestFentool.df_mod, target='median_house_value')
        self.assertAlmostEqual(fent.model._model.coef_.max(), 0.68, 1,
                               msg="Unexpected fit coefficient with "
                                   "sample data")

        self.assertAlmostEqual(fent.model.score(), 0.68, 1,
                               msg="Unexpected model score on test date")

    def test_evaluate_model(self):

        fent = Fentool(encoder_type='one-hot', input_treatment='normalize',
                       output_treatment='normalize')

        score = fent.evaluate_model(TestFentool.df
                                    , target='median_house_value')

        self.assertAlmostEqual(score.mean(), 0.65, 1,
                               msg="Unexpected model score on cross validation")

    def test_model_compare(self):

        fent = Fentool()

        models = ['linreg']
        encoder_types =['one-hot']
        input_trans = [None, 'normalize', 'standardize']
        output_trans = ['standardize']

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores = fent.model_compare(models=models,
                                        encoder_types=encoder_types,
                                        input_trans=input_trans,
                                        output_trans = output_trans,
                                        df=TestFentool.df,
                                        target='median_house_value')


        self.assertEqual(scores.shape, (10,3), msg="Missing score number "
                                                   "for one of the cases")
