""" Unittess for the Model class wrapper for regression & calssificatioin"""

from unittest import TestCase
import pytest
import os
import pandas as pd

from fentool.pre_process.encoders import Encoder
from fentool.pre_process.transformers import Minmax
from fentool.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, LassoCV

RESOURCE_PATH = '%s/resources' % os.path.dirname(os.path.realpath(__file__))


class TestModel(TestCase):

    @classmethod
    def setUpClass(cls):
        # read sample data
        cls.df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')
        # replace nulls
        cls.df.dropna(inplace=True)
        # encode the date set
        enc = Encoder()
        cls.df_enc = enc.fit_transform(cls.df)

        # prepare the data
        minmax_scaler = Minmax()
        minmax_scaler.fit(cls.df_enc)
        cls.df_scaled = minmax_scaler.transform(cls.df_enc)

        # define feature and target variables
        cls.x = cls.df_scaled.drop(columns='median_house_value')
        cls.y = cls.df_scaled['median_house_value']

    def test_setup_model(self):

        my_model = Model(model_type='linreg')
        mock_model = LinearRegression()

        self.assertTrue(type(my_model._model) == type(mock_model),
                        msg='Mismatch in initialized '
                            'model {}'.format(mock_model))

        my_model = Model(model_type='lasso')
        mock_model = Lasso()

        self.assertTrue(type(my_model._model) == type(mock_model),
                        msg='Mismatch in initialized '
                            'model {}'.format(mock_model))


    def test_train_test_split(self):

        # create a sample model
        my_model = Model(model_type='linreg')

        # assign x and y values
        my_model.x = TestModel.x
        my_model.y = TestModel.y

        # run the split train test
        my_model.train_test_split_()

        # check for the sample data split on default values
        self.assertEqual(my_model.x_train.shape, (1432, 14),
                         msg='Mismatch in training test proportions')

        self.assertEqual(my_model.x_test.shape, (614, 14),
                         msg='Mismatch in training test proportions')

        self.assertEqual(my_model.x_train.shape[0], my_model.y_train.shape[0],
                         msg='Mismatch in feature and target training sizes')

    def test_fit(self):

        # create a sample model
        my_model = Model(model_type='linreg')

        # only call the fit to update all the variables
        my_model.fit(TestModel.x, TestModel.y)

        self.assertAlmostEqual(my_model._model.coef_.max(), 1.45, 1,
                               msg='Maximum coefficient mismatch'
                                   ' for the fitted model')

        self.assertEqual(my_model._model.coef_.shape[0], 14,
                               msg='Wrong number of model coefficients!')

    def test_setup_feature_target(self):

        # create a sample model
        my_model = Model(model_type='linreg')

        my_model.setup_feature_target(TestModel.x, TestModel.y)

        self.assertEqual(my_model.x.shape, TestModel.x.shape,
                         msg='Shape mismatch!')

        self.assertEqual(my_model.y.shape, TestModel.y.shape,
                         msg='Shape mismatch!')

    def test_evaluate_model(self):

        # create a sample model
        my_model = Model(model_type='linreg')

        my_model.setup_feature_target(TestModel.x, TestModel.y)

        score = my_model.evaluate_model(my_model.x, my_model.y)

        self.assertAlmostEqual(score.mean(), 0.65, 1, msg='Unexpected score')