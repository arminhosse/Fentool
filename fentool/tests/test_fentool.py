"""Unittest function for fentool functios """

from unittest import TestCase, skipUnless
from fentool.toolkit import Fentool


class TestFentool(TestCase):

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

