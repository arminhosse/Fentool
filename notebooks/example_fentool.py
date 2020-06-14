""" Example for running fentool"""
import os

import seaborn as sns
import pandas as pd

from fentool.toolkit import Fentool

# define file location
RESOURCE_PATH = '%s/../fentool/tests/resources' % os.path.dirname(os.path.realpath(__file__))

# read dataframe
df = pd.read_csv(RESOURCE_PATH + '/sample_data.csv')

# create fentool object
fent = Fentool()

models = ['linreg', 'lassocv', 'svr']
encoder_types = ['one-hot']
input_trans = [None, 'normalize', 'standardize']
output_trans = ['standardize']

# run the loop on the combinations
scores = fent.model_compare(models=models,
                            encoder_types=encoder_types,
                            input_trans=input_trans,
                            output_trans=output_trans,
                            df=df,
                            target='median_house_value')

# plot the results
plot = sns.boxplot(data=scores)
plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
