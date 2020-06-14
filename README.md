# Fentool
The Fentool package provides different implementations 
necessary for preparing the features sets for machine learning 
models. It aims to provide the user with one library which 
can automate a lot of different steps in feature preparation as well as
assessing optimal steps to improve performance of an ML model.

<details>
<summary><strong><em>Table of Contents</em></strong></summary>
 
- [Modules](#Modules)
- [Usage](#Usage-started)
 
</details>

# Modules
The modules are a essentially a constellation of different classes that are designed
to enable the user to encode, transform and assess such transformation of different models.
This is done by: 
1) __Defining models, treatment parameters encoding methods in [toolkit.py](,/fentool/toolkit,py)__<br/> 
In this routine the user can determine what type of encoding would suit their feature set. They can also 
 the take advantage of the implemented minmax normalization as well as standardization using the average and
standard deviations. 

2) __General Model Class in [models.py](,/fentool/models,py)__<br/>
This is a general class wrapper around already implemented models. It adds a number of different methods
such that user is enable run different models using the same fit and predict method. It also provided scoring
based on the cross-validation methods. Currently time-series and classification is not supported by this package.
__In order to expand this class one can in essence create a Baseclass with the additional functionality and have 
the main Model class inherit from each selected model class (Lasso, Ridge, ...) by the user and the created Baseclass. This 
removes the need for a lot of code repetition.__<br>  

3) __Encoders in [encoders.py](./fentool/pre_process/encoders.py)__<br/> 
This class encompasses different encoding methods such as ordinal and one-hot encoding. This can also be accessed separately 
by the user without going through toolkit. 

4) __Transformers in [transformers.py](./fentool/pre_process/transformers.py)__<br/>
This class has implementation for minmax normalization and standardization using mean and standard deviation. The general idea is 
to include many more transformation features within this routine. The upcoming feature is the feature combinations based on user 
input as well as feature importance features based on feature permutation techniques. 

# Usage

An example for the usage can be found below:
```python 
fent = Fentool()

models = ['linreg','lassocv']
encoder_types =['one-hot']
input_trans = [None, 'normalize', 'standardize']
output_trans = ['standardize']

scores = fent.model_compare(models=models,
                            encoder_types=encoder_types,
                            input_trans=input_trans,
                            output_trans = output_trans,
                            df=TestFentool.df,
                            target='median_house_value')


```
In the above example fentool combines the models, encoder types as well as input
and output treatment effects and outputs a dataframe containing the combined use cases and their respective scoring.
In addition to the treatment and encoding settings one needs to provide the dataframe as well as the target variable in that 
dataframe.

Other examples of usage are given here with how one can focus on a single case to evaluate the scoring on the test(Default) set.
```python 
fent = Fentool(encoder_type='one-hot', input_treatment='normalize',
                   output_treatment='normalize')

score = fent.evaluate_model(TestFentool.df
                            , target='median_house_value')
```
