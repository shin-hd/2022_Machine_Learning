# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import SHD as util

"""
Please change "util" module name to your initial of name for scoring 
Eg) If your name is 홍길동 
--> util.py --> GDH.py
--> (In line 6) import util as util -> import GDH as util
"""

# data loading using pandas
df = pd.read_csv('iris.csv')

# categorical label to numerical label
df['variety'] = df['variety'].astype('category').cat.codes
# shuffling
df =  df.sample(frac=1).reset_index(drop=True)

# separation of data and labels 
data = df.iloc[:, :-1].to_numpy()
label = df.iloc[:, [-1]].to_numpy().squeeze()

label_num = max(label) + 1

# feature normalization
normalled_data = util.feature_normalization(data)
print("Mean:", np.mean(data, 0))
print("normalled_Mean:", np.mean(normalled_data, 0))

data = data # or normalled_data

# spilt data for testing
# 100 => training data : 100 / test data : 50 
split_factor = 100
training_data, test_data, training_label, test_label = util.split_data(data, label, split_factor)

# get train parameter of nomal distribution and prior probability
mu, sigma = util.get_normal_parameter(training_data, training_label, label_num)
prior = util.get_prior_probability(training_label, label_num)

# get postereior probability of each test data based on likelihood and prior
posterior = util.Gaussian_NB(mu, sigma, prior, test_data)

# classification using posterior
prediction = util.classifier(posterior)

# get accuracy
acc, hit_num = util.accuracy(prediction, test_label)

# print result
print(f'accuracy is {acc}% !')
print(f'the number of correct prediction is {hit_num} of {len(test_label)} !')

