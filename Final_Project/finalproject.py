#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 17:46:15 2018

@author: dustinscarter
"""
# imports
from patsy import dmatrices
import pandas as pd
import statsmodels.discrete.discrete_model as sm

# read in the data & create matrices
df1 = pd.read_csv("adultdata.csv")
a = []
a = df1

y, X = dmatrices('income ~ age + wclass + fnlweight + ed_num \
             + mstatus + relat + job + race + gender + \
             cg + cl + hours + origin',a, return_type = 'dataframe')
# sm
logit = sm.Logit(y, X)
result = logit.fit()
print(result.summary())



