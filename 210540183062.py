# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:32:18 2021

@author: Danish S. Kadri.

"""
Consider the dataset named Glass.csv (10 marks)
The outcome class is contained in a factor variable called Type
Variable Type is the target(response) variable. Try the following algorithms and examine which of these is 
the best fit for this data (with log loss):
a) Linear Discriminant Analysis ( K-Fold CV )
b) Gaussian Naive Bayes ( K-Fold CV )
c) Random Forest ( tune with few parameter sets )
( Marks: a) and b) 3 each, 1 for using correct function and 1 for Python code
d) 4 marks, 1 for using correct function and 3 for Python code for grid search)
"""
"""
import pandas as pd
import numpy as np


df = pd.read_csv("F:\eDBDA\Study material\ML\ML lab exam\PML_Labexam\Glass.csv")

X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# Unique classes
print(df['Type'].unique())

# Label Encoding for multi-class
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df.iloc[:,-1])

# Import the necessary modules
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

################ LDA #######################

da = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
results = cross_val_score(da, X, y, cv=kfold, scoring='neg_log_loss')
print(results)
print("Log_Loss: %.4f (%.4f)" % (results.mean(), results.std()))

# Using Accuracy Score
results = cross_val_score(da, X, y, cv=kfold)
print(results)
print("Accuracy Score: %.4f (%.4f)" % (results.mean(), results.std()))


###############################################################################################################################

###############  Gaussian Naive Bayes ( K-Fold CV ) #################################

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
results = cross_val_score(gaussian, X, y, cv=kfold,scoring='neg_log_loss')
print(results)
print("Log_Loss: %.4f (%.4f)" % (results.mean(), results.std()))
# Using Accuracy Score
results = cross_val_score(gaussian, X, y, cv=kfold)
print(results)
print("Accuracy Score: %.4f (%.4f)" % (results.mean(), results.std()))

######################################## Random Forest ( tune with few parameter sets ) ############################################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'max_features': np.arange(1,11)}

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)

model_rf = RandomForestClassifier(random_state=2021)
cv = GridSearchCV(model_rf, param_grid=parameters,cv=kfold,scoring='neg_log_loss')

cv.fit( X , y )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)

####################################################################################################################################3

