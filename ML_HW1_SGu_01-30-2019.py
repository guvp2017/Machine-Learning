#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 20:57:00 2019

@author: shanqinggu
"""

## Running enviroments: Python 3.5.6 |Anaconda custom (64-bit), IPython 6.5.0, Spyder 3.3.1
## References: 1) Chris's office hour files 
##             2) Titanic data from http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv
##             3) Grid search with multiple classifiers (modified from http://www.davidsbatista.net/blog/2018/02/23/model_optimization/)

## --------------------------------------------------------------------------------------------------------
## Import libraries
## --------------------------------------------------------------------------------------------------------
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from itertools import product

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 

print(__doc__)
 

## --------------------------------------------------------------------------------------------------------
## Recommend to be done before live class 2
## --------------------------------------------------------------------------------------------------------

#### 1. Write a function to take a list or dictionary of clfs and hypers 
####    ie use logistic regression, each with 3 different sets of hyper parrameters for each
## --------------------------------------------------------------------------------------------------------

# List of classfiers
clfsList_LR = [LogisticRegression, RandomForestClassifier] 

# Dictionary of params in each classifier
clfDict_LR = {'LogisticRegression': {"C": [0.1, 1, 10],
                                     "tol": [0.001,0.01,0.1],
                                     "penalty" : ['l1', 'l2' ]}, 

           'RandomForestClassifier': {"n_estimators":[1, 10, 50], 
                                      "max_depth":[5, 10, 15],
                                      "min_samples_split":[2, 4, 6, 8]}
           }
           
# Function myClfHypers() to take a list or dictionary of clfs and hypers
def myClfHypers_LR(clfsList_LR):
    for clf in clfsList_LR:
        clfString = str(clf)
        print("clf: ", clfString)
        
        for k1, v1 in clfDict_LR.items():  # go through first level of clfDict_LR
            if k1 in clfString:		       # if clfString1 matches first level
                for k2,v2 in v1.items():   # go through the inner dictionary of hyper parameters
                    print(k2)			   # for each hyper parameter in the inner list..	
                    for vals in v2:		   # go through the values for each hyper parameter 
                        print(vals)		   # and print                      

myClfHypers_LR(clfsList_LR)

## --------------------------------------------------------------------------------------------------------
## Recommend to be done before live class 3

# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
## --------------------------------------------------------------------------------------------------------

#### 2. expand to include larger number of classifiers and hyperparmater settings
## --------------------------------------------------------------------------------------------------------

## Itertools list with 3 classifiers
clfsList_LRD = [LogisticRegression, RandomForestClassifier, DecisionTreeClassifier] 

## Itertools dictionary with larger number of hyperparameter settings
clfDict_LRD = {'LogisticRegression': {"C" : [0.1, 1, 10, 50], 
                                  "tol" : [0.001,0.01,0.1, 1],
                                  "penalty" : ['l2' ],
                                  "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']},
               
           'RandomForestClassifier': {"n_estimators":[1, 10, 50, 100], 
                                      "max_depth":[5, 10, 15, 20],
                                      "min_samples_split":[2, 4, 6, 8]},
                                      
           'DecisionTreeClassifier': {"max_depth":[2, 4, 6], 
                                      "criterion":['entropy', 'gini']}
           }
# Use zip() and product() to feed the operator in run()
for k1, v1 in clfDict_LRD.items():  # go through the inner dictionary of hyper parameters
    k2,v2 = zip(*v1.items())        # use zip function
    for values in product(*v2):     # for the values in the inner dictionary, get their unique combinations from product()
            hyperSet_LRD = dict(zip(k2, values)) # create a dictionary from their values
            print(hyperSet_LRD)     # print out the results in a dictionary that can be used to feed into the operator in run()


#### 3. find some simple data and 
## --------------------------------------------------------------------------------------------------------
            
## Use prep_data() to prepare titanic data for logistic regression         
def prep_data(df):
    dummy_cols = ['Sex'] # make dummy columns for 'Sex' 
    drop_cols = ['Name', 'Age'] # drop 2 columns,  remove 'Age'because of 177 missing values
    for col in dummy_cols:
        new_cols = pd.get_dummies(df[col])
        df = df.join(new_cols)
    df = df.drop(dummy_cols, axis=1)
    return df.drop(drop_cols, axis=1)

# Import Titanic data from Stanford CS website
train = pd.read_csv('http://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

# Use prep_data() to prepare data and checke with dtypes() and head()
df_train = prep_data(train)
print(df_train.dtypes)
print(df_train.head(5)) 

# Create arrays for the features and the response variable
M = df_train.drop('Survived', axis=1).values
L = df_train['Survived'].values

# Use for 5-fold (k-fold) cross validation
n_folds = 5

# Pack the arrays together into "data"
data = (M,L,n_folds)

# Expand data
M, L, n_folds = data
kf = KFold(n_splits=n_folds)
print(kf)

# k-fold splitting 
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    print("k fold = ", ids)
    # print("            train indexes", train_index)
    # print("            test indexes", test_index)

# run() for all our classifiers against titanic data
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data           # unpack the "data" container of arrays
  kf = KFold(n_splits=n_folds)   # Establish the cross validation 
  ret = {}                       # classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): # interate through train and test indexes by using kf.split from M and L
                                                                   # simply splitting rows into train and test rows for 5 folds
    
    clf = a_clf(**clf_hyper) # unpack paramters into clf if they exist 
                             # give all keyword arguments except for those corresponding to a formal parameter in a dictionary
            
    clf.fit(M[train_index], L[train_index])   # first param, M when subset by "train_index", includes training X's 
                                              # second param, L when subset by "train_index", includes training Y                             
    
    pred = clf.predict(M[test_index])         # use M -our X's- subset by the test_indexes, predict the Y's for the test rows
    
    ret[ids]= {'clf': clf,                    # create arrays
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
    
  return ret


#### 4. generate matplotlib plots that will assist in identifying the optimal clf and parampters settings
## --------------------------------------------------------------------------------------------------------
  
# Use populateClfAccuracyDict() to create classifier accuracy dictionary with all values
def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) # convert k1 to a string
                        
        # String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
            
# Use myHyperSetSearch() to populate clfsAccuracyDict with results
def myHyperSetSearch(clfsList,clfDict):
    for clf in clfsList:
        clfString = str(clf) #check if values in clfsList are in clfDict
        print("clf: ", clfString)
        
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
            if k1 in clfString:
                k2,v2 = zip(*v1.items())
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperSet = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperSet) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results
 

for clfs in clfsList_LRD:
    results = run(clfs, data, clf_hyper={})


# run() with a list and a for loop
clfsList_LRD = [LogisticRegression, RandomForestClassifier, DecisionTreeClassifier] 

clfDict_LRD = {'LogisticRegression': {"C" : [0.1, 1, 10, 50], 
                                  "tol" : [0.001,0.01,0.1, 1],
                                  "penalty" : ['l2' ],
                                  "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']},
        
           'RandomForestClassifier': {"n_estimators":[1, 10, 50, 100],                                      
                                      "max_depth":[5, 10, 15, 20],
                                      "min_samples_split":[2, 4, 6, 8]},
                                      
           'DecisionTreeClassifier': {"max_depth":[2, 4, 6], 
                                      "criterion":['entropy', 'gini']}
           }

# declare empty clfsAccuracyDict to populate in aaa  
clfsAccuracyDict = {}

# run myHyperSetSearch()
myHyperSetSearch(clfsList_LRD, clfDict_LRD)  
print(clfsAccuracyDict)

# for determining maximum frequency (# of kfolds) for histogram y-axis
n = max(len(v1) for k1, v1 in clfsAccuracyDict.items())

# for naming the plots
filename_prefix = 'clf_Histograms_'

# initialize the plot_num counter for incrementing in the loop below
plot_num = 1 

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height
               
# create the histograms
for k1, v1 in clfsAccuracyDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values 
    fig = plt.figure(figsize =(10,5)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=15) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=12) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=12) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=10) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=10) # increase y-axis tick fontsize for readability
    ax.grid(False) # not show grid here

    # Activate these codes to save plot figures by passing in subplot adjustments from above.
    
    # plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    # plot_num_str = str(plot_num) #convert plot number to string
    # filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    # plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    # plot_num = plot_num+1 # increment the plot_num counter by 1
    
plt.show()

## --------------------------------------------------------------------------------------------------------
# Recommend to be done before live class 4 

# 5. Please set up your code to be run and save the results to the directory that it is executed from
# 6. Investigate grid search function
## --------------------------------------------------------------------------------------------------------

#### 5. Please set up your code to be run and save the results to the directory that it is executed from
## --------------------------------------------------------------------------------------------------------

# Run and saved the results to the directory that it is executed from

##### 6. Investigate grid search function
## --------------------------------------------------------------------------------------------------------

# 6_1 Grid search with LogisticRegression() by precision and recall
# By precision: best parameters set found on development set: {'penalty': 'l2', 'solver': 'newton-cg', 'C': 1, 'tol': 1}
# By recall, best parameters set found on development set: {'tol': 1, 'C': 1, 'penalty': 'l2', 'solver': 'newton-cg'}
## --------------------------------------------------------------------------------------------------------

# turn the titanic data (df_train) in a (samples, feature) matrix:
X = df_train.drop('Survived', axis=1).values
y = df_train['Survived'].values

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Set the LogisticRegression() parameters by cross-validation (cv = 5)
tuned_parameters = [{"C" : [0.1, 1, 10, 50], 
                     "tol" : [0.001,0.01,0.1, 1],
                     "penalty" : ['l2' ],
                     "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
   # print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    #print()
    
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    #print()


# 6_2 Randomized grid search with LogisticRegression() by precision and recall
## --------------------------------------------------------------------------------------------------------
    
# The model with rank 1:  mean validation score: 0.795 (std: 0.018) and parameters: {'tol': 0.01, 'C': 1, 'solver': 'liblinear'}    
    
clf = LogisticRegression(max_iter=100)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"C" : [0.1, 1, 10, 50],
              "tol" : [0.001,0.01,0.1, 1],
              "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']}

# run randomized search
max_iter_search = 100
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=max_iter_search, cv=5)

start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), max_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {"C" : [0.1, 1, 10, 50],
               "tol" : [0.001,0.01,0.1, 1],
               "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_) 


# 6_3 Grid search with multiple classifiers 
## --------------------------------------------------------------------------------------------------------

# Code modified from David S. Batista (http://www.davidsbatista.net/blog/2018/02/23/model_optimization/)

# The idea is to pass two dictionaries to a helper class (the models, the the parameters); 
# then call the fit method, wait until everything runs, 
# and after calling the score_summary() method,  a nice DataFrame to report each model instance, according to the parameters

# Define a EstimatorSelectionHelper class by passing the models and the parameters
class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
            
    # The score_summary() to check each model and each parameters
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})
                
        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]

# A dictionary of models with 3 classifiers     
models = {
        'RandomForestClassifier': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(),
        'DecisionTreeClassifier': DecisionTreeClassifier()
        }
# A dictionary of parameters to select in each classfier
params = {'LogisticRegression': {"C" : [0.1, 1, 10, 50], 
                                  "tol" : [0.001,0.01,0.1, 1],
                                  "penalty" : ['l2' ],
                                  "solver":['newton-cg','lbfgs','liblinear', 'sag', 'saga']},
           'RandomForestClassifier': {"n_estimators":[1, 10, 50, 100],                                      
                                      "max_depth":[5, 10, 15, 20],
                                      "min_samples_split":[2, 4, 6, 8]},
           'DecisionTreeClassifier': {"max_depth":[2, 4, 6], "criterion":['entropy', 'gini']}
}

# call the fit() function, which as signature similar to the original GridSearchCV object
helper = EstimatorSelectionHelper(models, params)
helper.fit(X, y, scoring='f1', n_jobs=2) # scoring with 'f1'

# Inspect results of each model and each parameters by calling the score_summary method
summary = helper.score_summary(sort_by='max_score')
print(summary.T) # show score summary (e.g.,the best and the worst classfier with parameters in 13 rows x 150 columns)

## --------------------------------------------------------------------------------------------------------

## --------------------------------------------------------------------------------------------------------
## The End of HW1
## --------------------------------------------------------------------------------------------------------
