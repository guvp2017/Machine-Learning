#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:30:14 2019

@author: shanqinggu
"""

## ---------------------------------------------------------------
## Homework 3 for DS 7335 Machine Learning

## ---------------------------------------------------------------
## Running enviroments: Python 3.7.0, IPython 7.2.0, Spyder 3.3.3
## References: Office hour codes from Chris 
## ---------------------------------------------------------------

## Import libraries
## ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm #https://matplotlib.org/api/cm_api.html
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import rankdata

## Decision making with Matrices

## This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.

## The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  
## Then you should decided if you should split into two groups so eveyone is happier.

## Displicte the simplictiy of the process you will need to make decisions regarding how to process the data.

## This process was thoughly investigated in the operation research community.  This approah can prove helpful on any number of 
## decsion making problems that are currently not leveraging machine learning.


## ----------------------------------------------------------------------------------------------------------
########## Assignment begins here ##########

## Q1: You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.
## ----------------------------------------------------------------------------------------------------------

## Generate random values for weights to sum 1 by np.random.dirichlet()
## or can sum column values to 1 by matrix/matrix.sum(axis=1,keepdims=1)

# print(np.array([np.random.dirichlet(np.ones(6),size=1)]))

people = {'Frank': {'willingness to travel': 0.30506713, 
                  'desire for new experience':0.10713288,
                  'cost_p':0.32906632,
                  'Asian food':0.07634888,
                  'Italian food':0.11721406,                  
                  'vegetarian': 0.06517074,                  
                  },
          'Jens': {'willingness to travel': 0.01347915,
                  'desire for new experience':0.23944182,
                  'cost_p':0.32467904,
                  'Asian food':0.00993189,
                  'Italian food':0.25654978,                  
                  'vegetarian': 0.15591832,
                  },
          'Marina': {'willingness to travel': 0.64085687 ,
                  'desire for new experience': 0.04966569,
                  'cost_p': 0.03263205,
                  'Asian food':0.0563543,
                  'Italian food':0.11814616,                  
                  'vegetarian': 0.10234494,
                  },
          'Wolfgang': {'willingness to travel': 0.02931469,
                  'desire for new experience': 0.21787225,
                  'cost_p': 0.06303935,
                  'Asian food':0.54456268,
                  'Italian food':0.05910163,
                  'vegetarian': 0.08610941,
                  },
          'Natalia': {'willingness to travel': 0.04145057,
                  'desire for new experience': 0.30058704,
                  'cost_p': 0.56440961,
                  'Asian food':0.02337695,
                  'Italian food':0.01847019,                  
                  'vegetarian': 0.05170564,
                  },
          'Roland': {'willingness to travel': 0.0192539,
                  'desire for new experience': 0.12451716,
                  'cost_p': 0.234014 ,
                  'Asian food':0.40058628,
                  'Italian food':0.21757027,                  
                  'vegetarian': 0.00405839,
                  },
          'Sabina': {'willingness to travel': 0.11021807,
                  'desire for new experience': 0.22829212,
                  'cost_p': 0.36609202,
                  'Asian food':0.15183427,
                  'Italian food':0.1184194,                  
                  'vegetarian': 0.02514412,
                  },
          'Georg': {'willingness to travel': 0.03948403,
                  'desire for new experience': 0.36041876,
                  'cost_p': 0.04094349,
                  'Asian food':0.03249659,
                  'Italian food':0.29251257,                  
                  'vegetarian': 0.23414456,
                  },
          'Alan': {'willingness to travel': 0.04329632,
                  'desire for new experience': 0.14277187,
                  'cost_p': 0.08572584,
                  'Asian food':0.22139147,
                  'Italian food':0.34694498,
                  'vegetarian': 0.15986952,
                  },
          'Jackson': {'willingness to travel': 0.20067984,
                  'desire for new experience': 0.10635372,
                  'cost_p': 0.30499546,
                  'Asian food':0.023558,
                  'Italian food':0.35644336,
                  'vegetarian': 0.00796962,
                  }                  
          }
          
## -------------------------------------------------------------------------------------------------------------------------------
## Q2: Transform the user data into a matrix(M_people). Keep track of column and row ids.
## Added: normalize the points for each user -- make their preferences add to 1 in the actual weights matrix you use for analysis.
## --------------------------------------------------------------------------------------------------------------------------------
          
## convert each person's values to a list

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            
## lists to show column keys and values
print(peopleKeys)
print(peopleValues)


peopleMatrix = np.array(peopleValues)
peopleMatrix.shape
peopleMatrix

## normalize their preferences and make column sum to 1
## Ref: https://stackoverflow.com/questions/43644320/how-to-make-numpy-array-column-sum-up-to-1
peopleMatrix= peopleMatrix/peopleMatrix.sum(axis=1,keepdims=1)
peopleMatrix
peopleMatrix.sum(1) # check if sum to 1

## -------------------------------------------------------------------------------------------------------------
## Q3 Next you collected data from an internet website. You got the following information.
##  Added: make these scores /10, on a scale of 0-10, where 10 is good. So, 10/10 for distance means very close. 
## -------------------------------------------------------------------------------------------------------------

# np.random.randint(10, size=10)+1 # generate random score scales from 0 to 10

restaurants  = {'Fukuyu':{'distance' : 7,
                        'novelty' : 5,
                        'cost': 8,
                        'average rating': 4,
                        'cuisine': 2,
                        'vegetarian': 3
                        },
              'Moretti':{'distance' : 10,
                        'novelty' : 9,
                        'cost': 5,
                        'average rating': 6,
                        'cuisine': 3,
                        'vegetarian': 1
                      },
              'La Tavola':{'distance' : 2,
                        'novelty' : 4,
                        'cost': 9,
                        'average rating': 3,
                        'cuisine': 7,
                        'vegetarian': 8
                      },                      
              'Bonsai':{'distance' : 3,
                        'novelty' : 9,
                        'cost': 7,
                        'average rating': 1,
                        'cuisine': 5,
                        'vegetarian': 6
                      },
              'DaVinci':{'distance' : 8,
                        'novelty' : 7,
                        'cost': 10,
                        'average rating': 8,
                        'cuisine': 5,
                        'vegetarian': 4
                      },
              'Fox':{'distance' : 7,
                        'novelty' : 9,
                        'cost': 2,
                        'average rating': 3,
                        'cuisine': 10,
                        'vegetarian': 2
                      },
              'Mirchi':{'distance' : 5,
                        'novelty' : 5,
                        'cost': 9,
                        'average rating': 4,
                        'cuisine': 7,
                        'vegetarian': 3
                      },
              'Momo Ghar':{'distance' : 4,
                        'novelty' : 4,
                        'cost': 2,
                        'average rating': 6,
                        'cuisine': 9,
                        'vegetarian': 2
                      },
              'Gogi':{'distance' : 2,
                        'novelty' : 10,
                        'cost': 10,
                        'average rating': 9,
                        'cuisine': 2,
                        'vegetarian': 5
                      },
              'Bravo':{'distance' : 10,
                        'novelty' : 2,
                        'cost': 6,
                        'average rating': 10,
                        'cuisine': 3,
                        'vegetarian': 3
                      }                      
}
 
## ------------------------------------------
########### Data processing ends #########
## ------------------------------------------  
########### Start with 2 numpy matrices if you're not excited to do data processing atm ##########              
## ------------------------------------------------------------------------------------------------
## Q4 Transform the restaurant data into a matrix(M_resturants) use the same column index.
## ------------------------------------------------------------------------------------------------
              
restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

## lists to show column keys and values
print(restaurantsKeys)
print(restaurantsValues)

len(restaurantsValues)


## converte lists to np.arrays
restaurantsMatrix = np.reshape(restaurantsValues, (10,6))
restaurantsMatrix
restaurantsMatrix.shape

## -----------------------------------------------------------------------------------------------
## Q5 The most imporant idea in this project is the idea of a linear combination.
## Informally describe what a linear combination is and how it will relate to our resturant matrix.
## -----------------------------------------------------------------------------------------------
"""
## Linear combination is a sum of the elements from some set with constant coefficients placed in front of each. 
## For example, a linear combination of the vectors {x, y, z} in a vector space, is given by ax+by+cz, where a, b, and c are constants.

## For resturant matrix, we will construct the preference vector for each user by a linear combination of the feature vectors of all the restaurants with the scales as the weight. 
## Rank the restaurants based on the similarities between feature vectors of the restaurants and preference vectors of the users.
## Then we use locally linear regression to predict ratings and the rankings as input and ratings as the labels.
"""

## ---------------------------------------------------------------------------------------
## Q6 Choose a person and compute(using a linear combination) the top restaurant for them.  
## What does each entry in the resulting vector represent?
## ---------------------------------------------------------------------------------------

"""    
## Choose Frank and resturant Fukuyu with scales (7, 5, 8, 4, 2, 3)  

## Frank and Fukuyu (matrix position(0,0)):
7*0.30506713+5*0.10713288+8*0.32906632+4*0.07634888+2*0.11721406+3*0.06517074
Out[]: 6.039000730000001

## Each entry in the resulting vector represents the resultant vector from two vector multiplication (e.g. Frank and Fukuyu).
"""    
print(peopleKeys)
print(peopleValues)

print(restaurantsKeys)
print(restaurantsValues)

restaurantsMatrix.shape, peopleMatrix.shape

## Need to swap axis on peopleMatrix
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.swapaxes.html
newPeopleMatrix = np.swapaxes(peopleMatrix, 0, 1)

newPeopleMatrix.shape, restaurantsMatrix.shape
restaurantsMatrix.shape, newPeopleMatrix.shape

## ----------------------------------------------------------------------------------------
## Q7 Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  
## What does the a_ij matrix represent?
## Let's check our answers
## ----------------------------------------------------------------------------------------

## Compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people
M_usr_x_rest = np.matmul(restaurantsMatrix, newPeopleMatrix)
M_usr_x_rest  
   
## What does the a_ij matrix represent?

"""
## a_ij is the score for a restaurant for a person, and x is the person, y is the restaurant
## x=0 (Frank), x=1 (Jens), x=2 (Marina), x=3 (Wolfgang), x=4 (Natalia), x=5 (Roland), x=6 (Sabina), x=7 (Georg), x=8 (Alan), x=9 (Jackson)
## y=0 (fFukuyu), y=1 (Moretti), y=2 (La Tavola), y=3 (Bonsai), y=4 (DaVinci),y=5 (Fox), y=6 (Mirchi), y=7 (Momo Ghar), y=8 (Gogi), y=9 (Bravo)
"""
## Let's check our answers: 
## After checking, the a(0,0) calculation in Q6 by using linear combination is correct (6.03900067) 
                
"""
array([[6.03900067, 4.90957755, 5.76412722, 4.35366105, 6.59393117, 4.67913593, 5.76133149, 3.82347511, 3.76180471, 5.20751874],
       [6.53510495, 4.89832208, 7.8136293 , 6.09998422, 6.18921502, 5.54355032, 6.27867782, 5.1499884 , 4.67559251, 5.70760688],
       [5.57117358, 6.97982763, 3.58891045, 4.23374722, 6.97800309, 5.39992029, 5.91402437, 5.90736875, 5.80095811, 6.20126818],
       [5.23630512, 6.69635782, 3.85914087, 3.84679702, 7.20646408, 4.32930211, 5.84272342, 6.54875216, 4.93025016, 5.54778374],
       [7.93867462, 7.01659364, 7.25178018, 7.38646393, 8.56599905, 7.67456647, 8.04805724, 5.90735227, 6.34836679, 7.40242864],
       [5.28932714, 5.80581862, 6.55346771, 4.68905489, 4.4825006 , 5.10903806, 5.24832466, 7.09294862, 6.212836  , 6.62298012],
       [6.34400303, 6.49004719, 5.10577634, 4.65357916, 7.16779059, 5.96249348, 6.49908437, 5.24801339, 5.49566281, 6.89337132],
       [3.95029394, 4.34141796, 4.43348542, 5.08633598, 2.90687435, 5.41287913, 4.11329326, 4.97737993, 5.68631712, 5.20340264],
       [6.21954794, 7.05024507, 3.35989683, 8.31755973, 9.23872877, 7.68452841, 7.89334537, 6.14080781, 5.85733053, 5.4796083 ],
       [6.54997811, 3.89847258, 7.9287086 , 6.98838735, 4.84643443, 6.5164061 , 5.70435032, 3.26627605, 4.96722018, 5.37829754]])
"""

## --------------------------------------------------------------------------------------------------------
## Q8 Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
## --------------------------------------------------------------------------------------------------------

## Sum all columns in M_usr_x_rest
M_usr_x_rest_sum = np.sum(M_usr_x_rest, axis=1)
M_usr_x_rest_sum

restaurantsKeys

## Get optimal resturant for all users (https://www.geeksforgeeks.org/zip-in-python/)

resturants_names = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']

## from scipy.stats import rankdata
ranking = rankdata([-1 * i for i in M_usr_x_rest_sum]).astype(int) # rank 1 as the the highest

## Use zip() to map values 
mapped = zip(resturants_names, M_usr_x_rest_sum, ranking) 

## Convert values to print as set 
mapped = set(mapped)
for rn, muxr, rnk in mapped:
    print ("Resturant :  %s  Scores : %d Ranking: %d" %(rn, muxr, rnk)) 

## DaVinci Resturant is ranked 1 with score 73
"""
Resturant :  Momo Ghar  Scores : 46 Ranking: 10
Resturant :  La Tavola  Scores : 56 Ranking: 6
Resturant :  Bravo  Scores : 56 Ranking: 7
Resturant :  Gogi  Scores : 67 Ranking: 2
Resturant :  Moretti  Scores : 58 Ranking: 4
Resturant :  Bonsai  Scores : 54 Ranking: 8
Resturant :  Fox  Scores : 57 Ranking: 5
Resturant :  Morchi  Scores : 59 Ranking: 3
Resturant :  DaVinci  Scores : 73 Ranking: 1
Resturant :  Fukuyu  Scores : 50 Ranking: 9
""" 
## ------------------------------
########### CLASS ENDS #########
## ------------------------------      

## --------------------------------------------
########### Discuss with class mates ##########
## --------------------------------------------
    
## --------------------------------------------------------------------------------------------------------
## Q9 Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
## Do the same as above to generate the optimal resturant choice.
    
## For each user

## Angela writes: Jack scores for 
##                 scores        ranking 
## Tacos             74            1    
## tapas             50            3   
## bar               70            2

## Say that rank 1 is best
## --------------------------------------------------------------------------------------------------------
M_usr_x_rest

## Rank the whole matrix (10:10)
M_usr_x_rest_rank_all = np.reshape(rankdata(-M_usr_x_rest, method='max'), (10, 10)) # -results to rank 1 is the best
M_usr_x_rest_rank_all

## Rank for each user (stackoverflow.com/questions/27736753/rank-within-columns-of-2d-array)
M_usr_x_rest_rank = M_usr_x_rest_rank_all.argsort(axis=0).argsort(axis=0) + 1
M_usr_x_rest_rank

## M_usr_x_rest_rank2 = M_usr_x_rest.argsort(axis=0)[::-1] +1 ## This one does noot sort correctly

## Ref: https://stackoverflow.com/questions/5048299/how-do-i-print-an-aligned-numpy-array-with-text-row-and-column-labels
peop_name = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabina', 'Georg', 'Alan', 'Jackson']
rest_name = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']
rank_table = np.zeros((11,11), object)
rank_table[1:,1:]= M_usr_x_rest_rank
rank_table[0,1:]=peop_name
rank_table[1:,0]=rest_name
rank_table[0,0]=''
printer = np.vectorize(lambda peop_name:'{0:10}'.format(peop_name,))

np.set_printoptions(linewidth=np.inf)
print(printer(rank_table).astype(object))

"""
M_usr_x_rest
array([[6.03900067, 4.90957755, 5.76412722, 4.35366105, 6.59393117, 4.67913593, 5.76133149, 3.82347511, 3.76180471, 5.20751874],
       [6.53510495, 4.89832208, 7.8136293 , 6.09998422, 6.18921502, 5.54355032, 6.27867782, 5.1499884 , 4.67559251, 5.70760688],
       [5.57117358, 6.97982763, 3.58891045, 4.23374722, 6.97800309, 5.39992029, 5.91402437, 5.90736875, 5.80095811, 6.20126818],
       [5.23630512, 6.69635782, 3.85914087, 3.84679702, 7.20646408, 4.32930211, 5.84272342, 6.54875216, 4.93025016, 5.54778374],
       [7.93867462, 7.01659364, 7.25178018, 7.38646393, 8.56599905, 7.67456647, 8.04805724, 5.90735227, 6.34836679, 7.40242864],
       [5.28932714, 5.80581862, 6.55346771, 4.68905489, 4.4825006 , 5.10903806, 5.24832466, 7.09294862, 6.212836  , 6.62298012],
       [6.34400303, 6.49004719, 5.10577634, 4.65357916, 7.16779059, 5.96249348, 6.49908437, 5.24801339, 5.49566281, 6.89337132],
       [3.95029394, 4.34141796, 4.43348542, 5.08633598, 2.90687435, 5.41287913, 4.11329326, 4.97737993, 5.68631712, 5.20340264],
       [6.21954794, 7.05024507, 3.35989683, 8.31755973, 9.23872877, 7.68452841, 7.89334537, 6.14080781, 5.85733053, 5.4796083 ],
       [6.54997811, 3.89847258, 7.9287086 , 6.98838735, 4.84643443, 6.5164061 , 5.70435032, 3.26627605, 4.96722018, 5.37829754]])


M_usr_x_rest ranking with rank 1 is best
[['          ' 'Frank     ' 'Jens      ' 'Marina    ' 'Wolfgang  ' 'Natalia   ' 'Roland    ' 'Sabina    ' 'Georg     ' 'Alan      ' 'Jackson   ']
 ['Fukuyu    ' '         6' '         7' '         5' '         8' '         6' '         9' '         7' '         9' '        10' '         9']
 ['Moretti   ' '         3' '         8' '         2' '         4' '         7' '         5' '         4' '         7' '         9' '         5']
 ['La Tavola ' '         7' '         3' '         9' '         9' '         5' '         7' '         5' '         4' '         4' '         4']
 ['Bonsai    ' '         9' '         4' '         8' '        10' '         3' '        10' '         6' '         2' '         8' '         6']
 ['DaVinci   ' '         1' '         2' '         3' '         2' '         2' '         2' '         1' '         5' '         1' '         1']
 ['Fox       ' '         8' '         6' '         4' '         6' '         9' '         8' '         9' '         1' '         2' '         3']
 ['Morchi    ' '         4' '         5' '         6' '         7' '         4' '         4' '         3' '         6' '         6' '         2']
 ['Momo Ghar ' '        10' '         9' '         7' '         5' '        10' '         6' '        10' '         8' '         5' '        10']
 ['Gogi      ' '         5' '         1' '        10' '         1' '         1' '         1' '         2' '         3' '         3' '         7']
 ['Bravo     ' '         2' '        10' '         1' '         3' '         8' '         3' '         8' '        10' '         7' '         8']]
"""    
## -------------------------------------------------------------------------------------------------------------
## Q10 Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
## -------------------------------------------------------------------------------------------------------------

"""
## The best resurant proposed by all users is DaVinci.
## This is in agreement with the best for Frank, Sabina, Alan and Jackson.
## This resturant is listed as the 2nd option for Jens, Wolfgang, Natalia and Roland 
## This resturant is the 3rd option for Marina and 5th option for Georg. 
## In the real world, this scenario always happens. Each individual may react differently even in a very similar situation. 
## This situation is noted as the Condorcet paradox (also known as voting paradox or the paradox of voting) in social choice theory.
## Collective preferences can be cyclic, even if the preferences of individual voters are not cyclic. 
## This is paradoxical, because it means that majority wishes can be in conflict with each other.
## It is because the conflicting majorities are each made up of different groups of individuals. 

## Same problem for the 2nd best resturant Gogi (1st option for Jens, Wolfgang, Natalia and Roland, 2nd for Sabine, 3rd for Georg and Alan, 5th for Frank, 7th for Jackson)

## Majority wishes can be in conflict with each other. When this occurs, it is because the conflicting majorities are each made up of different groups of individuals.
"""

## ---------------------------------------------------------------
## Q11 How should you preprocess your data to remove this problem.
## ---------------------------------------------------------------

"""
Reduce dimension (use less feature variables) can solve this problem. For example, if we only keep two features variables ('distance' and 'novelty'),
Resturant 'Moretti' is ranked as the first (the best option) from column sum to get optimal resurant.
This ranking gets 8/10 votes from 

## Also tried StandardScaler() to normalize, but it is not working in this way.

The necessary condition that a voting paradox can happen only if all three
preference lists remaining after cancellation have the same spin is not also suffi- cient.
(a) Give an example of a vote where there is a majority cycle and addition of one more voter with the same spin causes the cycle to go away.
(b) Can the opposite happen; can addition of one voter with a “wrong” spin cause a cycle to appear?
(c) Give a condition that is both necessary and sufficient to get a majority cycle.

Paradoxes in multi-attribute rankings occur more frequently than one would expect. 
According to Coombs’s condition [2], the chance of paradox is over 97% when six alternatives are ranked using multiple attributes. 
When ten alternatives with multiple attributes are considered, the chance of paradox is virtually 100%. 
Although extensive work has been done on minimizing the voting paradox, such as the Borda Count voting method by Saari [33, 34], 
these methods can only minimize voting paradoxes while not completely eliminating them.

"""
# peopleMatrix
# restaurantsMatrix

peopleMatrix_k2 = np.delete(peopleMatrix, (2,3,4,5), 1) # only keep 2 columns ('distance' and 'novelty')
peopleMatrix_k2

restaurantsMatrix_k2 = np.delete(restaurantsMatrix, (2,3,4,5), 1) #remove cost feature form resturant
restaurantsMatrix_k2

newPeopleMatrix_k2 = np.swapaxes(peopleMatrix_k2, 0, 1)
newPeopleMatrix_k2

## https://stackoverflow.com/questions/43644320/how-to-make-numpy-array-column-sum-up-to-1
peopleMatrix_k2s= peopleMatrix_k2/peopleMatrix_k2.sum(axis=1,keepdims=1)
peopleMatrix_k2s
peopleMatrix_k2s.sum(1)

results_k2= np.matmul(restaurantsMatrix_k2, newPeopleMatrix_k2)
results_k2 

M_usr_x_rest_k2 = np.sum(results_k2, axis=1)
M_usr_x_rest_k2

resturants_names = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']

# from scipy.stats import rankdata
ranking = rankdata([-1 * i for i in M_usr_x_rest_k2]).astype(int) # rank 1 as the the highest

# using zip() to map values 
mapped_k2 = zip(resturants_names, M_usr_x_rest_k2, ranking) 
# converting values to print as set 
mapped_k2 = set(mapped_k2) 
  
for rn, muxr, rnk in mapped_k2:
    print ("Resturant :  %s  Scores : %d Ranking: %d" %(rn, muxr, rnk)) 
    
    
M_usr_x_rest_rank_all_k2 = np.reshape(rankdata(-results_k2, method='max'), (10, 10)) # -results to rank 1 is the best
M_usr_x_rest_rank_all_k2

M_usr_x_rest_rank_k2 = M_usr_x_rest_rank_all_k2.argsort(axis=0).argsort(axis=0) + 1
M_usr_x_rest_rank_k2

peop_name = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabina', 'Georg', 'Alan', 'Jackson']
rest_name = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']
rank_table_k2 = np.zeros((11,11), object)
rank_table_k2[1:,1:]= M_usr_x_rest_rank_k2
rank_table_k2[0,1:]=peop_name
rank_table_k2[1:,0]=rest_name
rank_table_k2[0,0]=''
printer = np.vectorize(lambda peop_name:'{0:10}'.format(peop_name,))

np.set_printoptions(linewidth=np.inf)
print(printer(rank_table_k2).astype(object))     

"""
## Sum
Resturant :  Bravo  Scores : 18 Ranking: 7
Resturant :  Fukuyu  Scores : 19 Ranking: 6
Resturant :  Moretti  Scores : 31 Ranking: 1
Resturant :  Gogi  Scores : 21 Ranking: 4
Resturant :  Bonsai  Scores : 21 Ranking: 5
Resturant :  Fox  Scores : 26 Ranking: 2
Resturant :  Morchi  Scores : 16 Ranking: 8
Resturant :  La Tavola  Scores : 10 Ranking: 10
Resturant :  DaVinci  Scores : 24 Ranking: 3
Resturant :  Momo Ghar  Scores : 13 Ranking: 9

## Individual
[['          ' 'Frank     ' 'Jens      ' 'Marina    ' 'Wolfgang  ' 'Natalia   ' 'Roland    ' 'Sabina    ' 'Georg     ' 'Alan      ' 'Jackson   ']
 ['Fukuyu    ' '         5' '         6' '         5' '         6' '         6' '         6' '         6' '         6' '         6' '         5']
 ['Moretti   ' '         1' '         2' '         1' '         1' '         1' '         1' '         1' '         2' '         1' '         1']
 ['La Tavola ' '        10' '         9' '        10' '         9' '         9' '         9' '        10' '         9' '        10' '        10']
 ['Bonsai    ' '         7' '         4' '         8' '         4' '         4' '         4' '         5' '         4' '         4' '         6']
 ['DaVinci   ' '         3' '         5' '         3' '         5' '         5' '         5' '         4' '         5' '         5' '         3']
 ['Fox       ' '         4' '         3' '         4' '         3' '         3' '         3' '         2' '         3' '         2' '         2']
 ['Morchi    ' '         6' '         7' '         6' '         7' '         7' '         7' '         7' '         7' '         7' '         7']
 ['Momo Ghar ' '         9' '         8' '         7' '         8' '         8' '         8' '         9' '         8' '         8' '         9']
 ['Gogi      ' '         8' '         1' '         9' '         2' '         2' '         2' '         3' '         1' '         3' '         8']
 ['Bravo     ' '         2' '        10' '         2' '        10' '        10' '        10' '         8' '        10' '         9' '         4']]
"""

## We also can find each person's ranking to each resturant 
M_rest_x_usr_rank = M_usr_x_rest_rank_all.argsort(axis=1).argsort(axis=1) + 1
M_rest_x_usr_rank

peop_name = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabina', 'Georg', 'Alan', 'Jackson']
rest_name = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']
rank_table = np.zeros((11,11), object)
rank_table[1:,1:]= M_rest_x_usr_rank
rank_table[0,1:]=peop_name
rank_table[1:,0]=rest_name
rank_table[0,0]=''
printer = np.vectorize(lambda peop_name:'{0:10}'.format(peop_name,))

np.set_printoptions(linewidth=np.inf)
print(printer(rank_table).astype(object))

"""
## each resturant's guest preference ranking 

[['          ' 'Frank     ' 'Jens      ' 'Marina    ' 'Wolfgang  ' 'Natalia   ' 'Roland    ' 'Sabina    ' 'Georg     ' 'Alan      ' 'Jackson   ']
 ['Fukuyu    ' '         2' '         6' '         3' '         8' '         1' '         7' '         4' '         9' '        10' '         5']
 ['Moretti   ' '         2' '         9' '         1' '         5' '         4' '         7' '         3' '         8' '        10' '         6']
 ['La Tavola ' '         7' '         1' '        10' '         9' '         2' '         8' '         4' '         5' '         6' '         3']
 ['Bonsai    ' '         6' '         2' '         9' '        10' '         1' '         8' '         4' '         3' '         7' '         5']
 ['DaVinci   ' '         3' '         8' '         7' '         6' '         1' '         4' '         2' '        10' '         9' '         5']
 ['Fox       ' '         6' '         5' '         3' '         9' '        10' '         8' '         7' '         1' '         4' '         2']
 ['Morchi    ' '         5' '         4' '         9' '        10' '         1' '         6' '         3' '         8' '         7' '         2']
 ['Momo Ghar ' '         9' '         7' '         6' '         4' '        10' '         2' '         8' '         5' '         1' '         3']
 ['Gogi      ' '         6' '         5' '        10' '         2' '         1' '         4' '         3' '         7' '         8' '         9']
 ['Bravo     ' '         3' '         9' '         1' '         2' '         8' '         4' '         5' '        10' '         7' '         6']]

"""

## -----------------------------------
########### Clustering stuff #########
## -----------------------------------    
## ------------------------------------------------------------
## Q12 Find user profiles that are problematic, explain why?
## ------------------------------------------------------------

M_usr_x_rest
M_usr_x_rest.shape

print(peopleKeys)
print(restaurantsKeys)

## a_ij is the score for a restaurant for a person, and x is the person, y is the restaurant
## x=0 (Frank), x=1 (Jens), x=2 (Marina), x=3 (Wolfgang), x=4 (Natalia), x=5 (Roland), x=6 (Sabina), x=7 (Georg), x=8 (Alan), x=9 (Jackson)
## y=0 (fFukuyu), y=1 (Moretti), y=2 (La Tavola), y=3 (Bonsai), y=4 (DaVinci),y=5 (Fox), y=6 (Mirchi), y=7 (Momo Ghar), y=8 (Gogi), y=9 (Bravo)

## Plot heatmap for matrix M_usr_x_rest (seaborn.pydata.org/generated/seaborn.heatmap.html)
plot_dims = (12,10)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data= M_usr_x_rest, annot=True) 
plt.show()

print(peopleKeys)
print(restaurantsKeys)
 
"""
From the heatmap and also the ranking results in previous questions, we can find there are many clusters in user profiles. 
For example, some users are “similar” to one another and thus can be treated collectively as one group, but as a collection, they are sufficiently different from other groups.

By using clustering, we can assign users to automatically created groups based on similarity or association between users and resturants.

"""

## --------------------------------------------------------------------------------------------------------
## Q13 Think of two metrics to compute the disatistifaction with the group.
## --------------------------------------------------------------------------------------------------------

## Need to apply standardScaler() if the data is not scaled for peopleMatrix
## sc = StandardScaler() 
## peopleMatrixScaled = sc.fit_transform(peopleMatrix) 

## Already sumed to 1 in previous daa processing step
## peopleMatrix.sum(1) # check if sum to 1

## K-means is the most frequently used form of clustering due to its speed and simplicity
## hierarchical clustering is another very common clustering method

## Q13-1-A1 principal component analysis (PCA) for dimensionality reduction and kmeans for clustering
## -----------------------------------------------------------------------------------------------

"""
PCA is an unsupervised, linear and parametric algorithm that creates linear combinations of the original features for reducing the number of dimensions
in a dense dataset while retaining most information. 

Dimensionality reduction can be achieved by limiting the number of principal components to keep based on cumulative explained variance. 

PCA produces rotational transformation of the original data for preserving distances which is important to apply distance-based (e.g. k-means) clustering algorithms.

K-means clustering is the most frequently used form of clustering due to its speed and simplicity
"""

## Q13-1-A2: PCA and kmeans clustering for people with two feature variables
## ----------------------------------------------------------------------
pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

print(pca.components_)
print(pca.explained_variance_)

## The draw_vector function was adoped and modified from (jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

## plot principal components
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 1], ax=ax)
draw_vector([0, 0], [1, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
fig.show

## Use people PCA for clustering and plotting (scikit-learn.org/stable/modules/clustering.html) 
kmeans = KMeans(n_clusters=2)
kmeans.fit(peopleMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)

## plot (color selection: matplotlib.org/users/colors.html
##      annotation: matplotlib.org/users/annotations_intro.html and matplotlib.org/users/text_intro.html)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

colors = ["g.","r.","c."]
labelList = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabina', 'Georg', 'Alan', 'Jackson']

for i in range(len(peopleMatrixPcaTransform)):
   print ("coordinate:" , peopleMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1],colors[labels[i]],markersize=15)
   ax.annotate(labelList[i], (peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1]), size=15)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =15)

plt.show()

## cluster 0 is green (), cluster 1 is red
## The people labels: x=0 (Frank), x=1 (Jens), x=2 (Marina), x=3 (Wolfgang), x=4 (Natalia), x=5 (Roland), x=6 (Sabina), x=7 (Georg), x=8 (Alan), x=9 (Jackson)

## Q13-1-A3: Clustering performance evaluation for people matrices with the Calinski-Harabaz Index
## --------------------------------------------------------------------------------------------
## https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

print("\nThe Calinski-Harabaz Index is used to measure better defined clusters.")
print("The Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

range_n_clusters = [2, 3, 4, 5]
for n_clusters in range_n_clusters:
     kmCH_cluster = KMeans(n_clusters=n_clusters, random_state=1)
     kmCH_cluster_labels = kmCH_cluster.fit_predict(peopleMatrixPcaTransform)
     kmCH_score = metrics.calinski_harabaz_score(peopleMatrixPcaTransform, kmCH_cluster_labels)  
     print("The Calinski-Harabaz score for :", n_clusters, " clusters score is: ", kmCH_score)   
     
"""
The Calinski-Harabaz Index is used to measure better defined clusters.
The Calinski-Harabaz score is higher when clusters are dense and well separated.

The Calinski-Harabaz score for : 2  clusters score is:  5.483249495153372
The Calinski-Harabaz score for : 3  clusters score is:  12.201291734118119
The Calinski-Harabaz score for : 4  clusters score is:  13.297246454831745
The Calinski-Harabaz score for : 5  clusters score is:  16.90243080478937
"""    

## Q13-1-A4: Clustering performance evaluation for people matrices with the Davies-Bouldin score
## ------------------------------------------------------------------------------------------      
print("\nThe Davies-Bouldin Index is used to measure better defined clusters.")
print("The Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned).")
print("Zero is the lowest possible Davies-Bouldin score.\n")

range_n_clusters = [2, 3, 4, 5]
for n_clusters in range_n_clusters:
     kmDB_cluster = KMeans(n_clusters=n_clusters, random_state=1)
     kmDB_cluster_labels = kmDB_cluster.fit_predict(peopleMatrixPcaTransform)
     kmDB_score = metrics.davies_bouldin_score(peopleMatrixPcaTransform, kmDB_cluster_labels)  
     print("The Davies-Bouldin score for :", n_clusters, " clusters score is: ", kmDB_score)

"""
The Davies-Bouldin Index is used to measure better defined clusters.
The Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned).
Zero is the lowest possible Davies-Bouldin score.

The Davies-Bouldin score for : 2  clusters score is:  0.3421153169321699
The Davies-Bouldin score for : 3  clusters score is:  0.5393500698266132
The Davies-Bouldin score for : 4  clusters score is:  0.45521701126305936
The Davies-Bouldin score for : 5  clusters score is:  0.40781509687787815
"""     

## Q13-1-A5: Clustering performance evaluation for people matrices with the Silhouette Analysis
## ----------------------------------------------------------------------------------------- 
## Silhouette Analysis with Kmeans Clustering on the PCA transformed People Matrix
## https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

range_n_clusters = [2, 3, 4, 5]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2) # Create a subplot with 1 row and 2 columns
    fig.set_size_inches(12, 6)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(peopleMatrixPcaTransform) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator seed of 1 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = metrics.silhouette_score(peopleMatrixPcaTransform, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(peopleMatrixPcaTransform, cluster_labels)
    
    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

    print("\n\n\nFor n_clusters =", n_clusters,
          "\n\nThe average silhouette_score is :", silhouette_avg,
          "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
          "\n* Scores around zero indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
          "\n\nThe individual silhouette scores were :", sample_silhouette_values,
          "\n\nAnd their assigned clusters were :", cluster_labels,
          "\n\nWhich correspond to : 'Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabine', 'Georg', 'Alan', 'Jackson'")
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.9)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontsize=12)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=12)
    ax1.set_ylabel("Cluster label", fontsize=12)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(peopleMatrixPcaTransform[:, 0], peopleMatrixPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=400, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontsize=12)
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=12)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=12)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontsize=12, fontweight='bold')
        
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

plt.show()


## Q13-1-B1 PCA and kmeans for resturant
## -------------------------------------
## Do the same for restaurants (jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)

restaurantsMatrix.shape

pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  

print(pca.components_)
print(pca.explained_variance_)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

ax.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax)
draw_vector([0, 0], [3, 0], ax=ax)
ax.axis('equal')
ax.set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-10, 10), ylim=(-10, 10))
fig.show

## Use restaurantsMatrixPcaTransform for plotting (scikit-learn.org/stable/modules/clustering.html)
kmeans = KMeans(n_clusters=2)
kmeans.fit(restaurantsMatrixPcaTransform)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

print (centroid)
print(labels)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

## plot (color selection: matplotlib.org/users/colors.html
##      annotation: matplotlib.org/users/annotations_intro.html and matplotlib.org/users/text_intro.html)
colors = ["g.","r.","c."]
labelList = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']

for i in range(len(restaurantsMatrixPcaTransform)):
   print ("coordinate:" , restaurantsMatrixPcaTransform[i], "label:", labels[i])
   ax.plot(restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1],colors[labels[i]],markersize=15)
   ax.annotate(labelList[i], (restaurantsMatrixPcaTransform[i][0],restaurantsMatrixPcaTransform[i][1]), size=15)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =15)

plt.show()

## cluster 0 is green, cluster 1 is red
## The resturant labels: y=0 (Fukuyu), y=1 (Moretti), y=2 (La Tavola), y=3 (Bonsai), y=4 (DaVinci),y=5 (Fox), y=6 (Mirchi), y=7 (Momo Ghar), y=8 (Gogi), y=9 (Bravo)

## Q13-1-B2: Clustering performance evaluation for resturant matrices with the Calinski-Harabaz Index
## -----------------------------------------------------------------------------------------------
print("\nThe Calinski-Harabaz Index is used to measure better defined clusters.")
print("The Calinski-Harabaz score is higher when clusters are dense and well separated.\n")

range_n_clusters = [2, 3, 4, 5]
for n_clusters in range_n_clusters:
     kmCHr_cluster = KMeans(n_clusters=n_clusters, random_state=2)
     kmCHr_cluster_labels = kmCHr_cluster.fit_predict(restaurantsMatrixPcaTransform)
     kmCHr_score = metrics.calinski_harabaz_score(restaurantsMatrixPcaTransform, kmCHr_cluster_labels)  
     print("The Calinski-Harabaz score for :", n_clusters, " clusters score is: ", kmCHr_score)

"""
The Calinski-Harabaz Index is used to measure better defined clusters.
The Calinski-Harabaz score is higher when clusters are dense and well separated.

The Calinski-Harabaz score for : 2  clusters score is:  6.430333561474328
The Calinski-Harabaz score for : 3  clusters score is:  10.317853646619374
The Calinski-Harabaz score for : 4  clusters score is:  14.104291625286885
The Calinski-Harabaz score for : 5  clusters score is:  16.962577564176236
"""     
     
## Q14-7: Clustering performance evaluation for resturant matrices with the Davies-Bouldin score
## ---------------------------------------------------------------------------------------------     
print("\nThe Davies-Bouldin Index is used to measure better defined clusters.")
print("The Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned.")
print("Zero is the lowest possible Davies-Bouldin score.\n")

range_n_clusters = [2, 3, 4, 5]
for n_clusters in range_n_clusters:
     kmDBr_cluster = KMeans(n_clusters=n_clusters, random_state=10)
     kmDBr_cluster_labels = kmDBr_cluster.fit_predict(restaurantsMatrixPcaTransform)
     kmDBr_score = metrics.davies_bouldin_score(restaurantsMatrixPcaTransform, kmDBr_cluster_labels)  
     print("The Davies-Bouldin score for :", n_clusters, " clusters score is: ", kmDBr_score)

"""
The Davies-Bouldin Index is used to measure better defined clusters.
The Davies-Bouldin score is lower when clusters more separated (e.g. better partitioned.
Zero is the lowest possible Davies-Bouldin score.

The Davies-Bouldin score for : 2  clusters score is:  1.0313500237759128
The Davies-Bouldin score for : 3  clusters score is:  0.6685460605921857
The Davies-Bouldin score for : 4  clusters score is:  0.5922950551578674
The Davies-Bouldin score for : 5  clusters score is:  0.4034702235371019
"""

## Q13-1-B3: Clustering performance evaluation for resturant matrices with the Silhouette Analysis
## --------------------------------------------------------------------------------------------     
## Silhouette Analysis with Kmeans Clustering on the PCA transformed Restaurant Matrix

range_n_clusters = [2, 3, 4, 5]

for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2) # Create a subplot with 1 row and 2 columns
    fig.set_size_inches(12, 6)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(restaurantsMatrixPcaTransform) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator seed of 2 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(restaurantsMatrixPcaTransform)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = metrics.silhouette_score(restaurantsMatrixPcaTransform, cluster_labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(restaurantsMatrixPcaTransform, cluster_labels)
    
    # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    # Scores around zero indicate overlapping clusters.
    # The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

    print("\n\n\nFor n_clusters =", n_clusters,
          "\n\nThe average silhouette_score is :", silhouette_avg,
          "\n\n* The silhouette score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.",
          "\n* Scores around zero indicate overlapping clusters.",
          "\n* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster",
          "\n\nThe individual silhouette scores were :", sample_silhouette_values,
          "\n\nAnd their assigned clusters were :", cluster_labels,
          "\n\nWhich correspond to : 'Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo'")
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.9)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.", fontsize=12)
    ax1.set_xlabel("The silhouette coefficient values", fontsize=12)
    ax1.set_ylabel("Cluster label", fontsize=12)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)


    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(restaurantsMatrixPcaTransform[:, 0], restaurantsMatrixPcaTransform[:, 1], marker='.', s=300, lw=0, alpha=0.7, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=400, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.", fontsize=12)
    ax2.set_xlabel("Feature space for the 1st feature", fontsize=12)
    ax2.set_ylabel("Feature space for the 2nd feature", fontsize=12)

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontsize=12, fontweight='bold')
        
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

plt.show()


## Q13-2 Perform hierarchical clustering
## ----------------------------------------------------
## Ref: docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
## Use "complete" linkage here
## linkage options: "complete", "average", "single", "weighted", "centroid", "median", or "ward"

"""
Hierarchical clustering, as depicted by a tree or dendrogram, can be divided into agglomerative ("from the bottom up") and divisive clustering ("from the top down").

There are four linkages (complete, average, ward and single) linkage for hierarchical agglomerative clustering
with the basic idea to repeatedly merge two most similar groups as measured by the linkage. 

Complete linkage measures the farthest pair of points, 

average linkage measures the average dissimilarity over all pairs, 

ward linkage measures how much the sum of squares will increase when we merge two clusters, 

and single linkage measures the closest pair of points.

In this homework, use the hierarchical clustering with linkage 'complete'.

"""

## Q13-2A: PCA and hierarchical clustering for people
## ------------------------------------------
pca = PCA(n_components=2)  
peopleMatrixPcaTransform = pca.fit_transform(peopleMatrix)  

## Use "complete option" in heirarchical clustering
linked = linkage(peopleMatrixPcaTransform, 'complete') ## this option fits well with PCA plot as shown above

labelList = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabine', 'Georg', 'Alan', 'Jackson']

## explicit interface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', rotation = 60, labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()  

## Q13-2B: PCA and hierarchical clustering for restaurants
## -----------------------------------------------

pca = PCA(n_components=2)  
restaurantsMatrixPcaTransform = pca.fit_transform(restaurantsMatrix)  

## Use "complete option" in heirarchical clustering
linked = linkage(restaurantsMatrixPcaTransform, 'complete')

labelList = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1)
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True, ax=ax)
ax.tick_params(axis='x', which='major', rotation = 60, labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()  

## -----------------------------------------
## Q14 Should you split in two groups today?
## -----------------------------------------

"""
As for splitting into two groups or not, depending on how many feature variables will be considered.

If considering all 6 feature vaibles ('willingness to travel', 'desire for new experience', 'cost_p', 'Asian food', 'Italian food', 'vegetarian'),
we will get 3 groups from kmeans and hierarchical clustering results:

Group A: Wolfgang, Alan, Roland, Georg    
Group B: Natalia, Jens, Sabine,, Jackson, Frank
Group C: Marina

Since there is only Marina is one group, we can let Marina decide either group she would like to join. 
Dividing into two groups will satisfy all the people except Marina.

However, we do not have to divide into two groups if we can remove some features which are not to be considered. 
For example, if we only consider 'distance' and 'novelty' (i.e. 'willingness to travel', 'desire for new experience').
We can participate in one group for dining. All people will satisfy except Marina. Marina can choose to go together or skip today.
"""

## ----------------------------------------------------------
############# Did you understand what's going on? ###########
## ----------------------------------------------------------

## -------------------------------------------------------------------------------------------------------------------
## Q15 Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
## -------------------------------------------------------------------------------------------------------------------

## Since the boss the pay for the meal, cost will not be considered and should be removed from feature variable.
## Only need to consider ('willingness to travel', 'desire for new experience','Asian food', 'Italian food', 'vegetarian')

peopleMatrix_dc = np.delete(peopleMatrix, 2, 1) # remove cost feature from resturant
peopleMatrix_dc 

restaurantsMatrix_dc = np.delete(restaurantsMatrix, 2, 1) #remove cost feature form resturant
restaurantsMatrix_dc

newPeopleMatrix_dc = np.swapaxes(peopleMatrix_dc, 0, 1)
newPeopleMatrix_dc

## https://stackoverflow.com/questions/43644320/how-to-make-numpy-array-column-sum-up-to-1
peopleMatrix_dcs= peopleMatrix_dc/peopleMatrix_dc.sum(axis=1,keepdims=1)
peopleMatrix_dcs
peopleMatrix_dcs.sum(1)

results_dc = np.matmul(restaurantsMatrix_dc, newPeopleMatrix_dc)
results_dc 

M_usr_x_rest_dc = np.sum(results_dc, axis=1)
M_usr_x_rest_dc

resturants_names = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']

# from scipy.stats import rankdata
ranking = rankdata([-1 * i for i in M_usr_x_rest_dc]).astype(int) # rank 1 as the the highest

# using zip() to map values 
mapped_dc = zip(resturants_names, M_usr_x_rest_dc, ranking) 
# converting values to print as set 
mapped_dc = set(mapped_dc) 
  
for rn, muxr, rnk in mapped_dc:
    print ("Resturant :  %s  Scores : %d Ranking: %d" %(rn, muxr, rnk)) 
    
    
M_usr_x_rest_rank_all_dc = np.reshape(rankdata(-results_dc, method='max'), (10, 10)) # -results to rank 1 is the best
M_usr_x_rest_rank_all_dc

M_usr_x_rest_rank_dc = M_usr_x_rest_rank_all_dc.argsort(axis=0).argsort(axis=0) + 1
M_usr_x_rest_rank_dc

peop_name = ['Frank', 'Jens', 'Marina', 'Wolfgang', 'Natalia', 'Roland', 'Sabina', 'Georg', 'Alan', 'Jackson']
rest_name = ['Fukuyu', 'Moretti', 'La Tavola', 'Bonsai', 'DaVinci', 'Fox', 'Morchi', 'Momo Ghar', 'Gogi', 'Bravo']
rank_table_dc = np.zeros((11,11), object)
rank_table_dc[1:,1:]= M_usr_x_rest_rank_dc
rank_table_dc[0,1:]=peop_name
rank_table_dc[1:,0]=rest_name
rank_table_dc[0,0]=''
printer = np.vectorize(lambda peop_name:'{0:10}'.format(peop_name,))

np.set_printoptions(linewidth=np.inf)
print(printer(rank_table_dc).astype(object))     

"""
From the results below, we can find Fox resturant will be the best option if the boss is to take care of the dining cost.

## Sum result: Fox
Resturant :  Fox  Scores : 52 Ranking: 1
Resturant :  Morchi  Scores : 38 Ranking: 7
Resturant :  Fukuyu  Scores : 32 Ranking: 10
Resturant :  Moretti  Scores : 47 Ranking: 3
Resturant :  Bonsai  Scores : 37 Ranking: 8
Resturant :  Gogi  Scores : 43 Ranking: 4
Resturant :  La Tavola  Scores : 35 Ranking: 9
Resturant :  Bravo  Scores : 41 Ranking: 5
Resturant :  DaVinci  Scores : 50 Ranking: 2
Resturant :  Momo Ghar  Scores : 41 Ranking: 6

## Individual: 5 votes for 'Fox'
[['          ' 'Frank     ' 'Jens      ' 'Marina    ' 'Wolfgang  ' 'Natalia   ' 'Roland    ' 'Sabina    ' 'Georg     ' 'Alan      ' 'Jackson   ']
 ['Fukuyu    ' '         5' '         9' '         5' '         8' '         7' '         9' '         9' '         9' '        10' '         9']
 ['Moretti   ' '         1' '         8' '         2' '         4' '         2' '         6' '         2' '         6' '         9' '         4']
 ['La Tavola ' '        10' '         3' '         9' '         9' '         8' '         8' '        10' '         4' '         4' '         7']
 ['Bonsai    ' '         8' '         2' '         8' '        10' '         4' '        10' '         7' '         2' '         8' '         8']
 ['DaVinci   ' '         2' '         5' '         3' '         2' '         5' '         2' '         3' '         5' '         3' '         3']
 ['Fox       ' '         3' '         1' '         4' '         6' '         3' '         5' '         1' '         1' '         1' '         1']
 ['Morchi    ' '         6' '         7' '         6' '         7' '         6' '         7' '         8' '         8' '         6' '         5']
 ['Momo Ghar ' '         7' '         6' '         7' '         5' '         9' '         4' '         6' '         7' '         2' '         2']
 ['Gogi      ' '         9' '         4' '        10' '         1' '         1' '         1' '         4' '         3' '         5' '        10']
 ['Bravo     ' '         4' '        10' '         1' '         3' '        10' '         3' '         5' '        10' '         7' '         6']]
"""
## ----------------------------------------------------------------------------------------------------------------------------
## Q16 Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.
##     Can you find their weight matrix??
## ----------------------------------------------------------------------------------------------------------------------------

"""
Only based on their opitimal ordering matrix, it is possible to know their weight matrix, but very unlikely in this homework.

(1) For calculation, first need to compute if the determinant is the same for both ordering matrices (A and B). If not, then impossible.

(2) If yes, compute the eigenvalues for both matrices. If they are similar, it is possible to calculate with diagonal matrix. 
 
(3) Suppose we could get the same optimal ordering matrix, we can calculate with a transformation matrix (M_key) as shown below. 
"""

## Q16_(1) Calculate the key for conversion between weight (M_usr_x_rest) and ranking (M_usr_x_rest_rank) from Q11
np.linalg.det(M_usr_x_rest_rank) # M_usr_x_rest_rank from Q11 and its determinant is 3097710.000000002
np.linalg.det(results_dc) # results_dc from Q15 and its determinant is 9.72745461779481e-75, not equal.


## Q16_(2) Check eigenvalues, eigenvectors, diagnal matrix for M_usr_x_rest_rank in Q11
eigenvalues, eigenvectors = np.linalg.eig(M_usr_x_rest_rank)
inverseEigenVectors = np.linalg.inv(eigenvectors)
diagonal_M_usr_x_rest_rank= inverseEigenVectors.dot(M_usr_x_rest_rank).dot(eigenvectors) #M^(-1).A.M
print(diagonal_M_usr_x_rest_rank.round(3))

## Q16_(2) Check eigenvalues, eigenvectors, diagnal matrix for M_usr_x_rest_rank_dc in Q15
eigenvalues, eigenvectors = np.linalg.eig(M_usr_x_rest_rank_dc)
inverseEigenVectors = np.linalg.inv(eigenvectors)
diagonal_M_usr_x_rest_rank_dc= inverseEigenVectors.dot(M_usr_x_rest_rank_dc).dot(eigenvectors) #M^(-1).A.M
print(diagonal_M_usr_x_rest_rank_dc.round(3))

"""
## M_usr_x_rest_rank and M_usr_x_rest_rank_dc do not share same diagonal matrix. It is impossible to find the transformation matrix

diagonal_M_usr_x_rest_rank in Q11
[[55.   +0.j    -0.   +0.j    -0.   -0.j     0.   -0.j     0.   +0.j    -0.   -0.j    -0.   -0.j    -0.   +0.j     0.   -0.j     0.   -0.j   ]
 [-0.   +0.j    11.751+0.j    -0.   -0.j    -0.   +0.j    -0.   -0.j     0.   -0.j     0.   +0.j     0.   -0.j    -0.   +0.j     0.   +0.j   ]
 [-0.   -0.j     0.   -0.j    -7.05 -0.j    -0.   +0.j    -0.   -0.j    -0.   -0.j    -0.   +0.j    -0.   -0.j     0.   +0.j     0.   +0.j   ]
 [ 0.   -0.j    -0.   +0.j    -0.   +0.j    -0.301+3.788j  0.   -0.j     0.   +0.j     0.   -0.j     0.   +0.j    -0.   -0.j    -0.   -0.j   ]
 [ 0.   +0.j    -0.   -0.j    -0.   +0.j     0.   +0.j    -0.301-3.788j  0.   -0.j     0.   -0.j     0.   +0.j    -0.   +0.j    -0.   +0.j   ]
 [ 0.   +0.j    -0.   -0.j     0.   -0.j     0.   +0.j     0.   -0.j    -1.756-0.j     0.   -0.j     0.   +0.j    -0.   +0.j    -0.   -0.j   ]
 [ 0.   -0.j     0.   -0.j     0.   -0.j    -0.   -0.j    -0.   -0.j     0.   -0.j     1.588+3.085j  0.   -0.j    -0.   -0.j     0.   -0.j   ]
 [-0.   +0.j     0.   +0.j     0.   +0.j    -0.   +0.j    -0.   +0.j    -0.   +0.j     0.   +0.j     1.588-3.085j  0.   +0.j     0.   +0.j   ]
 [ 0.   -0.j     0.   -0.j     0.   +0.j     0.   +0.j    -0.   -0.j    -0.   +0.j     0.   +0.j     0.   -0.j     3.911-0.j    -0.   -0.j   ]
 [-0.   -0.j    -0.   -0.j    -0.   +0.j    -0.   +0.j    -0.   -0.j    -0.   +0.j    -0.   +0.j    -0.   -0.j     0.   -0.j     0.569-0.j   ]]

diagonal_M_usr_x_rest_rank_dc in Q15
[[55.   -0.j     0.   +0.j     0.   -0.j     0.   -0.j     0.   +0.j    -0.   +0.j     0.   -0.j     0.   +0.j     0.   +0.j     0.   -0.j   ]
 [-0.   +0.j    -1.923+1.382j -0.   +0.j     0.   -0.j     0.   -0.j    -0.   +0.j     0.   -0.j     0.   -0.j    -0.   -0.j     0.   +0.j   ]
 [-0.   -0.j    -0.   +0.j    -1.923-1.382j  0.   -0.j    -0.   +0.j    -0.   -0.j     0.   +0.j     0.   +0.j    -0.   +0.j     0.   -0.j   ]
 [-0.   +0.j     0.   +0.j    -0.   -0.j     0.316+3.395j -0.   +0.j    -0.   +0.j     0.   -0.j    -0.   -0.j    -0.   -0.j     0.   -0.j   ]
 [-0.   -0.j     0.   +0.j     0.   -0.j    -0.   -0.j     0.316-3.395j -0.   -0.j     0.   +0.j     0.   +0.j    -0.   +0.j     0.   +0.j   ]
 [ 0.   -0.j    -0.   -0.j    -0.   +0.j     0.   +0.j     0.   -0.j     6.157-0.j    -0.   +0.j     0.   +0.j     0.   +0.j    -0.   -0.j   ]
 [ 0.   -0.j    -0.   -0.j    -0.   +0.j     0.   +0.j     0.   -0.j     0.   -0.j     5.331+0.j     0.   +0.j     0.   +0.j    -0.   -0.j   ]
 [ 0.   -0.j     0.   +0.j     0.   -0.j    -0.   -0.j    -0.   +0.j    -0.   -0.j     0.   -0.j     0.255-0.j    -0.   -0.j     0.   -0.j   ]
 [-0.   +0.j    -0.   -0.j    -0.   +0.j     0.   +0.j     0.   -0.j     0.   +0.j     0.   -0.j     0.   -0.j     1.262+0.j    -0.   +0.j   ]
 [-0.   +0.j     0.   +0.j     0.   -0.j    -0.   -0.j    -0.   +0.j    -0.   +0.j     0.   -0.j    -0.   -0.j    -0.   -0.j     3.208-0.j   ]]

"""

## Q16_(3) Suppose we could get the same optimal ordering matrix, we can calculate with a transformation matrix (M_key) as shown below. 

M_usr_x_rest_rank # the resturant ranking matrix
Muxr_rank_inv = inv(np.matrix(M_usr_x_rest_rank)) # inversion matrix of the resturant ranking matrix
M_usr_x_rest # our weight matrix
M_key = np.matmul(Muxr_rank_inv, M_usr_x_rest) # get the key M_key for conversion between ranking and weight matrix
Muxr_check = np.matmul(M_usr_x_rest_rank, M_key) # confirm the M_key is working for conversion

"""
M_usr_x_rest
array([[6.03900067, 4.90957755, 5.76412722, 4.35366105, 6.59393117, 4.67913593, 5.76133149, 3.82347511, 3.76180471, 5.20751874],
       [6.53510495, 4.89832208, 7.8136293 , 6.09998422, 6.18921502, 5.54355032, 6.27867782, 5.1499884 , 4.67559251, 5.70760688],
       [5.57117358, 6.97982763, 3.58891045, 4.23374722, 6.97800309, 5.39992029, 5.91402437, 5.90736875, 5.80095811, 6.20126818],
       [5.23630512, 6.69635782, 3.85914087, 3.84679702, 7.20646408, 4.32930211, 5.84272342, 6.54875216, 4.93025016, 5.54778374],
       [7.93867462, 7.01659364, 7.25178018, 7.38646393, 8.56599905, 7.67456647, 8.04805724, 5.90735227, 6.34836679, 7.40242864],
       [5.28932714, 5.80581862, 6.55346771, 4.68905489, 4.4825006 , 5.10903806, 5.24832466, 7.09294862, 6.212836  , 6.62298012],
       [6.34400303, 6.49004719, 5.10577634, 4.65357916, 7.16779059, 5.96249348, 6.49908437, 5.24801339, 5.49566281, 6.89337132],
       [3.95029394, 4.34141796, 4.43348542, 5.08633598, 2.90687435, 5.41287913, 4.11329326, 4.97737993, 5.68631712, 5.20340264],
       [6.21954794, 7.05024507, 3.35989683, 8.31755973, 9.23872877, 7.68452841, 7.89334537, 6.14080781, 5.85733053, 5.4796083 ],
       [6.54997811, 3.89847258, 7.9287086 , 6.98838735, 4.84643443, 6.5164061 , 5.70435032, 3.26627605, 4.96722018, 5.37829754]])

M_usr_x_rest_rank
array([[ 6,  7,  5,  8,  6,  9,  7,  9, 10,  9],
       [ 3,  8,  2,  4,  7,  5,  4,  7,  9,  5],
       [ 7,  3,  9,  9,  5,  7,  5,  4,  4,  4],
       [ 9,  4,  8, 10,  3, 10,  6,  2,  8,  6],
       [ 1,  2,  3,  2,  2,  2,  1,  5,  1,  1],
       [ 8,  6,  4,  6,  9,  8,  9,  1,  2,  3],
       [ 4,  5,  6,  7,  4,  4,  3,  6,  6,  2],
       [10,  9,  7,  5, 10,  6, 10,  8,  5, 10],
       [ 5,  1, 10,  1,  1,  1,  2,  3,  3,  7],
       [ 2, 10,  1,  3,  8,  3,  8, 10,  7,  8]])

Muxr_rank_inv
matrix([[ 0.22686049, -0.11758751, -0.29609679, -0.11254507, -0.13907112,  0.00747907,  0.32948468,  0.27202934, -0.05157423, -0.31196981],
        [-0.93737567,  0.35483696,  0.21730633,  0.74622931,  0.57460899, -0.37414994, -0.54917923,  0.11791356, -0.21137163,  0.40778252],
        [-0.10743097,  0.02189682, -0.00324433,  0.04309635,  0.12171572,  0.09700069, -0.01400389, -0.15696757,  0.15154421,  0.09199376],
        [-0.3060577 ,  0.08895668,  0.46191929,  0.16853482, -0.07403469, -0.31756297, -0.2216857 ,  0.11100845, -0.22839711,  0.176205  ],
        [ 0.12536616,  0.16741012,  0.27291063, -0.312793  , -0.24440054,  0.04575251, -0.05623251, -0.00715755,  0.01461015, -0.12391476],
        [-0.12784605,  0.14487153, -0.00863864,  0.21922969,  0.4196132 ,  0.03141676, -0.33281037, -0.09556737, -0.00842558,  0.03898041],
        [ 0.63493097, -0.52142002, -0.50750006, -0.39191467, -0.33374912,  0.49349423,  0.622751  , -0.24724651,  0.26555488, -0.06305561],
        [ 0.36064835, -0.20451269, -0.23879769, -0.23139384,  0.01564995,  0.10059011,  0.29243344,  0.04238906,  0.02807848, -0.17530563],
        [ 0.55237256, -0.17079068, -0.36464033, -0.39010107, -0.41950344,  0.30518673,  0.52556243, -0.14454872,  0.21161761, -0.2376562 ],
        [-0.4032863 ,  0.2545206 ,  0.48496341,  0.2798393 ,  0.09735288, -0.37102537, -0.57813804,  0.12632913, -0.15345497,  0.21512214]])

M_key
matrix([[-1.90116029, -1.47537489, -1.82611258, -2.10054126, -2.10283565, -1.69781499, -1.84636181, -1.24540857, -1.29944334, -1.20397926],
        [ 2.6969816 ,  2.55653758,  2.98585099,  3.49976065,  2.58402756,  2.88012543,  2.70692304,  2.8932829 ,  2.83568941,  2.26967395],
        [ 2.01742789,  1.91767005,  1.6960161 ,  2.21159779,  2.481743  ,  2.04832546,  2.2241061 ,  1.74841013,  1.64231696,  1.73979868],
        [-1.31268744, -0.95751516, -1.38895222, -1.35711658, -1.60983628, -1.23343459, -1.53260016, -1.00735144, -0.71750574, -1.19222498],
        [-1.07031853, -0.97950833, -0.92169368, -1.11406715, -1.26485089, -1.07698061, -1.15342935, -0.85972191, -0.85162195, -0.95196546],
        [ 2.48590896,  2.1340656 ,  2.61672125,  2.54812942,  2.75621231,  2.17580725,  2.54459651,  2.19687274,  1.85805178,  2.01002515],
        [-0.27935879, -0.48545887, -0.45861193, -0.81527341, -0.1601232 , -0.39226443, -0.23627247, -0.8311206 , -0.71970933, -0.07086341],
        [ 0.0047408 , -0.1571119 , -0.1110397 , -0.40633937, -0.00746873, -0.05695421, -0.02006221, -0.44887558, -0.25947625,  0.17572035],
        [-1.04790573, -1.10484279, -1.13752501, -1.53919029, -0.84570166, -1.25159588, -0.96435136, -1.22787108, -1.40088798, -0.85569541],
        [-0.50865738, -0.39233984, -0.44267281,  0.08495965, -0.66433115, -0.33499851, -0.60794442, -0.23526454, -0.11038922, -0.83604841]])

Muxr_check 
matrix([[6.03900067, 4.90957755, 5.76412722, 4.35366105, 6.59393117, 4.67913593, 5.76133149, 3.82347511, 3.76180471, 5.20751874],
        [6.53510495, 4.89832208, 7.8136293 , 6.09998422, 6.18921502, 5.54355032, 6.27867782, 5.1499884 , 4.67559251, 5.70760688],
        [5.57117358, 6.97982763, 3.58891045, 4.23374722, 6.97800309, 5.39992029, 5.91402437, 5.90736875, 5.80095811, 6.20126818],
        [5.23630512, 6.69635782, 3.85914087, 3.84679702, 7.20646408, 4.32930211, 5.84272342, 6.54875216, 4.93025016, 5.54778374],
        [7.93867462, 7.01659364, 7.25178018, 7.38646393, 8.56599905, 7.67456647, 8.04805724, 5.90735227, 6.34836679, 7.40242864],
        [5.28932714, 5.80581862, 6.55346771, 4.68905489, 4.4825006 , 5.10903806, 5.24832466, 7.09294862, 6.212836  , 6.62298012],
        [6.34400303, 6.49004719, 5.10577634, 4.65357916, 7.16779059, 5.96249348, 6.49908437, 5.24801339, 5.49566281, 6.89337132],
        [3.95029394, 4.34141796, 4.43348542, 5.08633598, 2.90687435, 5.41287913, 4.11329326, 4.97737993, 5.68631712, 5.20340264],
        [6.21954794, 7.05024507, 3.35989683, 8.31755973, 9.23872877, 7.68452841, 7.89334537, 6.14080781, 5.85733053, 5.4796083 ],
        [6.54997811, 3.89847258, 7.9287086 , 6.98838735, 4.84643443, 6.5164061 , 5.70435032, 3.26627605, 4.96722018, 5.37829754]])
"""

## Q16_(3) M_key can not recover M_usr_x_rest_rank_dc in Q15 into its orignal weight matrix M_usr_x_rest_dc.

results_dc # orignal weight matrix in Q15
M_usr_x_rest_rank_dc # rank matrix in Q15
Muxr_check_dc = np.matmul(M_usr_x_rest_rank_dc, M_key) # use M-Key can not recover rank matrix into weight matrix

"""
results_dc: orignal weight matrix in Q15
array([[3.40647014, 2.31214523, 5.50307082, 3.84934625, 2.07865429, 2.80702393, 2.83259533, 3.49592719, 3.07599799, 2.76755506],
       [4.88977337, 3.27492688, 7.65046905, 5.78478747, 3.36716697, 4.37348032, 4.44821772, 4.94527095, 4.24696331, 4.18262958],
       [2.60957673, 4.05771627, 3.29522201, 3.66639307, 1.8983166 , 3.29379429, 2.61919619, 5.53887734, 5.02942555, 3.45630904],
       [2.9328409 , 4.42360454, 3.63071652, 3.40552158, 3.25559681, 2.69120411, 3.28007928, 6.26214773, 4.33016928, 3.41281552],
       [4.64801145, 3.76980324, 6.92545968, 6.75607043, 2.92190295, 5.33442647, 4.38713704, 5.49791737, 5.49110839, 4.35247404],
       [4.6311945 , 5.15646054, 6.48820362, 4.56297619, 3.35368138, 4.64101006, 4.51614062, 7.01106164, 6.04138432, 6.0129892 ],
       [3.38240618, 3.56793583, 4.81208789, 4.08622502, 2.0881041 , 3.85636748, 3.20425619, 4.87952198, 4.72413025, 4.14841218],
       [3.29216131, 3.69205988, 4.36822132, 4.96025728, 1.77805513, 4.94485113, 3.38110922, 4.89549295, 5.51486544, 4.59341172],
       [2.92888477, 3.80345467, 3.03357633, 7.68716623, 3.59463267, 5.34438841, 4.23242517, 5.73137291, 5.00007213, 2.4296537 ],
       [4.57558021, 1.95039834, 7.7329163 , 6.61015125, 1.45997677, 5.1123221 , 3.5077982 , 3.02061511, 4.45286514, 3.54832478]])

M_usr_x_rest_rank_dc: # rank matrix in Q15
array([[ 5,  9,  5,  8,  7,  9,  9,  9, 10,  9],
       [ 1,  8,  2,  4,  2,  6,  2,  6,  9,  4],
       [10,  3,  9,  9,  8,  8, 10,  4,  4,  7],
       [ 8,  2,  8, 10,  4, 10,  7,  2,  8,  8],
       [ 2,  5,  3,  2,  5,  2,  3,  5,  3,  3],
       [ 3,  1,  4,  6,  3,  5,  1,  1,  1,  1],
       [ 6,  7,  6,  7,  6,  7,  8,  8,  6,  5],
       [ 7,  6,  7,  5,  9,  4,  6,  7,  2,  2],
       [ 9,  4, 10,  1,  1,  1,  4,  3,  5, 10],
       [ 4, 10,  1,  3, 10,  3,  5, 10,  7,  6]])

Muxr_check_dc: M_Key can not recover M_usr_x_rest_rank_dc into weight matrix results_dc
matrix([[11.70508806,  9.54760152, 11.72302425, 10.70910964, 12.27972464, 10.27569231, 11.39556508,  8.33348635,  8.44158627,  9.85715361],
        [19.2375613 , 16.4010486 , 20.16198048, 20.37145845, 20.46739936, 17.67637219, 19.38369626, 16.48266919, 15.19992484, 15.68747274],
        [-3.78012002, -1.85507029, -5.65886517, -6.68343669, -3.16245367, -3.71497693, -3.54594822, -3.07254661, -2.7239003 , -1.11900314],
        [-0.62348987,  0.80901068, -2.55209975, -2.81160428,  1.38760822, -1.8723758 , -0.33035151, -0.15377666, -1.23379504, -0.4825106 ],
        [ 7.24565975,  6.30702351,  7.54052005,  7.90399507,  7.08038137,  7.42846829,  7.24504   ,  6.1741142 ,  6.53915277,  6.62636038],
        [ 4.5744957 ,  5.64805177,  4.12654008,  4.62443084,  4.85235904,  4.19165612,  4.30272507,  5.76865525,  5.44652307,  4.07092353],
        [10.33945095,  9.17701506,  9.58887207,  8.23410508, 11.06056263,  9.62034881, 10.50181759,  7.65578864,  7.99039958, 10.63988362],
        [ 5.98700124,  6.36150144,  5.54196627,  4.51181278,  5.71544064,  6.65210155,  6.31760746,  5.83882092,  6.43718859,  8.30144057],
        [ 2.52535038,  4.46073543,  0.49270717,  5.96062819,  4.57482566,  5.24092858,  4.40358632,  5.01524523,  4.5942631 ,  3.11145022],
        [ 2.46241161,  1.22976242,  4.69427748,  2.83507146, -0.07990674,  2.81360518,  1.62947427,  2.01990596,  3.04499603,  2.95109518]])
"""

## ------------ The end of homework 3 ----------------