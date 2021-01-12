#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:45:00 2019

@author: shanqinggu

"""
## --------------------------------------------------------------------------------------------------------
## Homework 2 for DS 7335 Machine Learning
## --------------------------------------------------------------------------------------------------------

## Running enviroments: Python 3.7.0, IPython 7.2.0, Spyder 3.3.3
## References: 1) Office hour codes and discussions 
##             2) Fix bug in np.genfromtxt (stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl)
##             3) String Operations in NumPy (docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html)
##             4) Sorting, Searching, and Counting in NumPy(docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html)

## --------------------------------------------------------------------------------------------------------
## Import libraries
## --------------------------------------------------------------------------------------------------------

import os
path="/Users/shanqinggu/Desktop/SMU_DS_2019/MSDS 7335_Machine Learning/HW2"
os.chdir(path)
print(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import functools
import io
import numpy as np
# import pandas as pd
import sys
import numpy.lib.recfunctions as rfn
import time

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from itertools import product
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

print(__doc__)

## --------------------------------------------------------------------------------------------------------
# Fix a bug in np.genfromtxt when Python Version (sys.version_info) is 3 or greater. 
# https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed
  
## --------------------------------------------------------------------------------------------------------
  
# Read the two first two lines of the file to show column names and the 1st column
with open('data/claim.sample.csv', 'rb') as f:
    print(f.readline())
    print(f.readline())
    
# Colunn names (total 29)
names = ["V1","Claim.Number","Claim.Line.Number",
         "Member.ID","Provider.ID","Line.Of.Business.ID",
         "Revenue.Code","Service.Code","Place.Of.Service.Code",
         "Procedure.Code","Diagnosis.Code","Claim.Charge.Amount",
         "Denial.Reason.Code","Price.Index","In.Out.Of.Network",
         "Reference.Index","Pricing.Index","Capitation.Index",
         "Subscriber.Payment.Amount","Provider.Payment.Amount",
         "Group.Index","Subscriber.Index","Subgroup.Index",
         "Claim.Type","Claim.Subscriber.Type","Claim.Pre.Prince.Index",
         "Claim.Current.Status","Network.ID","Agreement.ID"]

# For dtype (ref: docs.scipy.org/doc/numpy-1.12.0/reference/arrays.dtypes.html)
'''
typesCheck = [np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(float),
         np.dtype(object), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(float),
         np.dtype(object), np.dtype(object), np.dtype(object), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(float),
         np.dtype(float), np.dtype(float), np.dtype(float), np.dtype(object),
         np.dtype(object), np.dtype(object), np.dtype(float), np.dtype(object),
         np.dtype(object)]
'''

# Data types after using typesCheck instead of types in the below function
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']    

# numpy.genfromtxt (docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.genfromtxt.html)
# Creates array of structured arrays
CLAIMS = np.genfromtxt('data/claim.sample.csv', dtype=types, delimiter=',', names=True,
                       usecols=[0,1,2,3,4,5,6,7,8,9,
                                10,11,12,13,14,15,16,17,18,19,
                                20,21,22,23,24,25,26,27,28])

print(CLAIMS.dtype) # dtype to CLAIMS arryas
print(CLAIMS.shape) # the shape differs for using structured arrays
print(CLAIMS[0])    # slice into it to get a specific row
print(CLAIMS[0][1]) # slice into to get a specific value
print(CLAIMS.dtype.names) # get the names
print(CLAIMS['MemberID'] )# slice into a column
print(CLAIMS[0]['MemberID']) # slice into a column and a row value

# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl
# solution: in the with open(), use 'rb' instead of 'r'
test = 'J'
test = test.encode()

# example of substring searching
np.core.defchararray.find(CLAIMS['ProcedureCode'],test) # substring search 'ProcedureCode'
JcodeIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'],test)==1)

# using those indexes, subset CLAIMS to only Jcodes
Jcodes = CLAIMS[JcodeIndexes]

## --------------------------------------------------------------------------------------------------------
# QUESTION 1A: How do you find the number of 'Claim.Line.Number's that have J-codes with "Jcodes"?

print('---------------------------------------------------')
print('Number of claims that have Jcodes: ', len(Jcodes))  ## There are total 51029 Jcode claims
print('---------------------------------------------------')

print('---------------------------------------------------')
print('Q1A Answer: The unique Claim.Line.Numbers that have Jcodes: ', len(np.unique(Jcodes['ClaimLineNumber'])))
print('---------------------------------------------------')


## --------------------------------------------------------------------------------------------------------
# QUESTION 1B: How much was paid for J-codes to providers for 'in network' claims?

inNet = 'I' # Find 'I' in 'inOutofNetwork' in Jcodes
inNet = inNet.encode()

np.core.defchararray.find(Jcodes['InOutOfNetwork'],inNet) # substring search 'InoutofNetwork'
inNetIndexes = np.flatnonzero(np.core.defchararray.find(Jcodes['InOutOfNetwork'],"I".encode())==1) # only "I"

# using those indexes, subset Jcodes to only inNet
inNetJcodes = Jcodes[inNetIndexes]

print('---------------------------------------------------')
print('Q1B Answer: Amount paid for J-codes to providers for in network claim: 2417220.96029')
print(np.sum(inNetJcodes['ProviderPaymentAmount'])) 
print('---------------------------------------------------')

## --------------------------------------------------------------------------------------------------------
# QUESTION 1C: What are the top five J-codes based on the payment to providers?

# Sort Jcodes by ProviderPaymentAmount and reverse the sorted Jcodes
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')
Sorted_Jcodes = Sorted_Jcodes[::-1]

# Subset with ProcedureCode
ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
Jcodes = Sorted_Jcodes['ProcedureCode']

# Recall their data types and print
Jcodes.dtype
ProviderPayments.dtype
print(Jcodes)
print(ProviderPayments)

# Top 3 Jcodes with ProviderPayment
Jcodes[:3]
ProviderPayments[:3]

#Join arrays together
arrays = [Jcodes, ProviderPayments]

# Merge Jcodes and ProvicerPayments (www.numpy.org/devdocs/user/basics.rec.html)
Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

# Print and find the top 5 Jcodes with ProviderPayments
print(Jcodes_with_ProviderPayments[:5]) # not distinct Jcodes (three 'J9310'), should group and distinct?

# Recall the data types (51029)
Jcodes_with_ProviderPayments.shape

# Fast groupby-apply operations (esantorella.com/2016/06/16/groupby/) without Pandas, the Groupby class
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result

# See how long the groupby takes
start = time.clock()
# grouped = Groupby(Jcodes)

# Perform the groupby to get the group sums
group_sums = Groupby(Jcodes).apply(np.sum, ProviderPayments, broadcast=False)
print('time to compute group sums once with Grouped: {0}'\
      .format(round(time.clock() - start, 3)))

# Recall the data types (203)
group_sums.shape

# group_sums
np.set_printoptions(threshold=500, suppress=True)
print(group_sums)

# Get the JCodes for the group sums with the class Groupby
unique_keys, indices = np.unique(Jcodes, return_inverse = True)

print(unique_keys)
print(indices)

len(unique_keys)
len(group_sums)    

print(group_sums)

# Zip and sort
zipped = zip(unique_keys, group_sums) 
sorted_group_sums = sorted(zipped, key=lambda x: x[1]) # anonymous lambda function

# Reverse the sorted_group_sums to creat a new array
sorted_group_sums = sorted_group_sums[::-1]

# Print the top five J-codes based on the payment to providers
print(sorted_group_sums) # distinct 5 J-codes based on the payment to providers

print('---------------------------------------------------')
print('Q1C Answer: the top five J-codes based on the payment to providers: J7620, J7613, J3475, J0131, J1745')
print(sorted_group_sums[:5]) 
print('---------------------------------------------------')

## --------------------------------------------------------------------------------------------------------
## QUESTION 2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
## QUESTION 2A: Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.

# Aggregate Provider IDs and Provider Payment Amounts to get a total count for Unpaid claims, Paid claims
# Validate the counts by providers...
# Create labels for the providers to label them in the plot.

print(Sorted_Jcodes.dtype.names)

## labels for paid and unpaid Jcodes

## Find unpaid row indexes  
unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)
unpaid_mask.shape

## Find paid row indexes
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)
paid_mask.shape

# Here are our Jcodes for Unpaid and Paid
Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]
Paid_Jcodes = Sorted_Jcodes[paid_mask]

# These are still structured numpy arrays
print(Unpaid_Jcodes.dtype.names)
print(Unpaid_Jcodes[0])

print(Paid_Jcodes.dtype.names)
print(Paid_Jcodes[0])

# Now I need to create labels
print(Paid_Jcodes.dtype.descr)
print(Unpaid_Jcodes.dtype.descr)

# Create a new column and data type for both structured arrays
new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

print(new_dtype1)
print(new_dtype2)

# Create new structured array with labels

# First get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype2)

# Check the shape
Unpaid_Jcodes_w_L.shape
Paid_Jcodes_w_L.shape

# Look at the data
print(Unpaid_Jcodes_w_L)
print(Paid_Jcodes_w_L)

# Copy the data
Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

# And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

# Look at the data..
print(Unpaid_Jcodes_w_L)


# Do the same for the Paid set.

# Copy the data
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

# And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0

# Look at the data..
print(Paid_Jcodes_w_L)

#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)

#check the shape
Jcodes_w_L.shape

#44961 + 6068

# look at the transition between the rows around row 44961
print(Jcodes_w_L[44960:44963])

# We need to shuffle the rows before using classifers in sklearn
Jcodes_w_L.dtype.names

## Aggregate Provider IDs and Provider Payment Amounts to get a total count for: Unpaid claims, Paid claims
## Validate the counts by providers...
## Create labels for the providers to label them in the plot.

## Refer: https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
ProviderID1 = Unpaid_Jcodes_w_L['ProviderID']
ProviderID1[1:10]
ProviderIDN = Paid_Jcodes_w_L['ProviderID']
ProviderIDN[0:10]
UNPAIDAGG = Unpaid_Jcodes_w_L['ProviderPaymentAmount']
UNPAIDAGG[0:10]
PAIDAGG = Paid_Jcodes_w_L['ProviderPaymentAmount']
PAIDAGG[0:10] 

## Make 'ProviderID1' from the 'ProviderID'in Unpaid_Jcodes_w_L
ProviderID1 = Unpaid_Jcodes_w_L['ProviderID']
UP_ID,UNPAID = np.unique(ProviderID1,return_counts = True) 
print([UP_ID, UNPAID])
len(UP_ID) # UP_ID length is 15

## Make 'ProviderIDN' from the 'ProviderID'in Paid_Jcodes_w_L
ProviderIDN = Paid_Jcodes_w_L['ProviderID']
P_ID,PAIDAGG = np.unique(ProviderIDN,return_counts = True) 
print([P_ID, PAIDAGG])
len(P_ID) # P_ID length is 13

## Find the common 'ProviderID' for 'UP_ID' and 'P_ID'
mask = np.isin(UP_ID, P_ID, invert=True) 
UP_ID[mask] # find: b'"FA0001774002"', b'"FA1000016002"'

## Find and confirm 'mask' positions at [ 7, 14] with values [8, 6]
index=np.where(mask)
print(index, mask)

## Make 'UNPAIDAGG' to delete these 2 values from 'UNPAID' to make equal size for scatter plot
UNPAIDAGG=np.delete(UNPAID, index)
print(UNPAID, UNPAIDAGG)

print('---------------------------------------------------------')
print('Q2A1: Scatter plots with no log-transformed claim numbers')
print('---------------------------------------------------------')

## Produce the scatterplot as the answer to 2a with untranformed claim numbers
## Reference: stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
FIG, AX = plt.subplots()
AX.scatter(UNPAIDAGG, PAIDAGG)
AX.grid(linestyle='-', linewidth='0.5', color='red')

FIG = plt.gcf()
FIG.set_size_inches(10, 10)
plt.rcParams.update({'font.size': 12})

for i, txt in np.ndenumerate(P_ID):
    AX.annotate(txt, (UNPAIDAGG[i],PAIDAGG[i]))
   
plt.tick_params(labelsize=12)
plt.xlabel('# of Unpaid claims', fontsize=12)

plt.ylabel('# of Paid claims', fontsize=12)

plt.title('Scatterplot of Unpaid and Paid claims by Provider', fontsize=15)
plt.savefig('Paid_Unpaid_Scatterplot.png')


## Produce the scatterplot as the answer to 2a with log-transformed claim numbers
logPAIDAGG = np.log(PAIDAGG)
logUNPAIDAGG = np.log(UNPAIDAGG)

print('---------------------------------------------------------')
print('Q2A2: Scatter plots with log-transformed claim numbers')
print('---------------------------------------------------------')

FIG, AX = plt.subplots()
AX.scatter(logUNPAIDAGG, logPAIDAGG)
AX.grid(linestyle='-', linewidth='0.5', color='red')

FIG = plt.gcf()
FIG.set_size_inches(10, 10)
plt.rcParams.update({'font.size': 12})

for i, txt in np.ndenumerate(P_ID):
    AX.annotate(txt, (logUNPAIDAGG[i],logPAIDAGG[i]))
   
plt.tick_params(labelsize=12)
plt.xlabel('# of log Unpaid claims', fontsize=12)

plt.ylabel('# of log Paid claims', fontsize=12)

plt.title('Scatterplot of Unpaid and Paid log transformed claims by Provider', fontsize=15)
plt.savefig('Log_Paid_Unpaid_Scatterplot.png')

np.cov(UNPAIDAGG,PAIDAGG, bias=True)[0][1]
np.cov(logUNPAIDAGG,logPAIDAGG, bias=True)[0][1]

np.corrcoef(UNPAIDAGG,PAIDAGG) # corrcoef = 0.63 for untransformed claims
np.corrcoef(logUNPAIDAGG,logPAIDAGG) # corrcoef = 0.78 for log transformed claims


## --------------------------------------------------------------------------------------------------------
## QUESTION 2B: What insights can you suggest from the graph?

print('-------------------------------------------------------------------')
print('Q2B: The scatter plots, we can find: ', 
      '\n\n1. Most unpaid claims are higher than the paid claims', 
      '\n2. The provider FA0001389001 has the highest unpaid claims', 
      '\n3. The provider FA0001387002 has the highest paid claims', 
      '\n4. Positive relationship between paid and unpaid claims (corrcoef = 0.63 or 0.78 for un- or log- transformed claims)')
print('-------------------------------------------------------------------')

## --------------------------------------------------------------------------------------------------------
## QUESTION 2C:Based on the graph, is the behavior of any of the providers concerning? Explain.
print('-------------------------------------------------------------------')
print('Q2C: Is the behavior of any of the providers concerning?', 
      '\n\nThere are several outliers (e.g. FA10001387001) which have more unpaid claims', 
      '\nThere are several outliers (e.g. FA1000015001) which have more paid claims', 
      '\nNeed to investigate the claim entry date and process date to confirm if those claims are processed properly')
print('-------------------------------------------------------------------')
## --------------------------------------------------------------------------------------------------------

## --------------------------------------------------------------------------------------------------------
## QUESTION 3. Consider all claim lines with a J-code.
## QUESTION 3A: What percentage of J-code claim lines were unpaid?

up_cln = Unpaid_Jcodes_w_L['ClaimLineNumber']
p_cln = Paid_Jcodes_w_L['ClaimLineNumber']

print('-------------------------------------------------------------------')
print('Q3A: What percentage of J-code claim lines were unpaid: 0.8811', len(up_cln)/(len(p_cln)+len(up_cln)))
print('-------------------------------------------------------------------')
## --------------------------------------------------------------------------------------------------------

## QUESTION 3B: Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

print('-------------------------------------------------------------------')
print("Q3B: Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.",
      "\n\nFor this Q3B, 3 classifiers will be explpored. The output of binary logistic regression is more informative than other classification algorithms", 
      "since it expresses the relationship between the labels and each of features"
      "\n\nHowever, a random forest as well with other variations of",
      "the decision tree method will tell which predictors are more important to build the trees, without any information on the direction of association",
      "\n\nThe decision of which algorithm to finally deploy will be based on accuracy, precision and recall")
print('-------------------------------------------------------------------')

# Prep for machine learning with classifiers in sklearn

# We want to do the same for our data since we have combined unpaid and paid together, in that order. 
print(Jcodes_w_L[44957:44965])

# Apply the random shuffle
np.random.shuffle(Jcodes_w_L)
print(Jcodes_w_L[44957:44965])

# Columns are still in the right order
Jcodes_w_L

# Now get in the form for sklearn
Jcodes_w_L.dtype.names

label =  'IsUnpaid'


## Keep 17 cat_creatures and remove 'V1', 'DiagnosisCode', 'ClaimType'from categorical variables
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode', 'ServiceCode', 'PlaceOfServiceCode', 
                'ProcedureCode', 'DenialReasonCode','PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType', 'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 
                'NetworkID', 'AgreementID']

## Keep 'ClaimChargeAmount' from numeric_features and remove response varaible 'ProviderPaymentAmount' 
## Remove 'MemberID', 'SubscriberPaymentAmount', 'SubscriberIndex', 'SubgroupIndex','GroupIndex', 'ClaimNumber', 'ClaimLineNumber','MemberID' 
## Remove single value variables: 'SubscriberPaymentAmount', 'SubscriberIndex', 'SubgroupIndex', 'GroupIndex', 'ClaimNumber', 'ClaimLineNumber'
numeric_features = ['ClaimChargeAmount'] 


Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())

# Run the Label encoder
le = preprocessing.LabelEncoder()
for i in range(17):
    Mcat[:,i] = le.fit_transform(Mcat[:,i])

# Run the OneHotEncoder
ohe = OneHotEncoder(sparse=False) 
Mcat = ohe.fit_transform(Mcat)
print(Mcat)

# Check the shape of the matrix categorical columns that were OneHotEncoded   
Mcat.shape
Mnum.shape

print("%d Megabytes" % ((Mcat.size * Mcat.itemsize)/1048576))
print("%d Megabytes" % ((Mnum.size * Mnum.itemsize)/1048576))

# Concatenate the columns to create arrays for the features and the response variable
M = np.concatenate((Mcat, Mnum), axis=1)
L = Jcodes_w_L[label].astype(int)

# Match the label rows to the subset matrix rows.

M.shape
L.shape

# Use previous DeathToGridsearch code with cv = 5. 
n_folds = 5

# Pack the arrays together into "data"
data = (M,L,n_folds)

# run() for all classifiers against medical claim data
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
 
# run() with a list and a for loop
clfsList_LRD = [LogisticRegression, RandomForestClassifier, DecisionTreeClassifier] 

clfDict_LRD = {'LogisticRegression': {"C" : [1, 10],
                                      "tol" : [0.1, 1],
                                      "penalty" : ['l2' ],
                                      "solver":['lbfgs','liblinear']},
               'RandomForestClassifier': {"n_estimators":[10, 100],
                                          "max_depth":[5, 20],
                                          "min_samples_split":[2, 8]},
                                      
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
       
plt.show()

## --------------------------------------------------------------------------------------------------------
## QUESTION 3B1: LogisticRegression() with cross-validation (cv = 5) and more parameters
## --------------------------------------------------------------------------------------------------------

# Split the dataset in two equal parts
M_train, M_test, L_train, L_test = train_test_split(M, L, test_size=0.2, random_state=0)

# Set the LogisticRegression() parameters by cross-validation (cv = 5)
tuned_parameters = [{"C" : [1, 10, 50], 
                     "tol" : [0.01,0.1, 1],
                     "penalty" : ['l2' ], 
                     "solver":['lbfgs','liblinear', 'newton-cg']}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(M_train, L_train)

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
    
    L_true, L_pred = L_test, clf.predict(M_test)
    print(classification_report(L_true, L_pred))

## --------------------------------------------------------------------------------------------------------
## QUESTION 3B2: RandomForestClassifier() with cross-validation (cv = 5)
## --------------------------------------------------------------------------------------------------------
    
# Split the dataset in two equal parts
M_train, M_test, L_train, L_test = train_test_split(M, L, test_size=0.2, random_state=0)

# Set the RandomForestClassifier() parameters by cross-validation (cv = 5)
tuned_parameters = [{"n_estimators":[10, 100],
                     "max_depth":[5, 20],
                     "min_samples_split":[2, 8]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(M_train, L_train)

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
    
    L_true, L_pred = L_test, clf.predict(M_test)
    print(classification_report(L_true, L_pred))      
    
## --------------------------------------------------------------------------------------------------------------------------
## QUESTION 3B3: Try use multiclassifiers(RandomForestClassifier, LogisticRegression, DecisionTreeClassifier) with with cross-validation (cv = 5)
## --------------------------------------------------------------------------------------------------------------------------

# Define a EstimatorSelectionHelper class by passing the models and the parameters
    
import pandas as pd ## has to use pd for two lines of codes (pd.series({**params,**d}) and pd.concat) for EstimatorSelectionHelper()

class EstimatorSelectionHelper:
    
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=5, n_jobs=3, verbose=1, scoring=None, refit=False):
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
params = {'LogisticRegression': {"C" : [1, 10],
                                 "tol" : [0.1, 1],
                                 "penalty" : ['l2' ],
                                 "solver":['lbfgs','liblinear']},
        
           'RandomForestClassifier': {"n_estimators":[10, 100], 
                                      "max_depth":[5, 20],
                                      "min_samples_split":[2, 8]},
                                      
           'DecisionTreeClassifier': {"max_depth":[2, 4, 6],
                                      "criterion":['entropy', 'gini']}
           }
 
# call the fit() function, which as signature similar to the original GridSearchCV object
helper = EstimatorSelectionHelper(models, params)
helper.fit(M, L, scoring='f1', n_jobs=2) # scoring with 'f1'

# Inspect results of each model and each parameters by calling the score_summary method
summary = helper.score_summary(sort_by='max_score')
print(summary.T) # show score summary (e.g.,the best and the worst classfier with parameters in 13 rows x 150 columns)
print(summary)

print('----------------------------------------------------------')
print('3C:How accurate is your model at predicting unpaid claims?')
print('\n\nBest accuracy 0.96 for LR with C=50, penalty=l2, solver=newton-cg, tol=1',
      '\n\nBest accuracy 0.97 for RF with max_depth=20, min_samples_split= 2, n_estimators=50',
      '\n\nBest accuracy 0.99 for  DecisionTreeClassifier',
      '\n\nAll three models have very high accuracies for model prediction',
      '\n\nFurther background information should be given for variable selections to prevent model overfit')
print('----------------------------------------------------------')

print('-------------------------------------------------------------------------------')
print('3D: What data attributes are predominately influencing the rate of non-payment?')
print('-------------------------------------------------------------------------------')

## Here we use RandomForest and Logistic Regression to address Q3D

## Use optimized parameters for RandomForrestClassifier() to get the top 10 important features for influencing the rate of non-payment
 
# Get feature names from cat_fature
ohe.inverse_transform(Mcat)
ohe_features = ohe.get_feature_names(cat_features).tolist()

# Get al feature names
all_features = np.append(ohe_features, numeric_features)
  
##  Make selectKImportance function to get the top 10 important attributes
def selectKImportance(model, X, k=10):
    return X[:,model.feature_importances_.argsort()[::-1][:k]]

## Fit with optimized parameters as shown in Q3C
RF = RandomForestClassifier(n_estimators=100, max_depth = 20, min_samples_split =2)
RF.fit(M, L)

## Run selectKImportance function
RF10 = selectKImportance(RF,M,10)
RF10.shape
M.shape

## Get importances and indices
importances = RF.feature_importances_
indices = np.argsort(importances)[::-1]

## make selectFeature function to get all feature namess
def selectFeatures(model, X, T, k=10):
    return T[model.feature_importances_.argsort()[::-1][:k]]

F10=selectFeatures(RF, M, all_features, k=10)

# Print top 10 important feature ranking from RandomForestClassifier()
## Top 10 important features: 'DenialReasonCode_0.0' 'DenialReasonCode_54.0' 'DenialReasonCode_89.0', 'AgreementID_1.0' 
## 'ServiceCode_8.0' 'ServiceCode_10.0', 'LineOfBusinessID_5.0' 'NetworkID_9.0' 'PriceIndex_0.0', 'ReferenceIndex_2.0'
print('-------------------------------------------------------------------------------')
print("\nTop 10 important feature ranking from RandomForestClassifier():\n\n", F10)

for f in range(RF10.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot top 10 important feature ranking with feature_importance_ values
plt.figure()
plt.title("Feature importances")
plt.bar(range(RF10.shape[1]), importances[indices][:10], color="#b3ccff", align="center")
plt.xticks(range(RF10.shape[1]), F10, rotation = 60, ha='right')
plt.xlim([-1, RF10.shape[1]])
plt.show()

print('-------------------------------------------------------------------------------')

## --------------------------------------------------------------------------------------------------------
## The End of HW2
## --------------------------------------------------------------------------------------------------------


