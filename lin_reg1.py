#import numpy as np
import pandas as pd
import numpy as np
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

#Read in the data
df = pd.read_csv("cleaned_data.csv")

#Extract only the numerical values for now (not the 1-out-of-K)
cols = range(0, 27) 
raw_data = df.values 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])


# Split dataset into features and target vector
avgVote_idx = np.where(attributeNames=="avg_vote")[0][0]
y = X[:,avgVote_idx]
X_cols = list(range(0,avgVote_idx)) + list(range(avgVote_idx+1,len(attributeNames)))
X = X[:,X_cols]

#Removing metascore
metascore_idx = np.where(attributeNames=="metascore")[0][0]
X_cols2 = list(range(0,metascore_idx)) + list(range(metascore_idx+1,len(attributeNames)-1))
X = X[:,X_cols2]

N,M = X.shape

#Removing metascore and average vote from attribute names too
attributeNames = attributeNames[attributeNames != "avg_vote"]
attributeNames = attributeNames[attributeNames != "metascore"]
attributeNames = list(attributeNames)

#Standardization (Subtracting the mean from the data and dividing by the attribute standard deviation)
X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames 
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=5)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
opt_lambdas = np.empty((K,1))

internal_cross_validation = 10    
opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, internal_cross_validation)

k=0
figure(k, figsize=(12,8))
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()

