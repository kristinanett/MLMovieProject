#%%imports
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn import model_selection, tree
import matplotlib.pyplot as plt
from toolbox_02450 import mcnemar
import os
import contextlib
np.random.seed(2)

df = pd.read_csv("cleaned_data_one_genre.csv").drop(columns='Unnamed: 0')
#df = pd.read_csv("cleaned_data.csv").drop(columns='Unnamed: 0')
#df = df[np.sum(df.iloc[:,7:28],axis=1)==1]
#%% load the K-fold values from file
# put the file final_params.npz in the same folder to avoid running the kfold loop again
try:
    with np.load('final_params.npz') as f:
        table, X_train, y_train, X_test, y_test = f['table'], f['X_train'], f['y_train'], f['X_test'], f['y_test']
    K_fold = False
except FileNotFoundError:
    K_fold = True

#%%categorical encoding
classes = df.iloc[:,7:28]
# largest class is Drama corresponding to index 6 of classes (0 indexed)
# in the classes matrix

#largest_class = str(classes.columns[largest_class])
# y is target vector (true labels)
classes2 = np.array(classes)
y = np.zeros(len(df))
for j in range(len(df)):
    for i in range(20):
        y[j] += classes2[j,i]*(i)
#standardise
X = np.array(df.iloc[:,:7])
N = len(X)
X = (X - np.ones((N,1))*X.mean(axis=0))/np.std(X,axis=0)

#%%baseline
def base_classify(X_train, y_train, X_test, y_test):
    '''
    base classifier, returns the predicted class of X_test as
    the largest class
    '''
    largest_class = np.argmax(np.bincount(y_train.astype(int)))
    y_hat = np.ones(len(y_test))*largest_class
    return y_hat
#%% models
#  regularization strength was found during a trial run
# doesn't seem to change much going from 1e-12 to 1. gets worse if it's higher than 1
# Fit decision tree classifier, Gini split criterion
# depth of 5 was found based on trial runs
def run_base(X_train, y_train, X_test, y_test):
    base_predictions = base_classify(X_train, y_train, X_test, y_test)
    base_error_rate = sum(base_predictions!=y_test)/ len(y_test)
    return base_error_rate, base_predictions
def run_logreg(X_train, y_train, X_test, y_test, model_param):
    logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-7, random_state=1, C=1/model_param)
    logreg.fit(X_train, y_train)
    logreg_predictions = logreg.predict(X_test)
    logreg_error_rate = np.sum(logreg_predictions!=y_test) / len(y_test)
    return logreg_error_rate, logreg_predictions
def run_tree(X_train, y_train, X_test, y_test, model_param):
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=model_param)
    dtc = dtc.fit(X_train,y_train)
    tree_predictions = np.asarray(dtc.predict(X_test),dtype=int)
    tree_error_rate = sum(tree_predictions != y_test) / len(y_test)
    return tree_error_rate, tree_predictions

# %%
if K_fold:
    K = 10
    #model complexity
    tc = np.arange(2, 10, 1)
    regularization_strength = np.power(10.,range(-12,-4))

    optimal_tree_depth = np.zeros(K)
    optimal_rs = np.zeros(K)
    Error_test_tree = np.zeros(K)
    Error_test_in_tree = np.zeros((len(tc),K))
    Error_test_base = np.zeros(K)
    Error_test_logreg = np.zeros(K)
    Error_test_in_logreg = np.zeros((len(regularization_strength),K))

    CV = model_selection.KFold(K, shuffle=True, random_state=1)
    for k, (train_index, test_index) in enumerate(CV.split(X)):
        print('Computing CV fold: {0}/{1}..'.format(k+1,K))

        # extract training and test set for current CV fold
        X_train, y_train = X[train_index,:], y[train_index]
        X_test, y_test = X[test_index,:], y[test_index]
        CV2 = model_selection.KFold(K, shuffle=True, random_state=5)

        for j, (train_index, test_index) in enumerate(CV2.split(X_train)):
            X_train_in, y_train_in = X_train[train_index,:], y_train[train_index]
            X_test_in, y_test_in = X_train[test_index,:], y_train[test_index]
            for i, t in enumerate(tc):
                Error_test_in_tree[i,k], _ = run_tree(X_train_in, y_train_in, X_test_in, y_test_in, t)
            for i, t in enumerate(regularization_strength):
                Error_test_in_logreg[i,k], _ = run_logreg(X_train_in, y_train_in, X_test_in, y_test_in, t)

        Error_test_base[k], _ = run_base(X_train, y_train, X_test, y_test)

        #select optimal models
        Error_test_in_tree_mean = np.mean(Error_test_in_tree,axis=1)
        # +2 because the 0'th index in the tc vector i 2.
        optimal_tree_depth[k] = np.argmin(Error_test_in_tree_mean)
        Error_test_in_logreg_mean = np.mean(Error_test_in_logreg,axis=1)
        optimal_rs[k] = np.argmin(Error_test_in_logreg_mean)
        # 
        Error_test_tree[k], _ = run_tree(X_train, y_train, X_test, y_test, tc[optimal_tree_depth[k].astype(int)])
        Error_test_logreg[k], _ = run_logreg(X_train, y_train, X_test, y_test, regularization_strength[optimal_rs[k].astype(int)])


    # print(f'baseline error rate: {round(base_error_rate,ndigits=3)}')
    # print(f'Logistic (multinomial) regression error rate: {round(logreg_error_rate,ndigits=3)}')
    # print(f'Tree error rate: {round(tree_error_rate,ndigits=3)}')          

    #export 
    table = np.vstack((np.arange(0,K).astype(int), tc[optimal_tree_depth.astype(int)], Error_test_tree, regularization_strength[optimal_rs.astype(int)], Error_test_logreg, Error_test_base)).T
    pdtable = pd.DataFrame(table,columns=['i','x','Etest','lambda','Etest','Etest'])
    pdtable.to_latex('classtable.tex', index=False)
    np.savez('final_params',table=table,X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)
#%% print errors
print(f'base Egen: {np.round(table[:,5], 4)}')
print(f'Logreg Egen: {np.round(table[:,4], 4)}')
print(f'Tree Egen: {np.round(table[:,2], 4)}')  

# %% statistical evaluation. McNemar's test
_, base_predictions = run_base(X_train, y_train, X_test, y_test)
_, tree_predictions = run_tree(X_train, y_train, X_test, y_test, table[:,1][np.argmin(table[:,2]).astype(int)])
_, logreg_predictions = run_logreg(X_train, y_train, X_test, y_test, table[:,3][np.argmin(table[:,4]).astype(int)])

alpha= 0.05
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    #suppress output from mcnemar function
    #https://stackoverflow.com/a/46129367
    [thetahat, CI, p] = mcnemar(y_test, base_predictions, tree_predictions, alpha=alpha)
print('base vs. tree')
print("theta = theta_A-theta_B point estimate", round(thetahat,5), " CI: ", np.round(CI,4), "p-value", round(p,4))

#base vs logreg
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    [thetahat, CI, p] = mcnemar(y_test, base_predictions, logreg_predictions, alpha=alpha)
print('base vs. logistic regresion')
print("theta = theta_A-theta_B point estimate", round(thetahat,4), " CI: ", np.round(CI,4), f"p-value {p:.3e}")
#tree vs logreg
print('tree vs. logistic regresion')
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    [thetahat, CI, p] = mcnemar(y_test, logreg_predictions, tree_predictions, alpha=alpha)
print("theta = theta_A-theta_B point estimate", round(thetahat,4), " CI: ", np.round(CI,4), "p-value", round(p,4))


# %%

# clf part 5
# To display coefficients use print(logreg.coef_). For a 4 class problem with a 
# feature space, these weights will have shape (4, 2).
logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-7, random_state=1, C=1/(1e-06))
logreg.fit(X_train, y_train)
logreg_predictions = logreg.predict(X_test)
logreg_coefs =  pd.DataFrame(logreg.coef_,columns=list(df.columns[:7]))
print(logreg_coefs)

# %%
