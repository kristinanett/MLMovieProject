import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from matplotlib.pylab import (figure, xlabel, ylabel, legend, title, subplot, show, grid)
from toolbox_02450 import train_neural_net
import torch
from scipy import stats

plt.rcParams.update({'font.size': 12})

#Read in the data
df = pd.read_csv("cleaned_data.csv")

#Extract only the numerical values for now (not the 1-out-of-K)
cols = range(0, 27) 
raw_data = df.values
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])


# Split dataset into features and target vector
avgVote_idx = np.where(attributeNames=="avg_vote")[0][0]
y = X[:,[avgVote_idx]]
X_cols = list(range(0,avgVote_idx)) + list(range(avgVote_idx+1,len(attributeNames)))
X = X[:,X_cols]

#Removing metascore
metascore_idx = np.where(attributeNames=="metascore")[0][0]
X_cols2 = list(range(0,metascore_idx)) + list(range(metascore_idx+1,len(attributeNames)-1))
X = X[:,X_cols2]

#Removing metascore and average vote from attribute names too
attributeNames = attributeNames[attributeNames != "avg_vote"]
attributeNames = attributeNames[attributeNames != "metascore"]

#Standardization (Subtracting the mean from the data and dividing by the attribute standard deviation)
#X = X - np.ones((N, 1))*X.mean(0)
#X = X*(1/np.std(X,0))
N,M = X.shape
C = 2

# Normalize data
X = stats.zscore(X)

##################################################################################################
def inner_folds(X,y,hidden_units,K):

    # Parameters for neural network classifier
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000

    # K-fold crossvalidation
    CV = model_selection.KFold(K, shuffle=True, random_state=6)

    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    test_error = np.empty((K,len(hidden_units)))
    train_error = np.empty((K,len(hidden_units)))
    
    f=0

    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nInner crossvalidation fold: {0}/{1}'.format(k+1,K))    
    
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        # Train the net on training data

        for l in range(0,len(hidden_units)):
            # Define the model
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, hidden_units[l]), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(hidden_units[l], 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )

            net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            y_train_est = net(X_train)
    
            # Determine error
            se_test = (y_test_est.float()-y_test.float())**2 # squared error
            mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
            
            se_train = (y_train_est.float()-y_train.float())**2 # squared error
            mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
            
            test_error[f,l]=mse_test
            train_error[f,l] = mse_train
            #errors.append(mse) # store error rate for current CV fold 
        f+=1

    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_h = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    test_err_vs_h = np.mean(test_error,axis=0)
    train_err_vs_h = np.mean(train_error,axis=0)
    
    print("Optimal h is ", opt_h, "with test error", opt_val_err)
    return opt_val_err, opt_h, test_err_vs_h, train_err_vs_h
###################################################################################################

    

K = 10
CV = model_selection.KFold(K, shuffle=True, random_state=5)
#CV = model_selection.KFold(K, shuffle=False)

# Values of hidden units
hidden_units = np.arange(1,16) 
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

# Initialize variables
#T = len(lambdas)
Error_train_ann = np.empty((K,1))
Error_test_ann = np.empty((K,1))

opt_hs = np.empty((K,1))

k=0
for train_index, test_index in CV.split(X,y):
    print('\nOuter crossvalidation fold: {0}/{1}'.format(k+1,K)) 
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10  
    
    opt_val_err, opt_h, test_err_vs_h, train_err_vs_h = inner_folds(X_train, y_train, hidden_units, internal_cross_validation)
    
    #Convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index]) 
    
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_h), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_h, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )

    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    y_train_est = net(X_train)
    
    # Determine error
    se_test = (y_test_est.float()-y_test.float())**2 # squared error
    mse_test = (sum(se_test).type(torch.float)/len(y_test)).data.numpy() #mean
            
    se_train = (y_train_est.float()-y_train.float())**2 # squared error
    mse_train = (sum(se_train).type(torch.float)/len(y_train)).data.numpy() #mean
            
    Error_test_ann[k] = mse_test
    Error_train_ann[k] = mse_train
    
    opt_hs[k] = opt_h


    # Display the results for the last cross-validation fold
    #if k == K-1:
    figure(k, figsize=(12,8))
    subplot(1,2,1)
    title('Optimal h: {0}'.format(opt_h))
    plt.plot(hidden_units,train_err_vs_h.T,'b.-',hidden_units,test_err_vs_h.T,'r.-')
    xlabel('Regularization factor')
    ylabel('Squared error (crossvalidation)')
    legend(['Train error','Validation error'])
    grid()
    
    subplot(1,2,2)
    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Average vote: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
print('ANN:')
print('- Training error: {0}'.format(Error_train_ann.mean()))
print('- Test error:     {0}'.format(Error_test_ann.mean()))
print(pd.DataFrame(data=np.concatenate((opt_hs, Error_test_ann), axis=1), columns=["Optimal h", "Test error"]))