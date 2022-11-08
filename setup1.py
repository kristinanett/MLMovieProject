import numpy as np
from sklearn import model_selection
import scipy.stats as st
import pandas as pd
from toolbox_02450 import train_neural_net
import torch

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

test_proportion = 0.1

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

########################Train baseline model#########################################
def train_baseline(y_train, y_test):
    yhatB = np.full(len(y_test), y_train.mean())
    return yhatB



########################Train linear regression model################################
def train_lin_reg(X_train, y_train, X_test, lamb):
    K=1
    k=0
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = lamb * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()

    yhatA = X_test @ w_rlr[:,k]
    return yhatA


########################Train ANN model##############################################
def train_ANN(X_train, y_train, X_test, opt_h):
    
    n_replicates = 1
    max_iter = 10000
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    
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
    yhatC = net(X_test)
    return yhatC 


yhatA = train_lin_reg(X_train, y_train, X_test, 100)
#print(yhatA)
yhatB = train_baseline(y_train, y_test)
yhatC = train_ANN(X_train, y_train, X_test, 12)
#print(yhatC)

#########################Statistical comparison#######################################

# compute z with squared error.
zA = np.abs(y_test - yhatA[:,np.newaxis]) ** 2
zB = np.abs(y_test - yhatB[:,np.newaxis]) ** 2
zC = np.abs(y_test - yhatC.detach().numpy()) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))
CIC = st.t.interval(1-alpha, df=len(zC)-1, loc=np.mean(zC), scale=st.sem(zC))

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z1 = zA - zB
z2 = zB - zC
z3 = zC - zA

CI1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))  # Confidence interval
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value

CI2 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))  # Confidence interval
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value

CI3 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))  # Confidence interval
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value

print("The lin-reg VS baseline confidence interval is:", CI1, " The p-value is:", p1)
print("The baseline VS ANN confidence interval is:", CI2, " The p-value is:", p2)
print("The ANN VS lin-reg confidence interval is:", CI3, " The p-value is:", p3)
