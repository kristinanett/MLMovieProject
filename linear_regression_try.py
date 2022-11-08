#import numpy as np
import pandas as pd
import numpy as np
from matplotlib.pylab import figure, subplot, plot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm

#Read in the data
df = pd.read_csv("cleaned_data.csv")

#Extract only the numerical values for now (not the 1-out-of-K)
cols = range(0, 27) 
raw_data = df.get_values() 
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

#Standardization (Subtracting the mean from the data and dividing by the attribute standard deviation)
X = X - np.ones((N, 1))*X.mean(0)
X = X*(1/np.std(X,0))

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict alcohol content
y_est = model.predict(X)
residual = y_est-y

# Display scatter plot
figure()
subplot(2,1,1)
plot(y, y_est, '.')
xlabel('Average vote (true)'); ylabel('Average vote (estimated)');
subplot(2,1,2)
hist(residual,40)

show()

