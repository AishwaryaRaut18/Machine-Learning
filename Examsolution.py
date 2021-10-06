"""
Created on Fri Aug 27 15:26:04 2021

@author: Rutuja
"""
### Question 1
import os
os.chdir(r"C:\Users\DELL\Desktop\210453083003_Rutuja_Arsule")

import pandas as pd


df = pd.read_csv("K:\set\Set A/Glass.csv",sep=",")

df.head()

df.shape

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

lda = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=5, random_state=2021,
shuffle=True)
results = cross_val_score(lda, X, y, cv=kfold, 
                          scoring='neg_log_loss')
AUCs = results

print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()*-1))


qda = QuadraticDiscriminantAnalysis()
kfold = KFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(qda, X, y, cv=kfold, 
                          scoring='neg_log_loss')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()*-1))

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'max_features': np.arange(1,20)}

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)

model_rf = RandomForestClassifier(random_state=2021)
cv = GridSearchCV(model_rf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc_ovr')

cv.fit( X , y )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)

best_model = cv.best_estimator_

##question 2

import pandas as pd
import numpy as np

df = pd.read_csv("K:\set\Set A\Sacremento.csv")
df1=df.drop(['zip'],axis=1)
dum_df = pd.get_dummies(df1, drop_first=True)


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression as LR


#X = dum_df.iloc[:,:-1]
#y = dum_df.iloc[:,-1]

X = dum_df.drop('price',axis=1)
y = dum_df['price']
X.shape
y.shape
 
#Linear Regression ( K-Fold CV )
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
kfold = KFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(linreg, X, y, cv=kfold, scoring='r2')
print(results)
print("r2: %.4f (%.4f)" % (results.mean(), results.std()))


#**********
#b•	X G Boost ( tune with few parameter sets )

lr_range = [0.001,0.01,0.2,0.5,0.6,1]
n_est_range = [30,70,100,120,150]
depth_range = [3,4,5,6,7,8,9]


parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=depth_range)


from sklearn.model_selection import KFold
kfolds = KFold(n_splits=5, random_state=2021,shuffle=True)

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
clf = XGBRegressor()
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)
df_cv = pd.DataFrame(cv.cv_results_)
print(cv.best_params_)

print(cv.best_score_)
#0.6601419778784329
##Questipon 3

import numpy as np
df = pd.read_csv("K:\set\Set A/USArrests.csv"
                 ,index_col=0)


from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
scaled=scaler.fit_transform(df)

scaled = pd.DataFrame(scaled,
                          columns=df.columns,
                          index=df.index)

# Import KMeans
from sklearn.cluster import KMeans


clustNos = [1,2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i,random_state=2021)
    model.fit(scaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: Best k model
model = KMeans(n_clusters=2,random_state=2021)

# Fit model to points
model.fit(scaled)

# Cluster Centroids
print(model.cluster_centers_)

#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(scaled)


clusterID = pd.DataFrame({'ClustID':labels},index=df.index)
clusteredData = pd.concat([df,clusterID],
                          axis='columns')

clusteredData.groupby('ClustID').mean()
clusteredData.sort_values('ClustID')

###############PCA#######################################################

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(scaled)
print(pca.explained_variance_)
print(np.sum(pca.explained_variance_))
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_ratio_ * 70) 
 
import matplotlib.pyplot as plt
ys = pca.explained_variance_ratio_ * 70
xs = np.arange(1,5)
plt.plot(xs,ys)
plt.show()


# principalComponents are PCA scores

df_plot = pd.DataFrame(principalComponents,
                 columns = ['PC1', 'PC2','PC3','PC4'],
                 index = df.index)

pca_loadings = pd.DataFrame(pca.components_.T, index=df.columns, columns=['V1', 'V2','V3','V4'] )
pca_loadings


#biplot
import seaborn as sns
 
# Scatter plot based and assigne color based on 'label - y'
sns.lmplot('PC1', 'PC2', data=df_plot, fit_reg = False, size = 15, scatter_kws={"s": 100})
 
# set the maximum variance of the first two PCs
# this will be the end point of the arrow of each *original features*
xvector = pca.components_[0]
yvector = pca.components_[1]
 
# value of the first two PCs, set the x, y axis boundary
xs = df_plot['PC1']
ys = df_plot['PC2']
 
## visualize projections
 
## Note: scale values for arrows and text are a bit inelegant as of now,
##       so feel free to play around with them
for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    # we can adjust length and the size of the arrow
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.005, head_width=0.05)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(df.columns.values)[i], color='r')
 
for i in range(len(xs)):
    plt.text(xs[i]*1.08, ys[i]*1.08, list(df.index)[i], color='b') # index number of each observations
plt.title('PCA Plot of first PCs')
plt.show()


## Question 4.

df = pd.read_csv("K:\set\Set A/WGEM-IND_CPTOTNSXN.csv")
df.head()

import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error


df.plot.line(x = 'Date',y = 'Value')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['Value'], lags=10)
plt.show()

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df['Value'], lags=12)
plt.show()

y = df['Value']
y_train = y[:25]
y_test = y[25:]


from pmdarima.arima import auto_arima
model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True)

from statsmodels.tsa.ar_model import AR
# train autoregression
model = AR(y_train)
model_fit = model.fit(maxlag=15)
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

########################## MA ##############################
from statsmodels.tsa.arima_model import ARMA

# train MA
model = ARMA(y_train,order=(0,1))
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

########################## ARMA ##############################
from statsmodels.tsa.arima_model import ARMA

# train ARMA
model = ARMA(y_train,order=(7,0))
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(y_train), 
                                end=len(y_train)+len(y_test)-1, 
                                dynamic=False)
    
error = mean_squared_error(y_test, predictions)
print('Test RMSE: %.3f' % sqrt(error))
# plot results
plt.plot(y_test)
plt.plot(predictions, color='red')
plt.show()

# plot
y_train.plot(color="blue")
y_test.plot(color="pink")
predictions.plot(color="purple")

rms = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rms)

################# ARIMA ####################################

##

train_set = data2.iloc[0:31:, :1].values
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train_set)

for i in range(11, 30):
    X_train.append(training_set_scaled[i-11:i, 0])
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Defining the LSTM Recurrent Model
regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))


#Compiling and fitting the model
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 15, batch_size = 32)