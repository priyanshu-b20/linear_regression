#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[2]:


#loading dataset
diabetes = load_diabetes()

#df
features = pd.DataFrame(diabetes.data,columns=diabetes.feature_names)
target = pd.DataFrame(diabetes.target,columns=['target'])
df = pd.concat([features,target],axis=1)
print(df)


# In[3]:


# corr
corrs = [abs(df.corr('spearman')[x]['target']) for x in features]
lst = list(zip(corrs, features))
lst.sort(key = lambda x : x[0], reverse =True)
corrs, labels = list(zip((*lst)))

index = np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index, corrs, width=0.5)
plt.xlabel('attributes')
plt.ylabel('correlation with the target vars')
plt.xticks(index,labels)
plt.show()


# In[4]:


#selecting attr with most corr with the target
x = df['s5'].values
y = df['target'].values

#scaling
x_scaler = MinMaxScaler()
x = x_scaler.fit_transform(x.reshape(-1,1))
x = x[:,-1]
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y.reshape(-1,1))
y = y[:,-1]


# In[5]:


#splitting the data
xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=0.2, random_state=0)


# In[6]:


#learning
def error(m, x, c, t):
    return sum(((m * x + c) - t) ** 2) * 1/(2 * x.size)

def update(m, x, c, t, learning_rate):
    grad_m = sum(2 * ((m * x + c) - t) * x)
    grad_c = sum(2 * ((m * x + c) - t))
    m = m - grad_m * learning_rate
    c = c - grad_c * learning_rate
    return m, c

def gradient_descent(init_m, init_c, x, t, learning_rate, iterations, error_threshold):
    m = init_m
    c = init_c
    error_values = list()
    mc_values = list()
    for i in range(iterations):
        e = error(m, x, c, t)
        if e < error_threshold:
            print('error less than the threshold. stopping gradient descent')
            break
        error_values.append(e)
        m, c = update(m, x, c, t, learning_rate)
        mc_values.append((m,c))
    return m, c, error_values, mc_values


# In[7]:

init_m =0.9
init_c = 0
learning_rate = 0.001
iterations = 250
error_threshold = 0.001

m, c, error_values, mc_values = gradient_descent(init_m, init_c, xtrain, ytrain, learning_rate, iterations, error_threshold)
plt.scatter(xtrain, ytrain)
plt.plot(xtrain, (m * xtrain + c), c='r')
plt.show()
plt.plot(np.arange(len(error_values)),error_values)
plt.ylabel('error')
plt.xlabel('iterations')
plt.show()
# In[8]:


#predction
pred = (m * xtest) + c

#err
mean_squared_error(ytest,pred)


# In[9]:


#predicted vals
plt.scatter(xtest,ytest)
plt.plot(xtest, pred,c='r')


# In[10]:


#reshape to change the shape that is required by the scaler
pred = pred.reshape(-1,1)
xtest = xtest.reshape(-1,1)
ytest = ytest.reshape(-1,1)

xtest_scaled = x_scaler.inverse_transform(xtest)
ytest_scaled = y_scaler.inverse_transform(ytest)
pred_scaled = y_scaler.inverse_transform(pred)

#this is to remove the extra dimension
xtest_scaled = xtest_scaled[:,-1]
ytest_scaled = ytest_scaled[:,-1]
pred_scaled = pred_scaled[:,-1]

p = pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,pred_scaled)),columns=['x','target_y','pred_y'])
p = p.round(decimals = 2)
print(p.head())


# In[ ]:




