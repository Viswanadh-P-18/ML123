#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


import seaborn as sns
from sklearn import metrics


# In[3]:


from sklearn.ensemble import RandomForestRegressor


# In[4]:


gold_data=pd.read_csv("gld_price_data.csv")


# In[5]:


gold_data.info()


# In[6]:


gold_data.head()


# In[7]:


gold_data.shape


# In[8]:


gold_data.tail()


# In[9]:


gold_data.describe()


# In[10]:


correlation=gold_data.corr()
plt.plot(correlation)


# In[11]:


print(correlation['GLD'])


# In[12]:


sns.distplot(gold_data['GLD'],color='green')


# In[13]:


x=gold_data.drop(['GLD','Date'],axis=1)
y=gold_data['GLD']


# In[14]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[15]:


regressor=RandomForestRegressor(n_estimators=100)


# In[16]:


regressor.fit(X_train,Y_train)


# In[17]:


Y_predict=regressor.predict(X_test)


# In[18]:


from sklearn.metrics import mean_squared_error


# In[19]:


print("the mean squared error is ",mean_squared_error(Y_test,Y_predict))


# In[20]:


plt.plot(Y_test,color='yellow',label='Actual Value')
plt.plot(Y_predict,color='red',label='predicted value')
plt.xlabel('VALUES count')
plt.ylabel('GLD PRICE')
plt.legend()
plt.show()


# In[21]:


model=LinearRegression()


# In[22]:


model.fit(X_train,Y_train)


# In[23]:


Y_predict=model.predict(X_test)


# In[ ]:




