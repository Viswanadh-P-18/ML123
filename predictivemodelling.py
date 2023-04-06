#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[3]:


sonar_data=pd.read_csv("Copy of sonar data3.csv")


# In[4]:


sonar_data.describe()


# In[5]:


sonar_data.shape


# In[7]:


sonar_data['60'].value_counts()


# In[8]:


sonar_data.groupby('60').mean()


# # separating features and labels

# In[10]:


x=sonar_data.drop(columns='60',axis=1)
y=sonar_data['60']


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=1,stratify=y)


# In[12]:


print(x_train.shape,x_test.shape,x.shape)


# In[13]:


print(y_train.shape,y_test.shape,y.shape)


# In[14]:


model=LogisticRegression()


# # training the model

# In[15]:


model.fit(x_train,y_train)


# In[16]:


x_train_predict=model.predict(x_train)
accuracy=accuracy_score(x_train_predict,y_train)
print("the accuracy is",accuracy)


# In[17]:


x_test_predict=model.predict(x_test)
accuracy_test=accuracy_score(x_test_predict,y_test)
print(accuracy_test)


# In[19]:


input_data=(0.0453,0.0523,0.0843,0.0689,0.1183,0.2583,0.2156,0.3481,0.3337,0.2872,0.4918,0.6552,0.6919,0.7797,0.7464,0.9444,1.0000,0.8874,0.8024,0.7818,0.5212,0.4052,0.3957,0.3914,0.3250,0.3200,0.3271,0.2767,0.4423,0.2028,0.3788,0.2947,0.1984,0.2341,0.1306,0.4182,0.3835,0.1057,0.1840,0.1970,0.1674,0.0583,0.1401,0.1628,0.0621,0.0203,0.0530,0.0742,0.0409,0.0061,0.0125,0.0084,0.0089,0.0048,0.0094,0.0191,0.0140,0.0049,0.0052,0.0044)
input_data_array=np.asarray(input_data)
input_data_reshape=input_data_array.reshape(1,-1)
prediction=model.predict(input_data_reshape)

if prediction=='R':
    print("the obj is a rock")
else:
    print("the object is a mine")
    


# In[ ]:




