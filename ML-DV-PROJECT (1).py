#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install --user sklearn


# In[24]:


import pandas as pd
insurance = pd.read_csv("C:/Users/Anukriti/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Anaconda3 (64-bit)/insurance.csv")
insurance.head()


# In[25]:


insurance[['sex','smoker','region']].head()


# In[26]:


# Replacing string values to numbers
insurance['sex'] = insurance['sex'].apply({'male':0,      'female':1}.get) 
insurance['smoker'] = insurance['smoker'].apply({'yes':1, 'no':0}.get)
insurance['region'] = insurance['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[27]:


insurance.head()


# In[28]:


import seaborn as sns
# Correlation betweeen 'charges' and 'age' 
sns.jointplot(x=insurance['age'],y=insurance['charges'])


# In[29]:


# Correlation betweeen 'charges' and 'smoker' 
sns.jointplot(x=insurance['smoker'],y=insurance['charges'])


# In[30]:


insurance.columns


# In[31]:


# features
X = insurance[['age', 'sex', 'bmi', 'children','smoker','region']]
# predicted variable
Y = insurance['charges']


# In[32]:


X.head()


# In[33]:


Y.head()


# In[34]:


# importing train_test_split model
from sklearn.model_selection import train_test_split
# splitting train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)


# In[35]:


len(X_test) # 402
len(X_train) # 936
len(insurance) # 1338


# In[36]:


# importing the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Fit linear model by passing training dataset
model.fit(X_train,Y_train)


# In[37]:


# Predicting the target variable for test datset
predictions = model.predict(X_test)


# In[38]:


predictions[0:5]


# In[39]:


import matplotlib.pyplot as plt
plt.scatter(Y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[40]:


# Predict charges for new customer : Name- Frank
data = {'age' : 40,'sex' : 1,'bmi' : 45.50,'children' : 4,'smoker' : 1,'region' : 3}
index = [1]
frank_df = pd.DataFrame(data,index)
frank_df


# In[41]:


prediction_frank = model.predict(frank_df)
print("Medical Insurance cost for Frank is : ",prediction_frank)


# In[42]:


# Predict charges for new customer : Name- Frank
data = {'age' : 23,'sex' : 0,'bmi' : 40.5,'children' : 0,'smoker' : 0,'region' : 1}
index = [1]
tori_df = pd.DataFrame(data,index)
tori_df


# In[43]:


prediction_tori = model.predict(tori_df)
print("Medical Insurance cost for Tori is : ",prediction_tori)

