#!/usr/bin/env python
# coding: utf-8

# In[26]:


#covid-19(prediction of age vs corona confirmed)


# In[27]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[28]:


data = pd.read_csv("AgeGroupDetails.csv",header=0)
data.head(10)


# In[29]:


plt.scatter(data['Sno'],data['TotalCases'])
plt.xlabel("age of people")
plt.ylabel("total confirmed cases")
plt.title("corona with age pridiction")


# In[30]:


data_n = data.values
m = data_n[:,0].size
x = data_n[:,0].reshape(m,1)
y = data_n[:,2].reshape(m,1)
xTrain , xTest , yTrain , yTest = train_test_split(x,y,test_size = 0.2,random_state = 0)


# In[31]:


linearRegressor = LinearRegression()
linearRegressor.fit(xTrain, yTrain)


# In[32]:


yprediction = linearRegressor.predict(xTest)


# In[33]:


yprediction


# In[34]:


import matplotlib.pyplot as plot
plot.scatter(xTrain, yTrain, color = 'red')
plot.plot(xTrain,linearRegressor.predict(xTrain),color = 'blue')
plot.title('age vs corona confirmed')
plot.xlabel('Age')
plot.ylabel('confirmed cases')
plot.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




