#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv("/home/jovyan/binder/BreastCancer_data.csv")
data.head(5)


# In[3]:


print (data.shape)


# In[4]:


#Data cleaning
data.drop("Unnamed: 32", axis=1, inplace=True)
print (list(data.columns))


# In[5]:


#To check whether there are any null values
data.isnull().sum()
data.isna().sum()


# In[6]:


data["diagnosis"].value_counts()


# In[7]:


#plotting bar graph of diagnosis column
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[8]:


print ("Response rate: ")
print (data["diagnosis"].value_counts()/data.shape[0]*100)


# In[9]:


sns.countplot(x='diagnosis', data=data, palette='hls')
plt.show()
plt.savefig('count_fig')


# In[10]:


data.groupby('diagnosis').mean()


# In[11]:


data['diagnosis'].replace(["M","B"],["1","0"], inplace=True)


# In[12]:


X=data.iloc[:,2:31]
y=data.iloc[:,1]


# In[13]:


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[14]:


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


#Using Logistic Regression Algorithm to the Training Set

logreg = LogisticRegression(random_state = 0)
logreg.fit(X_train, y_train)


# In[16]:


y_pred=logreg.predict(X_test)
print ("Accuracy on Logistic Regression Classifier on test dataset: ", logreg.score(X_test, y_test))


# In[17]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold= model_selection.KFold(n_splits=10, shuffle=False, random_state=7)
modelCV=LogisticRegression()
scoring='accuracy'
result=model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print ("10-fold cross validation accuracy: ", result.mean())


# In[18]:


#Using confusion matrix to check accuracy
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, y_pred)
print (cm)


# In[24]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr,thresholds=roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label="Logistic Regression (area=%0.2f)" %logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristics")
plt.legends(loc="lower right")
plt.savefig("LOG_Reg")
plt.show()


# In[ ]:




