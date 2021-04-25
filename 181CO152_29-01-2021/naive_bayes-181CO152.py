#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>CS353 Machine Learning Lab</h1>
# <h1 align='center'>Naive Bayes (29/01/21)</h1>
# <h2 align='center'>Shumbul Arifa (181CO152)</h2>

# # Introduction

# *Topic: Performing Naive Bayes Classifier on Iris Dataset.*
# 
# Naive Bayes methods are a set of supervised learning algorithms based on
# applying Bayes’ theorem with a strong assumption that all the predictors are
# independent of each other i.e. the presence of a feature in a class is independent
# of the presence of any other feature in the same class. This is a naive
# assumption that is why these methods are called Naive Bayes methods. Bayes
# theorem states the following relationship in order to find the posterior probability
# of class i.e. the probability of a label and some observed features, P(Y | features).
# 
# **P(Y | features) = P(Y) * P(features | Y) / P(features)**
# 
# Here, P(Y| features) is the posterior probability of class. P(Y) is the prior
# probability of class. P(features | Y) is the likelihood which is the probability of the
# predictor given class. P(features) is the prior probability of the predictor.
# 

# # Dataset

# Iris Dataset is a standard dataset included in scikit learn standard library. 

# ## Importing Python Libraries

# In[1]:


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


# ## Loading dataset

# In[10]:


iris = load_iris()
X, y = load_iris(return_X_y=True)
X[0:5]


# ## Splitting the dataset

# The dataset is split in the ratio of 8:2 for training : test data respectfully, and the random state is set to 20.

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 20)


# In[16]:


y_train


# In[20]:


y_test


# ## Scaling data using StandardScaler

# In[21]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Fitting Model

# In[22]:


from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)


# In[24]:


y_pred = nvclassifier.predict(X_test)
y_pred


# ## Accuracy

# In[25]:


print("Accuracy score of Naive Bayes Model: ",  nvclassifier.score(X_test, y_test))


# ## Confusion Matrix

# In[28]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_pred))


# In[29]:


print(confusion_matrix(y_test,y_pred))


# <h1 align='center'> Observation </h1>

# **Accuracy score of Naive Bayes Model is 93.33%.**
