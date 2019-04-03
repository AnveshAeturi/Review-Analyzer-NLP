# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:28:00 2019

@author: Anvesh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 00:33:55 2018

@author: Anvesh
"""

#Natural Language Processing

import numpy as np
import scipy as spy
import pandas as pd
#%%
dataset = pd.read_csv('Reviews.tsv', delimiter = '\t', quoting = 3 )

#%%
#cleaning the Text
import re 
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#Stemming 
#It remove the past and future words present int the sentance
from nltk.stem.porter import PorterStemmer
review = re.sub('[^a-zA-z]', ' ', dataset['Review'][0])
#first parameter to keep which values
#second Parameter to assign what to keep after removing extra parameters
review.lower()
review = review.split() 
#.split changes the string to list to perform the word by word exe
ps = PorterStemmer()
#%%
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#now converting the List back to the string
review = ' '.join(review)
#%%
#Now Apply for all the data in dataset
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#%%
#Creating Bags of Words Model 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
#X is the IV value 
Y = dataset.iloc[:,1].values
#%%
# =============================================================================
# Using Navie Bayes Classification
#Accuracy =0.705
#Precition=0.53
#Recall   =0.69
# =============================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


#%%
# =============================================================================
#Decision Tree Classification 
#Accuracy =0.705
#Precition=0.66
#Recall   =0.77 
# =============================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
#%%
# =============================================================================
# Using Random Forest Classification
#Accuracy =0.845
#Precition=0.609
#Recall   =0.845
# =============================================================================
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
