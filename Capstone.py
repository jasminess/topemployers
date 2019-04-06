#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score 
from time import time


# In[2]:


dataemp = pd.read_csv('employee_reviews.csv', sep = ",", index_col =0)


# In[3]:


dataemp.head()


# In[4]:


dataemp.shape


# In[5]:


dataemp.info()


# In[6]:


dataemp.drop(columns = [ 'location', 'advice-to-mgmt', 'job-title', 'helpful-count', 'link'], inplace = True)


# In[7]:


dataemp.describe()


# In[8]:


dataemp['work-balance-stars'] = pd.to_numeric(dataemp['work-balance-stars'].replace('none', np.nan))
dataemp['culture-values-stars'] = pd.to_numeric(dataemp['culture-values-stars'].replace('none', np.nan))
dataemp['carrer-opportunities-stars'] = pd.to_numeric(dataemp['carrer-opportunities-stars'].replace('none', np.nan))
dataemp['comp-benefit-stars'] = pd.to_numeric(dataemp['comp-benefit-stars'].replace('none', np.nan))
dataemp['senior-mangemnet-stars'] = pd.to_numeric(dataemp['senior-mangemnet-stars'].replace('none', np.nan))


# In[9]:


dataemp.describe()


# In[ ]:





# In[10]:



bplot = sns.boxplot(y='overall-ratings', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Overall Ratings")


#           Overall Ratings by Company

# In[11]:


bplot = sns.boxplot(y='overall-ratings', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Overall Ratings")


#           Work-Balance Rating by Company

# In[12]:


bplot = sns.boxplot(y='work-balance-stars', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Work-Balance ratings")


# In[13]:


bplot = sns.boxplot(y='culture-values-stars', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Culture Values")


#           Culture Values by Company

# In[14]:


bplot = sns.boxplot(y='carrer-opportunities-stars', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Career Opportunities ratings")


# In[15]:


bplot = sns.boxplot(y='comp-benefit-stars', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Compensation Benefit Ratings")


#      Compensation Benefits by Company

# In[16]:


bplot = sns.boxplot(y='senior-mangemnet-stars', x='company',  data=dataemp, width=0.5,palette="colorblind").set_ylabel("Senior Management Ratings")


#      Senior Management ratings by Company

# # On comparing the ratings with boxplot, it is observed that the companies Facebook and Google are leading at the top

# In[17]:


dataemp.corr()


# In[18]:


sns.heatmap(dataemp.corr(), annot=True ,fmt=".2f")
plt.show()


# # As per correlation index, it is seen that work culture and senior management ratings are highly correlated to the overall ratings of the company

# In[19]:


meanval = dataemp['overall-ratings'].mean()


# # Classifying labels  - 1 - Satisfied happy employee 0- Employee is not satisfied with job. Greater than mean of overall-rating is considered as satisifed and less than mean of overall-rating is considered as unsatisfied for classification

# In[20]:


dataemp['label'] = dataemp['overall-ratings'].apply(lambda x: 1 if x > meanval  else 0)


# In[21]:


dataemp.head()


# In[22]:


pd.value_counts(dataemp['label']).plot.bar()
plt.show()


# In[23]:


def datatext_preprocess(total_text):
    removepunc = [char for char in total_text if char not in string.punctuation]    
    removepunc = ''.join(removepunc)
    re.sub('[^A-Za-z]+', '', removepunc)
    return ' '.join([word for word in removepunc.split() if word.lower() not in stopwords.words('english')])
        


# In[ ]:





# In[24]:


dataemp.info()


# In[25]:


dataemp['summary']=dataemp['summary'].fillna("")
dataemp['pros']=dataemp['pros'].fillna("")
dataemp['cons']=dataemp['cons'].fillna("")


# In[26]:


dataemp['summary'] = dataemp['summary'].apply(datatext_preprocess)
dataemp['pros'] = dataemp['pros'].apply(datatext_preprocess)
dataemp['cons'] = dataemp['cons'].apply(datatext_preprocess)


# After text preprocessing

# In[27]:


dataemp.info()


# In[28]:


dataemp['summary'].head()


# In[29]:


labeltext = dataemp.groupby("label")


# In[30]:


labeltext.describe()


# In[ ]:





# In[31]:



##wordcloud = WordCloud().generate(dataemp.reviews[0])


# Combing all the text data to be used as features for predicting employee sentiment on job satisfaction

# In[32]:


dataemp["reviews"] = dataemp["summary"] +' ' + dataemp["pros"]+' ' +dataemp["cons"]


# In[33]:


Satisfiedemp = dataemp.loc[dataemp["label"]==1]
UnSatisfiedemp = dataemp.loc[dataemp["label"]==0]


# In[34]:


Satisfiedemp.shape


# In[67]:


ignorewords = Satisfiedemp["pros"].isin(['Amazon','company','work', 'place', 'employee', 'team', 'time'])
satwords = Satisfiedemp.loc[~(ignorewords), "pros"]


# In[68]:


wordcloud = WordCloud().generate(' '.join(satwords))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[71]:


ignorewords = UnSatisfiedemp["cons"].isin(['management','manager','employee', 'Amazon', 'customer', 'team', 'time', 'job', 'people',])
unsatwords = UnSatisfiedemp.loc[~(ignorewords), "cons"]


# In[72]:


wordcloud = WordCloud().generate(' '.join(unsatwords))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()


# In[39]:


X = dataemp["reviews"]
y = dataemp["label"]


# In[ ]:





# Vectorizing the text data 

# In[40]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# Feeding the output of vectors into TFIDF transformer

# In[41]:



tfidf = TfidfTransformer()
tfidf.fit_transform(X)


# Splitting the training and test data in the ratio of 70% training data and 30% test data

# In[42]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 101)


# Naive Bayes Classification model

# In[43]:



nb = MultinomialNB()


# In[44]:


nb.fit(X_train, y_train)


# In[45]:


predictions = nb.predict(X_test)


# In[46]:


print(confusion_matrix(y_test, predictions))


# Accuracy about 77% with Naive Bayes Model

# In[47]:


print(classification_report(y_test, predictions))


# Random Forest Classification Model

# In[48]:



RFmodel = RandomForestClassifier(n_estimators=100)


# In[49]:


RFmodel.fit(X_train, y_train)


# In[50]:


predictions = RFmodel.predict(X_test)


# In[51]:


print(confusion_matrix(y_test, predictions))


# Accuracy for Random Forest is same as Naive Bayes around 76%

# In[52]:


print(classification_report(y_test, predictions))


# In[58]:





names = ["Random Forest Classifier","Logistic Regression", "Linear SVC", "LinearSVC with L1-based feature selection","Multinomial NB", 
         "Bernoulli NB", "Ridge Classifier", "AdaBoost", "Perceptron","Passive-Aggresive", "Nearest Centroid"]
classifiers = [
    RandomForestClassifier(n_estimators=100),
    LogisticRegression(solver='liblinear'),
    LinearSVC(max_iter = 100),
    Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 100))),
  ('classification', LinearSVC(penalty="l2"))]),
    MultinomialNB(),
    BernoulliNB(),
    RidgeClassifier(),
    AdaBoostClassifier(),
    Perceptron(max_iter=5, tol=None),
    PassiveAggressiveClassifier(max_iter=5, tol=None),
    NearestCentroid()
    ]
classifierlist = zip(names,classifiers)
vec = CountVectorizer()


# In[59]:


X = dataemp["reviews"]
y = dataemp["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[60]:


def review_summary(pipeline, X_train, X_test, y_train, y_test):
    t0 = time()
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    train_test_time = time() - t0
    accuracy = accuracy_score(y_test, y_pred)
    print ("Accuracy percentage %: ",accuracy*100)
    print ("Duration in seconds: ",train_test_time)
    print ("Classification report:\n")
    print(classification_report(y_test, y_pred))
    print ("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print("*"*80)
    return accuracy*100, train_test_time


# In[61]:


def classifiervectorizer_compare(vectorizer=vec, n_features=10000,  ngram_range=(1, 1), classifier=classifierlist):
    result = []
    for n,c in classifier:
        vec.set_params(stop_words=None, max_features=n_features, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
                ('vectorizer', vec),
                ('classifier', c)
        ])
            
        print ("Classifier: ", n)
        acc,tt_time = review_summary(checker_pipeline, X_train, X_test, y_train, y_test)
        result.append((n, acc ,tt_time))
    return result


# In[62]:


# import warnings filter
from warnings import simplefilter
#from sklearn.utils import ConvergenceWarning
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter('ignore', ConvergenceWarning)

vectorizernames = ["TFIDF vectorizer", "Count Vectorizer"]
vectorizers = [TfidfVectorizer(),CountVectorizer() ]
vectorizerlist = zip(vectorizernames, vectorizers )
tvec = TfidfVectorizer()
cvec = CountVectorizer()
bigram_result = classifiervectorizer_compare(vectorizer=tvec,n_features=100000,ngram_range=(1,2))


# In[ ]:


bigram_result


# On comparing the accuracy and the duration taken for each classifier, Multinomial NB suits best for solving this problem. It has the maximum accuracy of 78.17% and returned in the least duration of all.

# In[ ]:




