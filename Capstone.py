#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from nltk.corpus import stopwords


# In[10]:


dataemp = pd.read_csv('C:/Users/jasmi/Documents/Capstone/employee_reviews.csv', sep = ",")


# In[11]:


dataemp.head


# In[9]:


dataemp.shape


# In[ ]:




