
# coding: utf-8

# In[1]:


import datetime as dt
import time as tm


# In[5]:


dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow


# In[6]:


dtnow.day, dtnow.month, dtnow.year, dtnow.second, dtnow.minute


# In[8]:


delta = dt.timedelta(days = 100)
delta


# In[11]:


today = dt.date.today()
today


# In[12]:


today - delta


# In[13]:


today+delta

