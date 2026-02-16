#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append(r"C:\Users\rjbel\Python\Notebooks\Mapua")


# In[2]:


from DataLoaders.settings import Settings
from DataLoaders.revenues import Revenues
from DataLoaders.enrollees import Enrollees
from FeatureEngineering.credit_sales import CreditSales


# In[3]:


setting = Settings()

settings = setting.get_configs()
directory = setting.get_sub_directories()


# In[4]:


revenues = Revenues(directory['revenues_folder'], directory)
df_revenues = revenues.show_data()
df_revenues.reset_index(inplace=True)
df_revenues = df_revenues.rename(columns={'index': 'entry_number'})


# In[5]:


df_revenues


# In[6]:


df_revenues.to_excel(r"Database\revenues_pseudonymized.xlsx", index=False)


# In[7]:


enr = Enrollees(df_revenues)
df_enrollees = enr.show_data()
df_enrollees


# In[8]:


df_enrollees.to_excel(r"Database\enrollees_pseudonymized.xlsx", index=False)


# In[ ]:





# In[ ]:




