
# coding: utf-8

# In[1]:

import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

pd.set_option('display.max_columns', 500)
local_path = r"C:\Users\Chris\OneDrive\Documents\MIDS\WS231\final"


# # Retrieve Historic Crime Data

# In[2]:

# read in Chicago crime incident data
chgo_crimes = pd.read_csv(r"{0}\chicago_crimes_2001_2017.csv".format(local_path))
chgo_crimes.drop(chgo_crimes[['X Coordinate','Y Coordinate','Location','Updated On']],inplace=True,axis=1)
chgo_crimes.columns = ['id','case_num','date','block','iucr','primary_type','crime_short_desc',
                      'location_desc','is_arrested','is_domestic','beat','district','ward',
                      'community_area','fbi_code','year','latitude','longitude']

chgo_crimes


# # Feature Engineering

# In[3]:

# Datetime features
chgo_crimes['date'] = pd.to_datetime(chgo_crimes['date'])
chgo_crimes['date_time'] = chgo_crimes['date'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour))
chgo_crimes['hour'] = chgo_crimes['date'].dt.hour
chgo_crimes['month'] = chgo_crimes['date'].dt.month

# Crime description feature
chgo_crimes['crime_desc'] = chgo_crimes[['primary_type', 'crime_short_desc']].apply(lambda x: ' - '.join(x), axis=1)

# Geo features
chgo_crimes['short_lat'], chgo_crimes['short_long'] = chgo_crimes['latitude'], chgo_crimes['longitude']
chgo_crimes.round({'short_lat':2, 'short_long':2})
#chgo_crimes['short_geo'] = chgo_crimes[['short_lat', 'short_long']].apply(lambda x: ''.join(str(x)), axis=1)

chgo_crimes


# # District Crime Score

# In[8]:

# Total number of arrests per crime description
arrested_desc = chgo_crimes[chgo_crimes['is_arrested']][['id','crime_desc','district','month','hour']]
arrested_desc = arrested_desc.groupby(['district','month','hour','crime_desc']).id.nunique().reset_index()
arrested_desc.columns = ['district','month','hour','crime_desc','arrested_total']

# Total number of incidents (arrest or non-arrest) per crime description
crime_desc = chgo_crimes[['id','crime_desc','district','month','hour']]
crime_desc = crime_desc.groupby(['district','month','hour','crime_desc']).id.nunique().reset_index()
crime_desc.columns = ['district','month','hour','crime_desc','total']

# Calculate score
chgo_district_scores = pd.merge(crime_desc, arrested_desc, how = 'left')
chgo_district_scores.fillna(0, inplace = True)
chgo_district_scores['raw_intensity'] = chgo_district_scores['arrested_total'] / chgo_district_scores['total']
chgo_district_scores['final_intensity'] = chgo_district_scores['arrested_total'] * chgo_district_scores['raw_intensity']

# Summarize score
chgo_district_scores = chgo_district_scores.groupby(['district','month','hour'])[['final_intensity','total','arrested_total']].sum().reset_index()
chgo_district_scores.sort_values('final_intensity', inplace = True)

# Save pickle file 
chgo_district_scores.to_pickle('district_crime_scores.pkl')

chgo_district_scores


# # Training Set Auditing

# In[10]:

# Load pickle file
# chgo_district_scores = pd.read_pickle('district_crime_scores.pkl')


# In[ ]:



