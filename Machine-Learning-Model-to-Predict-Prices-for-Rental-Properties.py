#!/usr/bin/env python
# coding: utf-8

# # Regression model to predict the prices of rent.

# In[5]:


import numpy as np   
import pandas as pd    
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt   
import matplotlib.style


# ## Importing The Data

# In[6]:


df = pd.read_excel('Air_BNB.xlsx')


# In[7]:


df.drop('id',axis=1,inplace=True)


# In[8]:


df


# In[9]:


print("The number of colums is",df.shape[1])
print("The number of rows is",df.shape[0])


# In[10]:


df.head()


# In[11]:


df.tail(10)


# In[12]:


df.info()


# In[13]:


round(df.describe(include='all'),2).T


# In[14]:


df['room_type'].nunique()


# In[15]:


for column in df.columns:
    if df[column].dtype == 'object':
        print(column.upper(),':', df['room_type'].nunique())
        print()
        print(df[column].value_counts())
        print()


# In[16]:


df.columns


# In[17]:


df_2 = df.copy()


# In[18]:


df_2.drop_duplicates(inplace=True) # Droping duplicates


# In[19]:


df_2


# In[20]:


df_2.isnull().sum() # checking count to null


# In[21]:


df_2


# In[22]:


df_2 = pd.get_dummies(df_2,columns = ['room_type','cancellation_policy','instant_bookable'],drop_first = True,dtype = float) # Droping irrelevant columns


# In[23]:


df_2


# ## Treating Null values

# #### Using mean

# In[24]:


df_mean = df_2  #Using mean


# In[25]:


for column in df_mean.columns:
    
    
    if df_mean[column].dtype != 'object':
    
    
        mean = df_mean[column].mean()
    
    
        df_mean[column] = df_mean[column].fillna(mean)    
              


# In[26]:


df_mean


# #### Using median

# In[27]:


df_med = df_2.copy()


# In[28]:


for column in df_med.columns:
       
    if df_med[column].dtype != 'object':
       
        median = df_med[column].median()
        
        df_med[column] = df_med[column].fillna(median)    
              


# In[29]:


df_med


# In[30]:


df_med.isnull().sum()


# #### Using KNN imputer

# In[31]:


from sklearn.impute import KNNImputer                                              

imputer = KNNImputer(n_neighbors=5)
df_imputed=imputer.fit_transform(df_2)
df_knn = pd.DataFrame(data = df_imputed,columns=df_2.columns)


# In[32]:


df_knn


# # Outlier Treatment

# In[33]:


cont=df_knn.dtypes[(df_knn.dtypes!='uint8') & (df_knn.dtypes!='bool')].index
plt.figure(figsize=(10,10))
df_knn[cont].boxplot(vert=0)
plt.title('With Outliers',fontsize=16)
plt.show()


# In[34]:


def remove_outlier(col):
    
    sorted(col)
    
    Q1,Q3=np.percentile(col,[25,75])
    
    IQR=Q3-Q1
    
    lower_range= Q1-(1.5 * IQR)
    
    upper_range= Q3+(1.5 * IQR)
    
    return lower_range, upper_range


# In[35]:


remove_outlier(df_mean['review_scores_rating'])


# In[36]:


df_mean_out = df_mean.copy()
df_med_out = df_med.copy()
df_knn_out = df_knn.copy()


# In[37]:


for column in df_mean_out.columns:
    
    lr,ur=remove_outlier(df_mean_out[column])
    
    
    df_mean_out[column]=np.where(df_mean_out[column]>ur,ur,df_mean_out[column])
    
    
    df_mean_out[column]=np.where(df_mean_out[column]<lr,lr,df_mean_out[column])


# In[38]:


plt.figure(figsize=(10,10))
df_knn_out.boxplot(vert=0)
plt.title('Without Outliers',fontsize=16)
plt.show()


# In[44]:


for column in df_med_out.columns:
    
    lr,ur=remove_outlier(df_med_out[column])
    
    
    df_med_out[column]=np.where(df_med_out[column]>ur,ur,df_med_out[column])
    
    
    df_med_out[column]=np.where(df_med_out[column]<lr,lr,df_med_out[column])


# In[51]:


plt.figure(figsize=(10,10))
df_med_out.boxplot(vert=0)
plt.title('Without Outliers',fontsize=16)
plt.show()


# In[45]:


for column in df_knn_out.columns:
    
    lr,ur=remove_outlier(df_knn_out[column])
    
    
    df_knn_out[column]=np.where(df_knn_out[column]>ur,ur,df_knn_out[column])
    
    
    df_knn_out[column]=np.where(df_knn_out[column]<lr,lr,df_knn_out[column])


# In[52]:


plt.figure(figsize=(10,10))
df_knn_out.boxplot(vert=0)
plt.title('Without Outliers',fontsize=16)
plt.show()


# ## Data Distribution

# In[41]:


plt.figure(figsize=(10,10))
sns.heatmap(df_mean.corr(),annot=True)

plt.show


# In[ ]:


# The correlation between the data points indicates the relevance of each category to homestay prices. By analyzing these correlations, we can identify which factors most significantly impact pricing and refine our regression model to improve its predictive accuracy.


# # Train-Test-Split

# In[57]:


X_mean_out = df_mean_out.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y_mean_out = df_mean_out[['log_price']]


# In[58]:


X.head()


# In[59]:


y.head()


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)


# # Linear Regression Model

# In[61]:


regression_model = LinearRegression()

regression_model.fit(X_train, y_train)


# In[62]:


regression_model.score(X_train, y_train)


# ## Mean - Outlier not treated

# In[63]:


X = df_mean.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = df_mean[['log_price']]


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)


# In[65]:


regression_model.fit(X_train, y_train)


# In[66]:


regression_model.score(X_train, y_train)


# ## Med - Outler Treated

# In[67]:


X = df_med_out.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = df_med_out[['log_price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)

regression_model.fit(X_train, y_train)

regression_model.score(X_train, y_train)


# In[68]:


X = df_med.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = df_med[['log_price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)

regression_model.fit(X_train, y_train)

regression_model.score(X_train, y_train)


# In[69]:


X = df_knn_out.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = df_knn_out[['log_price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)

regression_model.fit(X_train, y_train)

regression_model.score(X_train, y_train)


# In[70]:


X = df_knn.drop('log_price', axis=1)

# Copy target into the y dataframe. 
y = df_knn[['log_price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=1)

regression_model.fit(X_train, y_train)

regression_model.score(X_train, y_train)


# In[71]:


df_2


# In[72]:


data_train = pd.concat([X_train, y_train], axis=1)
data_test=pd.concat([X_test,y_test],axis=1)
data_train.head()


# In[75]:


for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))


# In[76]:


intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[ ]:




