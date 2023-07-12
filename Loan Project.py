#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt,seaborn as sns
import math


# In[2]:


# Read the data from the Excel file
data = pd.read_csv('C:/Users/consumer/Desktop/ARMAN/Loan Application Data.csv')


# In[3]:


data.shape


# In[4]:


data.dtypes
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
data.dtypes


# In[5]:


data.info()


# In[6]:


data.describe()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
data.describe()


# In[7]:


Null = pd.DataFrame(data.isnull().mean().round(4)*100, columns = ['missing_values_perc']).sort_values(by=['missing_values_perc'])


# In[8]:


print(Null).sor_values(by['missing_values_perc'])
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)


# In[9]:


print(str(round(100.0 * Null[Null['missing_values_perc']==0].count()/len(Null),2)) + "%" + " Columns have no missing value")
print(str(round(100.0 * Null[(Null['missing_values_perc']>0) & (Null['missing_values_perc']<25)].count()/len(Null),2)) + "%" + " Columns have missing values between 0-25%")
print(str(round(100.0 * Null[(Null['missing_values_perc']>25) & (Null['missing_values_perc']<50)].count()/len(Null),2)) + "%" + " Columns have missing values between 25-50%")
print(str(round(100.0 * Null[Null['missing_values_perc']>50].count()/len(Null),2)) + "%" + " Columns have missing values more than 50%")


# Dropping The Columns With High Percentage of Missing Values

# In[10]:


# Dropping columns having more than 50% null values
Data = data.drop(data.columns[data.apply(lambda col: (col.isnull().sum() / len(data)) > 0.5)], axis=1)
print(Data.columns)


# In[11]:


Data.shape


# In[12]:


round(100.0 * Data.isnull().sum()/len(Data), 2).sort_values()


# Imputation

# In[13]:


sns.boxplot(Data['AMT_REQ_CREDIT_BUREAU_YEAR'])


# To Reshape Boxplot in Identical form as it is affected by Outliers

# In[14]:


#Lets Understand Value Records
Data['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts()


# tHE Summary shows that there are only few records with 0 value. It expain the reason for disorted Boxplots
# For AMT_REQ_CREDIT_BUREAU_DAY, WE HAVE only 2 approaches:
#     1. Exclude Missing Values
#     2. Impute the Column with values 0 Which is present in more than 99% of the rows

# In[15]:


#Number of Missing values in AMT_REQ_CREDIT_BUREAU_YEAR
Data['AMT_REQ_CREDIT_BUREAU_YEAR'].isnull().sum()


# In[16]:


# Calculate perc of Each Record in the Dataframe
Data['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts(normalize=True)*100


# In[17]:


Data['AMT_REQ_CREDIT_BUREAU_YEAR'].describe()


# In[19]:


sns.boxplot(Data['AMT_REQ_CREDIT_BUREAU_DAY'])


# In[20]:


Data['AMT_REQ_CREDIT_BUREAU_DAY'].isnull().sum()


# In[21]:


# Calculate perc of Each Record in the Dataframe
Data['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts(normalize=True)*100


# In[22]:


Data['AMT_REQ_CREDIT_BUREAU_DAY'].describe()


# The Summary shows that there are only few records with 0 value. It expain the reason for disorted Boxplots

# For AMT_REQ_CREDIT_BUREAU_DAY, WE HAVE only 2 approaches:
#     1. Exclude Missing Values
#     2. Impute the Column with values 0 Which is present in more than 99% of the rows

# Now Lets Fix the Datatypes 

# In[23]:


#Checking the Data types to Identify the Data type
Data.dtypes


# # Now Let's Identify Outliers in Numeric Columns

# For Outlier Analysis we'll Consider Following Columns:
# 'AMT_CREDIT',
# 'AMT_ANNUITY',
# 'AMT_GOODS_PRICE
# 'APARTMENTS_AVG',
# 'FLOORSMAX_AVG',
# 'LIVINGAREA_AVG',
#    

# In[24]:


sns.boxplot(Data['AMT_CREDIT'])


# In[25]:


#Checking Column Statistics
Data['AMT_CREDIT'].describe()


# In[26]:


#Max Value For Boxplots 
IQR_AMT_CREDIT = Data['AMT_CREDIT'].quantile(0.75) - Data['AMT_CREDIT'].quantile(0.25)
Upper_Limit_IQR_AMT_CREDIT = Data['AMT_CREDIT'].quantile(0.75) + IQR_AMT_CREDIT*1.5
print(Upper_Limit_IQR_AMT_CREDIT)
                              


# In[27]:


round(100.0* len(Data[Data['AMT_CREDIT']> Upper_Limit_IQR_AMT_CREDIT])/len(Data),2)


# The % of Outliers is 2.13%

# In[28]:


sns.boxplot(Data['AMT_GOODS_PRICE'])


# In[29]:


#Checking Column Statistics
Data['AMT_GOODS_PRICE'].describe()


# In[30]:


#Max Value For Boxplots 
IQR_AMT_CREDIT = Data['AMT_GOODS_PRICE'].quantile(0.75) - Data['AMT_GOODS_PRICE'].quantile(0.25)
Upper_Limit_IQR_AMT_CREDIT = Data['AMT_GOODS_PRICE'].quantile(0.75) + IQR_AMT_CREDIT*1.5
print(Upper_Limit_IQR_AMT_CREDIT)


# In[31]:


round(100.0* len(Data[Data['AMT_GOODS_PRICE']> Upper_Limit_IQR_AMT_CREDIT])/len(Data),2)


# The % of Outliers is 4.79%

# In[32]:


sns.boxplot(Data['AMT_ANNUITY'])


# In[33]:


# Checking Column Statistics
Data['AMT_ANNUITY'].describe()


# In[34]:


# Max Value For Boxplots
IQR_AMT_ANNUITY = Data['AMT_ANNUITY'].quantile(0.75) - Data['AMT_ANNUITY'].quantile(0.25)
Upper_Limit_IQR_AMT_ANNUITY = Data['AMT_ANNUITY'].quantile(0.75) + IQR_AMT_ANNUITY * 1.5
print(Upper_Limit_IQR_AMT_ANNUITY)


# In[35]:


round(100.0 * len(Data[Data['AMT_ANNUITY'] > Upper_Limit_IQR_AMT_ANNUITY]) / len(Data), 2)


# The % of Outliers for ANT_ANNUITY is 2.44%

# In[36]:


# Column 'FLOORSMAX_AVG'
sns.boxplot(Data['FLOORSMAX_AVG'])


# In[37]:


Data['FLOORSMAX_AVG'].describe()


# In[38]:


IQR_FLOORSMAX_AVG = Data['FLOORSMAX_AVG'].quantile(0.75) - Data['FLOORSMAX_AVG'].quantile(0.25)
Upper_Limit_IQR_FLOORSMAX_AVG = Data['FLOORSMAX_AVG'].quantile(0.75) + IQR_FLOORSMAX_AVG * 1.5
print("Upper Limit for FLOORSMAX_AVG:", Upper_Limit_IQR_FLOORSMAX_AVG)


# In[39]:


percentage_above_upper_limit_FLOORSMAX_AVG = round(100.0 * len(Data[Data['FLOORSMAX_AVG'] > Upper_Limit_IQR_FLOORSMAX_AVG]) / len(Data), 2)
print("The % of Outliers for FLOORSMAX_AVG:", percentage_above_upper_limit_FLOORSMAX_AVG)


# In[40]:


# Column 'YEARS_BEGINEXPLUATATION_AVG'
sns.boxplot(Data['YEARS_BEGINEXPLUATATION_AVG'])


# In[41]:


Data['YEARS_BEGINEXPLUATATION_AVG'].describe()


# In[42]:


IQR_YEARS_BEGINEXPLUATATION_AVG = Data['YEARS_BEGINEXPLUATATION_AVG'].quantile(0.75) - Data['YEARS_BEGINEXPLUATATION_AVG'].quantile(0.25)
Upper_Limit_IQR_YEARS_BEGINEXPLUATATION_AVG = Data['YEARS_BEGINEXPLUATATION_AVG'].quantile(0.75) + IQR_YEARS_BEGINEXPLUATATION_AVG * 1.5
print("Upper Limit for YEARS_BEGINEXPLUATATION_AVG:", Upper_Limit_IQR_YEARS_BEGINEXPLUATATION_AVG)


# In[43]:


percentage_above_upper_limit_YEARS_BEGINEXPLUATATION_AVG = round(100.0 * len(Data[Data['YEARS_BEGINEXPLUATATION_AVG'] > Upper_Limit_IQR_YEARS_BEGINEXPLUATATION_AVG]) / len(Data), 2)
print("Percentage of Outliers for YEARS_BEGINEXPLUATATION_AVG:", percentage_above_upper_limit_YEARS_BEGINEXPLUATATION_AVG)


# In[44]:


print("There are No Outliers")


# Binning of continuous Variable

# For the binning, we will use following columns:
# 1. AGE_GROUP
# 2. AMT_CATEGORY

# In[45]:


# 1. DAYS_BIRTH column can be binned 0-10,10-20,20-30,30-40, 40-50 and so on
Data['AGE_GROUP'] = pd.cut(x=Data.DAYS_BIRTH, bins=[0,19,29, 39, 49, 59,69,79,89], labels=['10s','20s', '30s', '40s' ,'50s', '60s','70s', '80s'])


# In[46]:


# 2. AMT_INCOME_TOTAL column can be binned 'Low','Average', 'Good', 'Best' ,'High', 'Very High'
Data['AMT_CATEGORY'] = pd.cut(x=Data.AMT_INCOME_TOTAL, bins=[0,100000, 200000, 300000, 400000, 500000, 600000], labels=['Low','Average', 'Good', 'Best' ,'High', 'Very High'])


# ANALYSIS

# For further analysis, we will remove irrelevant columns and continue analysis with a few selected columns

# In[ ]:


# list of columns to be dropped
drop_columns = ['FLAG_CONT_MOBILE',
                'FLAG_MOBIL',
                'FLAG_EMP_PHONE',
                'FLAG_WORK_PHONE',
                'FLAG_PHONE',
                'FLAG_EMAIL',
                'HOUR_APPR_PROCESS_START',
                'WEEKDAY_APPR_PROCESS_START',
                'FLOORSMAX_AVG',
                'EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'FLOORSMAX_AVG',
                'FLOORSMAX_MODE',
                'FLOORSMAX_MEDI',
                'TOTALAREA_MODE',
                'EMERGENCYSTATE_MODE',
                'REGION_POPULATION_RELATIVE',
                'YEARS_BEGINEXPLUATATION_AVG',
                'YEARS_BEGINEXPLUATATION_MEDI',
                'YEARS_BEGINEXPLUATATION_MODE',
                'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY',
                'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_3',
                'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11',
                'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21']


# In[47]:


Ndata = Data.drop(columns = ['FLAG_CONT_MOBILE',
                'FLAG_MOBIL',
                'FLAG_EMP_PHONE',
                'FLAG_WORK_PHONE',
                'FLAG_PHONE',
                'FLAG_EMAIL',
                'HOUR_APPR_PROCESS_START',
                'WEEKDAY_APPR_PROCESS_START',
                'FLOORSMAX_AVG',
                'EXT_SOURCE_2',
                'EXT_SOURCE_3',
                'FLOORSMAX_AVG',
                'FLOORSMAX_MODE',
                'FLOORSMAX_MEDI',
                'TOTALAREA_MODE',
                'EMERGENCYSTATE_MODE',
                'REGION_POPULATION_RELATIVE',
                'YEARS_BEGINEXPLUATATION_AVG',
                'YEARS_BEGINEXPLUATATION_MEDI',
                'YEARS_BEGINEXPLUATATION_MODE',
                'REG_REGION_NOT_LIVE_REGION',
                'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION',
                'REG_CITY_NOT_LIVE_CITY',
                'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY',
                'FLAG_DOCUMENT_2',
                'FLAG_DOCUMENT_3',
                'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5',
                'FLAG_DOCUMENT_6',
                'FLAG_DOCUMENT_7',
                'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9',
                'FLAG_DOCUMENT_10',
                'FLAG_DOCUMENT_11',
                'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13',
                'FLAG_DOCUMENT_14',
                'FLAG_DOCUMENT_15',
                'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17',
                'FLAG_DOCUMENT_18',
                'FLAG_DOCUMENT_19',
                'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21'],axis = 1)


# In[48]:


# looking at the columns with missing value in remaining dataframe
round(100.0 * Ndata.isnull().sum()/len(Ndata), 2).sort_values()


# In[49]:


Ndata.info()


# # Checking imbalance in data

# In[50]:


# Finding % of people with outstanding dues and no outstanding dues.

target_0_percentage = (Ndata['TARGET'].value_counts(normalize=True)[0]) * 100
print("Target_0_percentage:", round(target_0_percentage, 2), "%")

target_1_percentage = (Ndata['TARGET'].value_counts(normalize=True)[1]) * 100
print("Target_1_percentage:", round(target_1_percentage, 2), "%")


# As the percentage of Target =0 and Target =1 are different, there is an imbalance

#  Creation of two data sets - one for each Target = 1 and Target = 0

# In[ ]:


# Creating Dataframe of the non-defaulters  
target_0_df = Ndata[Ndata['TARGET'] == 0]
target_0_df


# In[ ]:


target_0_df.shape


# Creating Target_1_df, people having outstanding dues

# In[ ]:


# Creating Dataframe of the defaulters
target_1_df = app_df_2.query('TARGET=="1"')
target_1_df


# In[ ]:


target_1_df.shape


# In[ ]:


# Checking unique values in each columns
Ndata.nunique().sort_values()


# In[ ]:


# Cheking column types
Ndata.dtypes


# Any column which is either of object type or have less than 40 values is considered categorical. Remaining columns of type float or int will be considered numerical

# In[ ]:


#list of all categorical columns
categorical_columns = ['NAME_CONTRACT_TYPE',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'CODE_GENDER',
                       'NAME_EDUCATION_TYPE',
                       'AMT_CATEGORY',
                       'AGE_GROUP',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE',                       
                       'REGION_RATING_CLIENT_W_CITY',
                       'REGION_RATING_CLIENT',
                       'AMT_REQ_CREDIT_BUREAU_HOUR',
                       'DEF_60_CNT_SOCIAL_CIRCLE',
                       'AMT_REQ_CREDIT_BUREAU_WEEK',
                       'AMT_REQ_CREDIT_BUREAU_DAY',
                       'DEF_30_CNT_SOCIAL_CIRCLE',
                       'AMT_REQ_CREDIT_BUREAU_QRT',
                       'CNT_CHILDREN',
                       'CNT_FAM_MEMBERS',
                       'AMT_REQ_CREDIT_BUREAU_MON',
                       'AMT_REQ_CREDIT_BUREAU_YEAR',
                       'OBS_30_CNT_SOCIAL_CIRCLE',
                       'OBS_60_CNT_SOCIAL_CIRCLE',
                      ]


# In[ ]:


# list of all continuous numerical column
numerical_columns= ['AMT_GOODS_PRICE',
                    'DAYS_LAST_PHONE_CHANGE',
                    'DAYS_ID_PUBLISH',
                    'AMT_INCOME_TOTAL',
                    'DAYS_EMPLOYED',
                    'DAYS_REGISTRATION',
                    'DAYS_BIRTH',
                    'AMT_CREDIT',
                    'AMT_ANNUITY'
                   ]


# # Univariate Analysis for categorical variable

# Under univariate analysis, we will look at percentage distribution of values of categorial variable

# In[ ]:


#loop for performing univariate analysis
for i in categorical_columns:
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    target_0_df[i].value_counts(normalize=True).plot.bar()
    plt.title(i+ '- Target = 0')
    plt.subplot(1,2,2)
    target_1_df[i].value_counts(normalize=True).plot.bar()
    plt.title(i+ '- Target = 1')
    


# The above chart shows the distribution of customers across categorical variable for both Target = 0 and Target = 1

# # Correlation for numerical columns

# Correlation for numerical columns

# In[ ]:


# correlation analysis for the entire Target data
plt.figure(figsize=(20,10))
sns.heatmap(Ndata[numerical_columns].corr(),annot=True)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.yticks(rotation = 0)
plt.show()


# In[ ]:


# correlation analysis for Target = 0
plt.figure(figsize=(20,10))
sns.heatmap(target_0_df[numerical_columns].corr(),annot=True)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.yticks(rotation = 0)
plt.show()


# In[ ]:


#correlation analysis for target =1 
plt.figure(figsize=(20,10))
sns.heatmap(target_1_df[numerical_columns].corr(),annot=True)
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.yticks(rotation = 0)
plt.show()


# In[ ]:


plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sns.heatmap(target_0_df[numerical_columns].corr(),annot=True)
plt.title('heatmap- Target = 0')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.yticks(rotation = 0)
plt.subplot(1,2,2)
sns.heatmap(target_1_df[numerical_columns].corr(),annot=True,)
plt.title('heatmap- Target = 1')
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.yticks(rotation = 0)
plt.show()


# Through the heatmap we can see same set of columns seem to have a high correlation across all three data sets. Top correlate colums are:
# 1. AMT_GOOD_PRICE vs AMT_CREDIT
# 2. AMT_GOOD_PRICE vs AMT_ANNUITY
# 3. AMT_CREDIT_AMT_ANNUITY

# # 4.5 Checking if Variables with highest coeffecient are same in both file

#  This analysis is conducted to understand if top 10 high correlation variables are common across for both data - target =0 and target =1

# In[ ]:


# Correlation for numberical columns for Target = 0
corr = target_0_df.corr()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf0 = corr.unstack().reset_index()
corrdf0.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf0.dropna(subset = ['Correlation'], inplace = True)
corrdf0['Correlation'] = round(corrdf0['Correlation'], 2)
# Since we see correlation as an absolute value, we are converting it into absolute value
corrdf0['Correlation_abs'] = corrdf0['Correlation'].abs()
corrdf0.sort_values(by = 'Correlation_abs', ascending = False).head(10)


# In[ ]:


# Correlation for numberical columns for Target = 1
corr = target_1_df.corr()

corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
corrdf0 = corr.unstack().reset_index()
corrdf0.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf0.dropna(subset = ['Correlation'], inplace = True)
corrdf0['Correlation'] = round(corrdf0['Correlation'], 2)
# Since we see correlation as an absolute value, we are converting it into absolute value
corrdf0['Correlation_abs'] = corrdf0['Correlation'].abs()
corrdf0.sort_values(by = 'Correlation_abs', ascending = False).head(10)


# #### 8 out of top 10 pair of high correlated variables are same for both 0 and 1

# ### Univariate Analysis for Numerical Values

# In[ ]:


for i in numerical_columns:
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    sns.boxplot(target_0_df[i])
    plt.title(i+ '- Target = 0')
    plt.subplot(1,2,2)
    sns.boxplot(target_1_df[i])
    plt.title(i+ '- Target = 1')


# #### Key interpretation from univariate analysis of numerical variables
# ##### In this section, we will only highlight variables having significant difference for target = 0 and target = 1  
# 
# - **DAYS_LAST_PHONE_CHANGE**: Median value and 75 percentile value for Defaulters (Target = 1) is lesser than non-defaulters. It implies defaulter more often change phone number before application
# - **DAYS_ID_PUBLISH**: Defaulters seem to change IDs more frequetly than non-defaulters
# - **DAYS_BIRTH**: 25 percentile, median, and 75 percentile for the age of defaulter applicants are smaller than younger applicants. This means defaulter population is younger than non-defaulter.

# ### Bivariate Analysis for Categorical Variable

# #### For bi variate analysis wrt target-0&1, we will use the data set where full target column is available. Further, we will calculate mean of target column for each categorical variable as it will tell us percentage of defaulters for the category 

# In[ ]:


for i in categorical_columns:
    (app_df_2.groupby(i)['TARGET'].mean()).plot.bar()
    plt.title(i+ 'vs' +'Target')
    plt.show()


# #### Key interpretation from bivariate analysis of categorical variables
# ##### In this section, we will only highlight key outcomes from Bivariate analysis
# - **CODE_GENDER**: Male customers have a higher probability of defaulting
# - **NAME_EDUCATION_TYPE**:Customers with lower secondary education have a higher risk of default
# - **AGE_GROUP**- Customers in 20s and 30s have higher chances of deafaulting
# - **NAME_HOUSING_TYPE**: Customers living in rented apartments and living with parents seem to default more
# - **NAME_INCOME_TYPE**:Unemployed and Customers on maternity leave have higher 
# - **OCCUPATION_TYPE**:Low-skill laborers default more
# - **REGION_RATING_CLIENT**&**REGION_RATING_CLIENT_W_CITY**: Customers with rating 3 have higher risk of defaulting    

# ## Bivariate Analsyis for Numerical columns

# #### The objective of this analysis is to find pattern in defaulter vs non-defaulter customers wrt target

# In[ ]:


# loop for bivariate analysis for numerical variables
for i in numerical_columns:
    sns.boxplot(data = app_df_2, x='TARGET',y= i)
    plt.show()


# #### Key interpretation from bivariate analysis of categorical variables
# ##### In this section, we will only highlight key outcomes from Bivariate analysis
# - **DAYS_LAST_PHONE_CHANGE**:Defaulter customers change phone closer to the submission of application
# - **DAYSID_PUBLISH**:Defaulter customers changes id closer to submission of application
# - **DAYS_REGISTRATION**: Defaulter customers changes registration on a date closer to submission of application
# - **DAYS_BIRTH**: Defaulter customers are relatively younger than non-defaulters

# In[ ]:




