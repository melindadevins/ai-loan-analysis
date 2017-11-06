
# coding: utf-8

# # Process Data
# 
# This notebook process the input data.
# 
# Reference:  https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/
# 
# It will be converted to python code to be shared with various models

# In[23]:

# Pretty display for notebooks
# get_ipython().magic(u'matplotlib inline')

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
from time import time
import cPickle
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
#import visuals as vs


# ## Load the data from csv file
# for now, you have to manualy unzip the file on your computer before running the following code

# In[24]:


df_orig = pd.read_csv('./data/ny_hmda_2015.csv', low_memory=False, header=0, delimiter=",")



# ## Inspect the data

# In[25]:

# group dataframe by unique combination of columns, and return a dataframe with count as the added last column
# df: the dataframe that contains all columns
# 
def get_count_of_unique_columns(df, list_of_columns):
    df_temp = df[list_of_columns]  #select the columns of list_of_columns
    df_count = df_temp.dropna().groupby(list_of_columns, as_index=False).size().reset_index().rename(columns={0:'count'})
    return df_count


# In[26]:

# Make a copy of original dataframe so that the original is not altered when manipulate the df
df = df_orig.copy()

# to inspect the dada
print ("dataframe head:")
#print(df.head())
display(df.head(n=2))

num_rows = df.shape[0]
num_col = df.shape[1]
# print ("Total number of records: {}".format(num_rows))
# print ("Toatl numver of features: {}".format(num_col))

# print("Display all columns and 3 rows")
# display(df.loc[0:2, 'action_taken':'applicant_income_000s'])
# display(df.loc[0:2, 'applicant_race_1':'applicant_race_name_5'])
# display(df.loc[0:2, 'applicant_sex':'co_applicant_race_name_5'])
# display(df.loc[0:2, 'co_applicant_sex':'edit_status_name'])
# display(df.loc[0:2, 'hoepa_status':'msamd_name'])
               
# display(df.loc[0:2, 'owner_occupancy':'state_name'])
# display(df.loc[0:2, 'hud_median_family_income':'tract_to_msamd_income'])



# ## Inspect Missing Data in Columns with Continous Value
# 

# In[27]:

scale_column_indexer = ('applicant_income_000s', 'hud_median_family_income', 'loan_amount_000s','number_of_1_to_4_family_units',
                'number_of_owner_occupied_units','minority_population','population','tract_to_msamd_income')

#select rows whose scale_column contains missing data
# print("rows whose columns of continuous value contain missing data")
df_missing = df[df.loc[:, scale_column_indexer].isnull().any(axis=1)]
#display(df_missing)

# display(df_missing.loc[:, 'action_taken':'applicant_income_000s'])
# display(df_missing.loc[:, 'applicant_race_1':'applicant_race_name_5'])
# display(df_missing.loc[:, 'applicant_sex':'co_applicant_race_name_5'])
# display(df_missing.loc[:, 'co_applicant_sex':'edit_status_name'])
# display(df_missing.loc[:, 'hoepa_status':'msamd_name'])
               
# display(df_missing.loc[:, 'owner_occupancy':'state_name'])
# display(df_missing.loc[:, 'hud_median_family_income':'tract_to_msamd_income'])


# ## Get Count of Unique Columns (may be used as lookup table)
# It can also be used as a lookup table after delete the xx_name columns

# In[28]:


# df_action_count = get_count_of_unique_columns(df, ['action_taken','action_taken_name'])
# # print("df_action_count")
# # display(df_action_count)

# df_ethnic_count = get_count_of_unique_columns(df, ['applicant_ethnicity', 'applicant_ethnicity_name'])
# print("df_ethnic_count")
# display(df_ethnic_count)

# df_race_count = get_count_of_unique_columns(df, ['applicant_race_1', 'applicant_race_name_1'])
# print("df_race_count")
# display(df_race_count)


# df_gender_count = get_count_of_unique_columns(df, ['applicant_sex', 'applicant_sex_name'])
# print("df_gender_count")
# display(df_gender_count)


# df_agency_count = get_count_of_unique_columns(df, ['agency_code', 'agency_abbr', 'agency_name'])
# print("df_agency_count")
# display(df_agency_count)

# df_count = get_count_of_unique_columns(df, ['hoepa_status', 'hoepa_status_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['lien_status', 'lien_status_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['loan_purpose', 'loan_purpose_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['loan_type', 'loan_type_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['msamd', 'msamd_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['owner_occupancy', 'owner_occupancy_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['preapproval', 'preapproval_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['property_type', 'property_type_name'])
# print("df_count")
# display(df_count)

# df_count = get_count_of_unique_columns(df, ['purchaser_type', 'purchaser_type_name'])
# print("df_count")
# display(df_count)

# #df_count = get_count_of_unique_columns(df, ['respondent_id'])
# #print("df_count")
# #display(df_count)

# #df_count = get_count_of_unique_columns(df, ['sequence_number'])
# #print("df_count")
# #display(df_count)

# df_count = get_count_of_unique_columns(df, ['state_code', 'state_abbr', 'state_name'])
# print("df_count")
# display(df_count)



# #df_count = get_count_of_unique_columns(df, ['rate_spread'])
# #print("df_count")
# #display(df_count)

# df_count = get_count_of_unique_columns(df, ['tract_to_msamd_income'])
# print("df_count")
# display(df_count)


# ### Some Other Helpful Commands 

# In[29]:

# if read from zip file directly ...
#import zipfile
#with zipfile.ZipFile("./data/ny-home-mortgage.zip") as z:
#   with z.open("ny_hmda_2015.csv") as f:
#      train = pd.read_csv(f, low_memory=False, header=0, delimiter="\t")
#      print(train.head())    # print the first 5 rows


# In[30]:

# to get all columns as a list
#list(data)


# In[31]:

# To get portion of the dataframe:  data.loc[startrow:endrow,startcolumn:endcolumn]
# To slice the rows, and include all columns:  data.loc[0:5, :]
# To select all rows from a single column:    data.loc[: , "my_column_name"]

# first 3 rows, all columns
#data.loc[0:2, :]


# ## Convert Y column (action_taken) to binary value
# 
# In order to simplify the problem to binary classification, conver the Y column (action_taken) to bunary value 0 and 1:
# 
#     action_taken new value	action_taken_name	count     
#     1	1   Loan originated	228054   
#     2	0   Application approved but not accepted	14180  
#     3	0   Application denied by financial institution	79697
#     4	0   Application withdrawn by applicant	39496  
#     5	0   File closed for incompleteness	16733  
#     6	1   Loan purchased by the institution	61490  
#     7	0   Preapproval request denied by financial instit...	4  

# In[32]:

df = df_orig.copy()

df['action_taken']= df['action_taken'].map({ 1:1, 2:1, 3:0, 4:0, 5:0, 6:1, 7:0})

df_action_count = get_count_of_unique_columns(df, ['action_taken','action_taken_name'])
print("df_action_count")
display(df_action_count)


# ## Drop Redundant, irrelevant, or None Columns
# 
# Columns that are redundant with other columns, or irrelevant with the outcome can distort the result, reduce the predition performance. Same goes with the columns that contain only empty values.   It is part of preprocessing to remove these columns,

# In[33]:

#df = df_orig.copy()

drop_column = [
    #'action_taken',
 'action_taken_name',
 #'agency_code', #Need one-hot-encoding
 'agency_abbr',
 'agency_name',
 #'applicant_ethnicity',  #Need one-hot-encoding
 'applicant_ethnicity_name',
 #'applicant_income_000s',
 #'applicant_race_1',     #Need one-hot-encoding
 'applicant_race_2',     
 'applicant_race_3',   
 'applicant_race_4',    
 'applicant_race_5',    
 'applicant_race_name_1',
 'applicant_race_name_2',
 'applicant_race_name_3',
 'applicant_race_name_4',
 'applicant_race_name_5',
 #'applicant_sex',          #Need one-hot-encoding
 'applicant_sex_name',
 'application_date_indicator',
 'as_of_year',
 'census_tract_number',
 'co_applicant_ethnicity', 
 'co_applicant_ethnicity_name',
 'co_applicant_race_1',    
 'co_applicant_race_2',    
 'co_applicant_race_3',    
 'co_applicant_race_4',     
 'co_applicant_race_5',    
 'co_applicant_race_name_1',
 'co_applicant_race_name_2',
 'co_applicant_race_name_3',
 'co_applicant_race_name_4',
 'co_applicant_race_name_5',
 'co_applicant_sex',    
 'co_applicant_sex_name',
 'county_code',        
 'county_name',
 'denial_reason_1',   
 'denial_reason_2',
 'denial_reason_3',
 'denial_reason_name_1',
 'denial_reason_name_2',
 'denial_reason_name_3',
 'edit_status',         #?
 'edit_status_name',
 #'hoepa_status',     #Need one-hot-encoding
 'hoepa_status_name',
 #'lien_status',     #Need one-hot-encoding
 'lien_status_name',
 #'loan_purpose',     #Need one-hot-encoding
 'loan_purpose_name',
 #'loan_type',     #Need one-hot-encoding
 'loan_type_name',
 'msamd',     
 'msamd_name',
 #'owner_occupancy',      #Need one-hot-encoding
 'owner_occupancy_name',
 #'preapproval',     #Need digitalize and one-hot-encoding
 'preapproval_name',
 #'property_type',       #Need one-hot-encoding
 'property_type_name',
 #'purchaser_type',       #Need one-hot-encoding
 'purchaser_type_name',
 'respondent_id',       
 'sequence_number',     
 'state_code',
 'state_abbr',
 'state_name']
 #'hud_median_family_income',
 #'loan_amount_000s',
 #'number_of_1_to_4_family_units',
 #'number_of_owner_occupied_units',
 #'minority_population',
 #'population',
 #'rate_spread',  
 #'tract_to_msamd_income']


# What we need is one-hot-encoding, and build a lookup dict for the columns that is been one-hot-encoded


# Drop the columns in the drop_column list
df.drop(drop_column, axis=1, inplace=True)

# df is altered after dropping the column. Inspect the data again.
# display(df.loc[3000:3002, 'action_taken':'lien_status'])
# display(df.loc[3000:3002, 'loan_purpose':'loan_amount_000s'])
# display(df.loc[3000:3002, 'number_of_1_to_4_family_units':'tract_to_msamd_income'])
    


# Other commands that may be helpful when pre-procses the dataframe
# 
# ```python
# 
# df.replace('n/a', np.nan,inplace=True)
# df.emp_length.fillna(value=0,inplace=True)
# 
# df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
# df['emp_length'] = df['emp_length'].astype(int)
# 
# df['term'] = df['term'].apply(lambda x: x.lstrip())
# '''
# 

# ## Feature Scaling (used in KNN, or any distance based model)
# 
# Feature scaling is the method to limit the range of variables so that they can be compared on common grounds. It is performed on continuous variables. Lets plot the distribution of all the continuous variables in  the data set.

# In[34]:

import matplotlib.pyplot as plt
#df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")]
#                        .index.values].hist(figsize=[11,11])

df[df.dtypes[(df.dtypes=="float64")].index.values].hist(figsize=[11,11])


# As we see different features have different range, if we try to apply distance based methods such as kNN on these features, feature with the largest range will dominate the outcome results and we’ll obtain less accurate predictions. We can overcome this trouble using feature scaling. 
# 
# We can solve  this problem by scaling down all the features to a same range. sklearn provides a tool MinMaxScaler that will scale down all the features between 0 and 1. Mathematical formula for MinMaxScaler is.
# 
# 
# $$X_{norm}=\frac{X-X_{min}}{X_{max}-X{min}}$$
# 

# In[35]:

#Decide which columns are to be scaled
scale_column = ['applicant_income_000s', 'hud_median_family_income', 'loan_amount_000s','number_of_1_to_4_family_units',
                'number_of_owner_occupied_units','minority_population','population','tract_to_msamd_income']
scale_column_indexer = ('applicant_income_000s', 'hud_median_family_income', 'loan_amount_000s','number_of_1_to_4_family_units',
                'number_of_owner_occupied_units','minority_population','population','tract_to_msamd_income')


#Drop rows whose scale_column contain missing data, NaN value, because the NaN will mess up the scaling
df_dropped = df.dropna(axis=0, how='any', subset=scale_column)

display(df_dropped.loc[0:2, :])

#df_dropped[scale_column] = df_dropped[scale_column].apply(lambda x: MinMaxScaler().fit_transform(x))

#df_dropped.loc[:, scale_column_indexer] = MinMaxScaler().fit_transform(df_dropped[scale_column].as_matrix())
#df_dropped[scale_column].apply(lambda x: MinMaxScaler().fit_transform(x))

from sklearn.preprocessing import minmax_scale
df_dropped.is_copy = False
df_dropped.loc[:, scale_column_indexer] = minmax_scale(df_dropped.loc[:, scale_column_indexer])


display(df_dropped.loc[0:2, :])


#np_minmax = min_max.fit_transform(dr_dropped[scale_column])

#Note np_minmax is the numpy np array after fit_transform
#print(np_minmax[3000:3002:1][0:8:1])


# ## Feature Standardization
# Standardization (or Z-score normalization) is the process where the features are rescaled so that they’ll have the properties of a standard normal distribution with μ=0 and σ=1, where μ is the mean (average) and σ is the standard deviation from the mean. Standard scores (also called z scores) of the samples are calculated as follows : 
# 
# $$Z=\frac{x-μ}{σ}$$
# 
# Elements such as l1 ,l2 regularizer in linear models (logistic comes under this category) and RBF kernel in SVM in objective function of learners assumes that all the features are centered around zero and have variance in the same order. Features having larger order of variance would dominate on the objective function as it happened in the previous section with the feature having large range.
# 
# Standardizing the data when using a estimator having l1 or l2 regularization helps us to increase the accuracy of the prediction model. Other learners like kNN with euclidean distance measure, k-means, SVM, perceptron, neural networks, linear discriminant analysis, principal component analysis may perform better with standardized data.

# In[36]:

# Standardizing the data
from sklearn.preprocessing import scale


np_scale=scale(df_dropped[scale_column])

#Note np_scale is the numpy np array after scale
print(np_scale[3000:3002:1][0:8:1])


#from sklearn.preprocessing import StandardScaler
#scale = StandardScaler()

#dfTest[['A','B','C']] = scale.fit_transform(dfTest[['A','B','C']].as_matrix())





# In[37]:

from sklearn.preprocessing import StandardScaler
import pandas
import numpy

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(df_dropped)
rescaledX = scaler.transform(df_dropped)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])



from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(df.values)

#Assign the scaled data to a DataFrame (Note: use the index and columns keyword arguments to keep your original indices and column names:
scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


# In[ ]:

#https://stackoverflow.com/questions/35723472/how-to-use-sklearn-fit-transform-with-pandas-and-return-dataframe-instead-of-num

from sklearn_pandas import DataFrameMapper

mapper = DataFrameMapper([(df.columns, StandardScaler())])
scaled_features = mapper.fit_transform(df.copy(), 4)
scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)


# ## Label Encoding
# 
# categorical features have string values, such as "Gender" has values of "male"m "female".  These features have to be converted into numeric values. Sklearn probides LabelEncoder to encode string labels with values between 0 and n_classes-1.  We do not need to encode because out dataset already contains categorical features in numeric columns.
# 
# The following is an example in case encoding is needed for other dataset:
# 
#     # Importing LabelEncoder and initializing it
#     >> from sklearn.preprocessing import LabelEncoder
#     >> le=LabelEncoder()
#     # Iterating over all the common columns in train and test
#     >> for col in X_test.columns.values:
#            # Encoding only categorical variables
#            if X_test[col].dtypes=='object':
#            # Using whole data to form an exhaustive list of levels
#            data=X_train[col].append(X_test[col])
#            le.fit(data.values)
#            X_train[col]=le.transform(X_train[col])
#            X_test[col]=le.transform(X_test[col])
#            

# ## One-Hot Encoding
# 
# One-Hot Encoding transforms each categorical feature with n possible values into n binary features, with only one active.
# 
#     Encode categorical integer features using a one-hot aka one-of-K scheme.
#     The input to this transformer should be a matrix of integers, denoting the values taken on by categorical (discrete) features.
#     The output will be a sparse matrix where each column corresponds to one possible value of one feature.
#     It is assumed that input features take on values in the range [0, n_values).
#     This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.
# 
# 

# May use sklearn for numpy nparray, or panda for panda dataframe
# 
#     # import preprocessing from sklearn
#     from sklearn import preprocessing
# 
#     # 1. INSTANTIATE
#     enc = preprocessing.OneHotEncoder()
# 
#     # 2. FIT
#     enc.fit(X_2)
# 
#     # 3. Transform
#     onehotlabels = enc.transform(X_2).toarray()
#     onehotlabels.shape
# 
#     # as you can see, you've the same number of rows 891
#     # but now you've so many more columns due to how we changed all the categorical data into numerical data
# 

# In[66]:

one_hot_column = ['agency_code','applicant_ethnicity','applicant_race_1', 'applicant_sex']

pd.get_dummies(data=df, columns=one_hot_column)


df_clean = df[~np.isnan(df).any(axis=1)]   ##only ~8k rows remain here
df_onlyIncome_clean = df[~np.isnan(df['applicant_income_000s'])]  ## ~380k rows here
pd.get_dummies(data=df_income_clean, columns=one_hot_column)

# ## Write Data To cPickle(Optional)

# In[2]:

cPickle.dump(df, open('data/cleanData.p', 'wb'))


# In[ ]:



