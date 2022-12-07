#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Lab 1

# ## Assignment 2 (Deadline : 4/12/2022 11:59PM)
# 
# Total Points : 100
# 
# Your answers must be entered in LMS by midnight of the day it is due. 
# 
# If the question requires a textual response, you can create a PDF and upload that. 
# 
# The PDF might be generated from MS-WORD, LATEX, the image of a hand- written response, or using any other mechanism. 
# 
# Code must be uploaded and may require demonstration to the TA. 
# 
# Numbers in the parentheses indicate points allocated to the question. 
# 
# **Naming Convention**: FirstName_LastName_Lab2_TLP23.ipynb

# # Question 1 (50 points)

# ## 1. Read the data into a numpy array

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# Read data from file: 'data.csv'. 
# Note that the first row has the number of inputs and number of outputs specified.
# For your reference: you can use "np.loadtxt".
data_np = np.loadtxt('data.csv', dtype=str)
df = pd.read_csv('data1.csv', skiprows=0)
df


# ## 2. Plot and explore the data to get a better understanding

# In[2]:


# EDA
print(df.shape)
print(df.index)
plt.scatter(df['A'], df['B'], cmap="rgb")

plt.show()


# In[3]:


df.columns


# In[4]:


plt.figure(figsize=(15,15))
df.hist(bins=50)
plt.show()


# In[5]:


plt.figure(figsize=(10,8))
plt.scatter(df['A'], df['Target'], color='orange')
plt.scatter(df['B'], df['Target'], color='red')
plt.xlabel('Values of column A and B (inputs)')
plt.ylabel('Target values')
plt.show()


# In[6]:


plt.figure(figsize=(15,15))
sns.pairplot(df, palette='coolwarm', hue='Target')
plt.show()


# In[7]:


df.describe()


# ## 3. Prepare the data for modelling

# In[8]:


# Separate the data: input and output.
input_df = df[['A', 'B']]
output_df = df[['Target']]
output_df


# In[9]:


input_df


# In[10]:


# Perform a train-val-test split of 60-20-20.
# Use "random_state=42".
# For your reference: you can use "train_test_split" from sklearn twice to perform this. 
X_1, X_test, y_1, y_test = train_test_split(input_df, output_df, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_1, y_1, test_size = 0.25, random_state=42)
    


#  * From your understanding of the dataset, is stratified sampling necessary in this case? Why / Why not?
# 

# ## 4. Modelling

# ### 4.1 Linear Regression

# In[11]:


# Implement Linear Regression.
# Use both train and val data for training purpose.
# Make predictions on both training(train+val) and test data.
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(X_1, y_1)
train_predictLR = modelLR.predict(X_1)
test_predictLR = modelLR.predict(X_test)


# ### 4.2 Ridge Regression

# In[12]:


# Implement Ridge Regression.
# Perform cross validation to find a good value for your hyper-parameter.
# After choosing a good value for your hyper-parameter, use both train and val data for final training purpose of your model.
# Make predictions on both training(train+val) and test data.
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
modelRdg = Ridge()
alphaVals = {'alpha':np.linspace(0,1,200)}
gridSearch = GridSearchCV(modelRdg,param_grid=alphaVals, cv=4, scoring='neg_mean_squared_error')
gridSearch.fit(X_1, y_1)
train_predictRdg = gridSearch.predict(X_1)
test_predictRdg = gridSearch.predict(X_test)

'''
mseValRdg = r2_score(y_val, val_predictRdg)
mseTestRdg = r2_score(y_test, test_predictRdg)
print(mseValRdg*100, " ", mseTestRdg*100)
'''


# In[13]:


gridSearch.best_params_


# ### 4.3 Lasso Regression

# In[14]:


# Implement Lasso Regression.
# Perform cross validation to find a good value for your hyper-parameter.
# After choosing a good value for your hyper-parameter, use both train and val data for final training purpose of your model.
# Make predictions on both training(train+val) and test data.
from sklearn.linear_model import Lasso
modelLasso = Lasso()
alphaVals = {'alpha':np.linspace(1,7,200)}
gridSearchLasso = GridSearchCV(modelLasso,param_grid=alphaVals, cv=4, scoring='neg_mean_squared_error')
gridSearchLasso.fit(X_1, y_1)
train_predictLasso = gridSearchLasso.predict(X_1)
test_predictLasso = gridSearchLasso.predict(X_test)


# In[15]:


gridSearchLasso.best_params_


# ### 4.4 ElasticNet Regression

# In[16]:


# Implement ElasticNet Regression.
# Perform cross validation to find a good value for your hyper-parameters.
# After choosing a good value for your hyper-parameter, use both train and val data for final training purpose of your model.
# Make predictions on both training(train+val) and test data.
from sklearn.linear_model import ElasticNet
modelElasticNet = ElasticNet()
alphaVals = {'alpha':np.linspace(1,7,200)}
gridSearchEN = GridSearchCV(modelElasticNet,param_grid=alphaVals, cv=4, scoring='neg_mean_squared_error')
gridSearchEN.fit(X_1, y_1)
train_predictElasticNet = gridSearchEN.predict(X_1)
test_predictElasticNet = gridSearchEN.predict(X_test)


# ## 5. Analysis

# In[17]:


# Write down the actual value along with the predictions from all the regression models, for the first 10 points in test data.
c1 = pd.DataFrame(test_predictLR[:10])
c2 = pd.DataFrame(test_predictRdg[:10])
c3 = pd.DataFrame(test_predictLasso[:10])
c4 = pd.DataFrame(test_predictElasticNet[:10])

scores_df = pd.DataFrame(y_test[:10])
scores_df['LR'] = test_predictLR[:10]
scores_df['Ridge'] = test_predictRdg[:10]
scores_df['Lasso'] = test_predictLasso[:10]
scores_df['ElasticNet'] = test_predictElasticNet[:10]
scores_df.rename(columns={'index':'True'}, inplace=True)
scores_df


# In[18]:


# Find both RMSE and MAE for all the regression models on both training(train+val) and testing data.
from sklearn.metrics import mean_squared_error, mean_absolute_error    
rmseLR = mean_squared_error(y_1, train_predictLR) ** 0.5
maeLR = mean_absolute_error(y_1, train_predictLR)
print("RMSE and MAE for training via Linear Regression:", rmseLR, " ", maeLR)
rmseLRtest = mean_squared_error(y_test, test_predictLR) ** 0.5
maeLRtest = mean_absolute_error(y_test, test_predictLR)
print("RMSE and MAE for testing via Linear Regression:", rmseLRtest, " ", maeLRtest)
print('\t')
rmseLRRidge = mean_squared_error(y_1, train_predictRdg) ** 0.5
maeLRRidge = mean_absolute_error(y_1, train_predictRdg)
print("RMSE and MAE for training via Ridge Regression:", rmseLRRidge, " ", maeLRRidge)
rmseRdgtest = mean_squared_error(y_test, test_predictRdg) ** 0.5
maeRdgtest = mean_absolute_error(y_test, test_predictRdg)
print("RMSE and MAE for testing via Ridge Regression:", rmseRdgtest, " ", maeRdgtest)
print('\t')
rmseLasso = mean_squared_error(y_1, train_predictLasso) ** 0.5
maeLasso = mean_absolute_error(y_1, train_predictLasso)
print("RMSE and MAE for training via Lasso Regression:", rmseLasso, " ", maeLasso)
rmseLassotest = mean_squared_error(y_test, test_predictLasso) ** 0.5
maeLassotest = mean_absolute_error(y_test, test_predictLasso)
print("RMSE and MAE for testing via Lasso Regression:", rmseRdgtest, " ", maeRdgtest)
print('\t')
rmseEN = mean_squared_error(y_1, train_predictElasticNet) ** 0.5
maeEN = mean_absolute_error(y_1, train_predictElasticNet)
print("RMSE and MAE for training via Lasso Regression:", rmseEN, " ", maeEN)
rmseENtest = mean_squared_error(y_test, test_predictElasticNet) ** 0.5
maeENtest = mean_absolute_error(y_test, test_predictElasticNet)
print("RMSE and MAE for testing via Lasso Regression:", rmseENtest, " ", maeENtest)


# ## 6. Final Observations
# Write down your final conclusions and observations

# My observations are that the two input columns are easily able to predict the final feature value. This is because the two inputs, their linear combination is an output which is provided which is why it is easy for the Linear Regression model to capture this trend quite easily. Since the output is quite easily predictable, Lasso, Ridge and Elastic Net also result in the same outputs, i.e, they perform same as Linear Regression since there is no room to capture any misprediction and penalize it(which is what Ridge and Lasso regressions do).

# # Question 2 (30 points)
# 
# Given the automobile dataset at https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv , 

# ## 1. Load and examine the dataset

# In[19]:


df = pd.read_csv('automobileEDA.csv')
df.head()


# In[20]:


df.info()


# In[21]:


df.shape


# In[22]:


df.isna().sum()


# In[23]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# In[24]:


plt.figure(figsize=(20,20))
sns.pairplot(df)


# In[25]:


plt.figure(figsize=(15,12))
sns.heatmap(df.corr(), annot=True, 
            cmap=sns.diverging_palette(220, 10, as_cmap=True))
plt.show()


# In[26]:


df_numerical = df.select_dtypes(include=np.number)
corr_matrix = df_numerical.corr()

target_corr = abs(corr_matrix["city-L/100km"])
features = target_corr[target_corr>0.5]
features.index
df_cols = ['length', 'width', 'curb-weight', 'engine-size', 'bore', 'horsepower',
       'city-mpg', 'highway-mpg', 'price', 'city-L/100km']


# In[27]:


plt.figure(figsize=(16,16))
sns.heatmap(df[df_cols].corr().abs(), annot=True)


# In[28]:


cat_cols = df.select_dtypes(include='object')
cat_cols.columns


# In[29]:


def isunique(col_name):
    return df[col_name].nunique()
for cols in cat_cols.columns:
    if isunique(cols) <= 7:
        print(cols)
cols = np.array(cols).T
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
encoder_df = pd.DataFrame(ohe.fit_transform(df[["aspiration",
"num-of-doors",
"body-style",
"drive-wheels",
"engine-location",
"engine-type",
"num-of-cylinders"]]).toarray())
encoder_df


# In[30]:


final_df = df[df_cols]
final_df = final_df.join(encoder_df)


# In[31]:


final_df


# ## 2. Visualise/Plot the regression model

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = df["city-L/100km"]
final_df.drop("city-L/100km", inplace=True, axis=1)


# In[33]:


X = final_df.loc[:, final_df.columns]
X


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
preds = linreg.predict(X_test)


# In[36]:


preds


# In[37]:


plt.plot(preds, y_test, 'o', color='purple')
plt.xlabel('Mileage Predictions')
plt.ylabel('True mileage values')
plt.title('Predictions vs actual mileage')
plt.show()


# ## 3. Generate a Linear Regression equation

# In[38]:


ypreds = linreg.coef_ @ X_test.to_numpy().T + linreg.intercept_


# The linear regression equation given above is of form 
#   
#   ### prediction = wT * X+ b
#  
# Our wT(the weights assigned) is the linear regression coefficients generated. X is the X_test on which we calculate our prediction and finally intercept(b) is the linear regression intercept generated by the model.

# ## 4. Use a residual plot to inspect if LR fits the model

# In[39]:


res = y_test - preds
fig, ax = plt.subplots(1,2, figsize = (20,10))
bins = np.linspace(0,50,10)
ax[0].set_xlabel('Residuals',size=15,color='red')
ax[0].set_ylabel('Frequency',size=15,color='purple')
ax[0].title.set_text('Histogram for residues')
ax[0].hist(res,bins, color='red', alpha=1)

ax[1].scatter(y_test, res, s = 100, color='green')
ax[1].set_xlabel('Predicted values',size=15,color='purple')
ax[1].set_ylabel('Residuals',size=15,color='purple')
ax[1].title.set_text('Scatter plot Residues vs Actual Values(Test Data)')
fig.suptitle('Residual Analysis',size=30,color='darkblue')
plt.show();


# ## 5. Use R2 and MSE to determine the accuracy of the LR fit 

# In[40]:


r2_score = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)


# In[41]:


print(mse, " ", r2_score)


# In[42]:


rmse = mse ** 0.5
rmse


# In[43]:


''' 
from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42,
     loss='squared_error').fit(X_train, y_train)
'''


# In[44]:


#mean_squared_error(y_test, est.predict(X_test))


# In[45]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
modelRdg = Ridge()
alphaVals = {'alpha':np.linspace(0,1,200)}
gridSearch = GridSearchCV(modelRdg,param_grid=alphaVals, cv=4, scoring='neg_mean_squared_error')
gridSearch.fit(X_train, y_train)
predsRidge = gridSearch.predict(X_test)


# In[46]:


plt.plot(predsRidge, y_test, 'o', color='orange')
plt.xlabel('Mileage Predictions')
plt.ylabel('True mileage values')
plt.title('Predictions vs actual mileage')
plt.show()


# In[47]:


#r2_scoreRdg = r2_score(y_test, predsRidge)
mseRdg = mean_squared_error(y_test, predsRidge)
print(mseRdg)


# In[48]:


from sklearn.linear_model import Lasso
modelLasso = Lasso()
alphaVals = {'alpha':np.linspace(1,7,200)}
gridSearchLasso = GridSearchCV(modelLasso,param_grid=alphaVals, cv=4, scoring='neg_mean_squared_error')
gridSearchLasso.fit(X_train, y_train)
predsLasso = gridSearchLasso.predict(X_test)


# In[49]:


plt.plot(predsLasso, y_test, 'o', color='cyan')
plt.xlabel('Mileage Predictions')
plt.ylabel('True mileage values')
plt.title('Predictions vs actual mileage')
plt.show()


# In[50]:


trend = np.polyfit(predsLasso, y_test, deg=1)
plt.plot(predsLasso, y_test, 'o')
trendpoly = np.poly1d(trend) 
plt.plot(y_test, trendpoly(y_test))
plt.show()


# In[51]:


mseRdg = mean_squared_error(y_test, predsRidge)
print(mseRdg)


# # Question 3 (20 points)

# ### In this assignment, you will experiment with a toy dataset – the Iris dataset.

# ## 1. Load the Iris dataset.

# In[63]:


# use -  sklearn.datasets.load_iris
from sklearn.datasets import load_iris


iris = load_iris()

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


# In[64]:


df


# ## 2. The Iris data has 3 classes. For the purpose of this assignment, you will modify it such that it has two classes – specifically, you will merge the “setosa” and “versicolor” classes.

# In[65]:


df['new_target'] = df.apply(lambda x: 0 if x['target'] == 0 or x['target'] == 1 else 1, axis=1)
df


# In[66]:


df = df.drop(['target', 'species'], axis=1)


# In[67]:


df


# ## 3. Construct a training set and a testing set using 80-20 split using random sampling.

# In[69]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = train.iloc[:,:3], train.iloc[:,4]
X_test, y_test = test.iloc[:, :3], test.iloc[:, 4]


# ## 4. Use Logistic Regression as a black box classifier and assess the performance.

# In[70]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)


# In[71]:


from sklearn.metrics import accuracy_score, classification_report
acc = accuracy_score(pred, y_test)
print(acc, " \n", classification_report(pred, y_test))


# ## 5. Implement stratified sampling, again using the 80-20 split. Decide the stratification variable and explain your choice.

# In[73]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["new_target"]):
    train = df.loc[train_index]
    test = df.loc[test_index]

X_train, y_train = train.iloc[:,:3], train.iloc[:,4]
X_test, y_test = test.iloc[:, :3], test.iloc[:, 4]


# I have performed the split based on the new target value created. The data is easily seperable based on the new classes we have generated. But in order to assure that there is an equal split of these classes in our test data, we perform startified split on the target class.

# ## 6. Re-assess the performance. Comment on any changes from the random sampling case. If there is no significant change, comment on why or why not?

# In[74]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
acc = accuracy_score(pred, y_test)
print(acc, " \n", classification_report(pred, y_test))


# The given iris dataset in not a very large dataset, It had three categories of which we clubbed and made two categories. Stratified sample split would work better than a random split if there was a misrepresentation of classes in the splits. But with two classes, the random split is already working well enough to capture the model parameters required for classification. This is possibly the reason why startified split and random split work equally well. Had we had a detailed big dataset, random split would perform poorly as compared to the startified split becuase a startified split allows the test data to maintain the same ration between target classes as is maintained in the training dataset which helps us to better understand if the model we have made is generalizing better to the given data
