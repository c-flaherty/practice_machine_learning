#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
LOADING MODULES AND DATA
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)


training_data = pd.read_csv("titanic/train.csv")
eval_data = pd.read_csv("titanic/test.csv")


# In[2]:


'''
DATA EXPLORATION
'''
print(training_data.info())
print(training_data.sample(5))
for col in [col for col in training_data.columns if col not in ['PassengerId','Age','Name','Fare','Ticket']]:
    print(col+": ",training_data[col].unique())
    
print("How many passengers have their cabin listed as 'nan'?")
print(training_data["Cabin"].isnull().sum(),"out of", training_data["Cabin"].size)


# In[3]:


'''
FILLING IN NULL DATA
'''

print("Counts of NaN entries by column: \n",training_data.isnull().sum())

# Most passengers were not in a cabin. 
# Reduce Cabin to either Cabin or Not Cabin

# Fill Nans for Cabin
training_data.loc[training_data["Cabin"].notnull(),"Cabin"] = "Cabin"
training_data["Cabin"].fillna("Not Cabin", inplace = True)

fig1, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
ax1.hist(training_data["Age"])
ax2.hist(training_data.loc[training_data["Embarked"].notnull(),"Embarked"])
fig1.show()

# Since the ages of passengers approximately exhibit a skewed Gaussian,
#   a good estimator for the age would be the median
median_age = training_data["Age"].median()
training_data["Age"].fillna(median_age, inplace = True)

# Since the origin location is overwhelmingly "S", 
#   S would make for a good estimator for the Embarked variable
training_data["Embarked"].fillna("S", inplace = True)


# In[4]:


print("Counts of NaN entries by column: \n",training_data.isnull().sum())

fig2, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
ax1.hist(training_data["Age"])
ax2.hist(training_data["Embarked"])
fig2.show()


# In[5]:


'''
ONE HOT ENCODING OF CATEGORICAL VARIABLES
'''
 
print(training_data.columns)
print(training_data.sample(1))
training_data = pd.get_dummies(training_data,columns=["Sex", "Pclass","Embarked","Cabin"])
print(training_data.columns)
print(training_data.sample(1))


# In[6]:


'''
TRYING DIFFERENT MODELS
'''

# Training Data
Features = training_data.drop(["Survived","Name","Ticket", "PassengerId"],axis=1)
Outcome = training_data["Survived"]

# 5-fold Cross Validation on a model
def test_model(model):
    scores = cross_val_score(model,Features,Outcome,cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Models to Try
MLA = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),
    
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    neighbors.KNeighborsClassifier(),
    
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    ]

for model in MLA:
    test_model(model)
    


# In[13]:


'''
OPTIMIZING HYPERPARAMETERS ON OPTIMAL MODEL
'''
# It looks like ensemble.GradientBoostingClassifier() is the best classifier
#   with an accuracy of 0.83 +/- 0.04

# Optimizing Hyperparameters of the GradientBoosting Classifier
from sklearn import model_selection

model = ensemble.GradientBoostingClassifier()
param_grid = {'n_estimators':[50,100,150,200]}

tuned_model = model_selection.GridSearchCV(model,
                                           param_grid=param_grid,
                                           cv=5,
                                           n_jobs=-1)
tuned_model.fit(Features, Outcome)


# In[8]:


'''
FILLING IN NULL DATA OF EVALUATION DATA
'''

print("Counts of NaN entries by column: \n",eval_data.isnull().sum())

# Fill Nans for Cabin
eval_data.loc[eval_data["Cabin"].notnull(),"Cabin"] = "Cabin"
eval_data["Cabin"].fillna("Not Cabin", inplace = True)

fig1, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
ax1.hist(eval_data["Age"])
ax2.hist(eval_data.loc[eval_data["Fare"].notnull(),"Fare"])
fig1.show()

# Since the ages of passengers approximately exhibit a skewed Gaussian,
#   a good estimator for the age would be the median
median_age = eval_data["Age"].median()
eval_data["Age"].fillna(median_age, inplace = True)

# Since the fares of passengers are heavily skewed,
#   a good estimator for the fares would be the median
median_fare = eval_data["Fare"].median()
eval_data["Fare"].fillna(median_fare, inplace = True)


# In[9]:


print("Counts of NaN entries by column: \n",eval_data.isnull().sum())

fig2, (ax1,ax2) = plt.subplots(ncols=1,nrows=2,figsize=(10,10))
ax1.hist(eval_data["Age"])
ax2.hist(eval_data["Fare"])
fig2.show()


# In[10]:


'''
ONE HOT ENCODING OF CATEGORICAL VARIABLES OF EVALUATION DATA
'''
 
print(training_data.columns)
print(training_data.sample(1))
eval_data = pd.get_dummies(eval_data,columns=["Sex", "Pclass","Embarked","Cabin"])
print(training_data.columns)
print(training_data.sample(1))


# In[14]:


'''
PREDICTING SURVIVAL FROM EVALUATION DATA
'''
Eval_Features = eval_data.drop(["Name","Ticket", "PassengerId"],axis=1)
Outcomes = tuned_model.predict(Eval_Features)


# In[15]:


'''
SAVING PREDICTIONS TO CSV FOR SUBMISSION
'''
output = pd.DataFrame({"PassengerId": eval_data["PassengerId"], "Survived": Outcomes})
output.to_csv("evaluation_results.csv",index=False)

