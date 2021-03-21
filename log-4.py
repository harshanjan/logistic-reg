#importing all required modules at the beginining itself
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
df = pd.read_csv("C:/Users/user/Desktop/datasets/bank_data.csv",encoding= 'unicode_escape')
df.columns

df.head(8) #shows first 8 observations of the data
df.describe() #similar to summary in R
df.isna().sum() #checking na values
#no na values


#when i run VIF it thrws an error saying aliased coeff in model which is leading to perfect multicollinearity
#therefore, im considering only few variables according to pairs and correlation.



#considering only few required variables for building a model
df= df[['age','balance','default','y']]


#univariate analysis
df.balance.hist()
df.age.hist()
df.default.hist()

#bivariate analysis


plt.scatter('age','y')
plt.scatter('balance','y')
plt.scatter('default','y')

df.corr()
#when i run VIF it thrws an error saying aliased coeff in model which is leading to perfect multicollinearity



# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('y ~ age + balance + default', data = df).fit()

#summary
logit_model.summary()
logit_model.summary2() #for AIC value
pred = logit_model.predict(df.iloc[ :,0:3])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.y, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr) 
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
df["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(df["pred"], df["y"])
classification ##precision,recall,f1 score values


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.2) # 20% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ age + balance + default', data = train_data).fit()

#summary
model.summary2() # for AIC value
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(9043)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (6592+293)/(9043) 
accuracy_test #76%

# classification report
classification_test = classification_report(test_data["test_pred"], test_data['y'])
classification_test #precision,recall,f1 score values

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data['y'], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test 


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:3 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(36168)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (26942+1101)/(36168)
print(accuracy_train) #77%
#training accuracy is 77 and testing accuracy is 76% and equal.