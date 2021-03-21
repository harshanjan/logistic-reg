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
df = pd.read_csv("C:/Users/user/Desktop/datasets/election_data.csv",encoding= 'unicode_escape')
df.columns

df.head(7) #shows first 7 observations of the data
df.describe() #similar to summary in R
df.isna().sum() #checking na values
#excluding first observation due to missingness of the data
df = df.iloc[1:,:]

#changing colnames for better typing
df.columns = 'electionid','result','year','amount','popularityrank'

#univariate analysis
df.electionid.hist()
df.result.hist()
df.year.hist()
df.amount.hist()
df.popularityrank.hist()
#bivariate analysis
a = df[['electionid','year','amount','popularityrank']]
for i in a:
    sns.countplot(x = 'result',data= df)
    plt.show()
plt.scatter(df.amount,df.popularityrank)
plt.scatter(df.result,df.popularityrank)
plt.scatter(df.result,df.amount)
#person who has popularity and money are more likely to win

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('result~ electionid + year + amount + popularityrank', data = df).fit()
#PerfectSeparationError: Perfect separation detected, results not available
#i have tried removing popularityrank variable to solve the issue
logit_model = sm.logit('result~ electionid + year + amount ', data = df).fit()
#summary
logit_model.summary()
logit_model.summary2() #for AIC value
pred = logit_model.predict(df.iloc[ :, [0,2,3,4 ]])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.result, pred)
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

roc_auc = auc(fpr, tpr) #99%
print("Area under the ROC curve : %f" % roc_auc)
#87%
# filling all the cells with zeroes
df["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(df["pred"], df["result"])
classification ##precision,recall,f1 score values


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.2) # 20% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('result~ electionid + year + amount ', data = train_data).fit()

#summary
model.summary2() # for AIC value
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(2)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['result'])
confusion_matrix

accuracy_test = (1+1)/(2) 
accuracy_test #1

# classification report
classification_test = classification_report(test_data["test_pred"], test_data['result'])
classification_test #precision,recall,f1 score values

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data['result'], test_pred)



# prediction on train data
train_pred = model.predict(train_data.iloc[ :, [0,2,3,4] ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(8)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['result'])
confusion_matrx

accuracy_train = (2+3)/(8)
print(accuracy_train) #62%
#for 10 observations, we wont do training and testing
#When the accuracy is 100% then there is some problem with the data and the model doesn’t work in reality.
 