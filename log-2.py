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
df = pd.read_csv("C:/Users/user/Desktop/datasets/Advertising.csv",encoding= 'unicode_escape')
df.columns

df.head(8) #shows first 8 observations of the data
df.describe() #similar to summary in R
df.isna().sum() #checking na values
#no na values

#considering only few required variables for building a model
df= df[['Daily_Time_ Spent _on_Site', 'Age', 'Area_Income','Daily Internet Usage',  'Male',  'Clicked_on_Ad']]
#changing colnames for better typing
df.columns = 'dailytimespent','age','areaincome','internetusage','male','clickedonad'

#univariate analysis
df.dailytimespent.hist()
df.age.hist()
df.areaincome.hist()
df.male.hist()
df.internetusage.hist()
#bivariate analysis
a = df[['dailytimespent','age','areaincome','internetusage']]
for i in a:
    sns.barplot(x=i,y='clickedonad',data=df)
    plt.show()

#majority of the age are from 28 to 42
#People whose age is more than 53 did not click the ad
#majority of users who spend more than 80 are clicking  the ad

# Model building 
# import statsmodels.formula.api as sm
logit_model = sm.logit('clickedonad ~ dailytimespent+age+areaincome+internetusage+male', data = df).fit()

#summary
logit_model.summary()
logit_model.summary2() #for AIC value
pred = logit_model.predict(df.iloc[ :, 0:5 ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.clickedonad, pred)
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

# filling all the cells with zeroes
df["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(df["pred"], df["clickedonad"])
classification ##precision,recall,f1 score values


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.2) # 20% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('clickedonad ~ dailytimespent+age+areaincome+internetusage+male', data = train_data).fit()

#summary
model.summary2() # for AIC value
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(200)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['clickedonad'])
confusion_matrix

accuracy_test = (98+96)/(200) 
accuracy_test #97%

# classification report
classification_test = classification_report(test_data["test_pred"], test_data['clickedonad'])
classification_test #precision,recall,f1 score values

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data['clickedonad'], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test #are under the curve is 79%


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 0:5 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(800)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['clickedonad'])
confusion_matrx

accuracy_train = (396+382)/(800)
print(accuracy_train) #97%
#training accuracy and testing accuracy are 97% and equal => good fit model