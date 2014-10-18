# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:50:42 2014

@author: SpectralMD2
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:48:02 2014

@author: Valued Customer
"""

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation 
import pylab as plt

plt.close('all')
#%% Loading the data
def OpenCVS(path):
    f = open(path,'rb')
    reader = csv.reader(f)
    labels = []
    for row in reader:
        labels.append(row)
    f.close()
    data = np.array(labels)
    return data
train_path = 'C:/Users/SpectralMD2/Desktop/Kaggle/training.csv'
test_path = 'C:/Users/SpectralMD2/Desktop/Kaggle/sorted_test.csv'
train = OpenCVS(train_path)
test = OpenCVS(test_path)

targetName = train[0,-5:]

def findfeature(data):
    Index = []
    for i in range(data.shape[1]):
        if train[0,i][0] == 'm':
            Index.append(i)
    return Index
Index = findfeature(train)
Data = train[1:,Index]

indexTest = findfeature(test)
TestData = test[1:,Index]

Target= train[1:,train.shape[1]-5:]

trainSpace= np.array([float(i) for i in Data.ravel() ]).reshape(Data.shape)
target = np.array([float(i) for i in Target.ravel() ]).reshape(Target.shape)

testData = np.array([float(i) for i in TestData.ravel() ]).reshape(TestData.shape)

train = None
#%% do the feature space
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

clf_SVR = SVR(kernel='rbf', C=1e3, degree=6)

estimator = SVR(kernel="linear")



clf = svm.SVC(kernel='linear')
from sklearn import svm, datasets, feature_selection, cross_validation
transform = feature_selection.SelectPercentile(feature_selection.f_classif)

clf = Pipeline([('anova', transform), ('svr', clf_SVR)])
score_means = list()
score_stds = list()
percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)


for percentile in percentiles:
    clf.set_params(anova__percentile=percentile)
    # Compute cross-validation score using all CPUs
    
    
    this_scores = cross_validation.cross_val_score(clf,trainSpace,target[:,1],cv = 10,\
                                                    scoring='r2')
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())

plt.figure()
plt.errorbar(percentiles, score_means, np.array(score_stds))

plt.title(
    'Performance of the SVM-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')

plt.axis('tight')
plt.show()








#%% data pre-processing, data corss validtion
from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(target[:,1],20)
trainIndex = []
testIndex = []
count = 0
for train, test in skf:
    trainIndex.append(train)
    testIndex.append(test)


#  
Index =4


trainData_1 = feature[trainIndex[Index],:]
testData_1 = feature[testIndex[Index],:]

output = []
error = []



    
class pattern:
    def __init__(self,clf,trainData,testData,targetTrain,targetTest,output,error):
        self.trainData = trainData
        self.testData= testData
        self.targetTrain = targetTrain
        self.targetTest = targetTest
        self.clf = clf
        self.output = []
        self.error = []
    
    def Train(self):
        """training and test the data"""
        self.clf.fit(self.trainData,self.targetTrain)
        
        for i in range(len(self.testData)):
            self.output.append(self.clf.predict(self.testData[i,:]))
            self.error.append(self.targetTest[i]-self.clf.predict(self.testData[i,:]))
        return self.output,self.error
    
    def validation(self):
        """Plot the error bar"""
        plt.figure()
        plt.plot(np.arange(len(self.testData)),self.output,color = 'r',label='Output')
        plt.plot(np.arange(len(self.testData)),self.targetTest,color = 'b',label = 'True')
        plt.legend()
        plt.title('Model validation %s' % self.clf)


#%% Do the wavelength selection

trainData_1 = feature[trainIndex[Index],:]
testData_1 = feature[testIndex[Index],:]
#%% data 


from sklearn import linear_model
from sklearn.feature_selection import RFE
clf = linear_model.LinearRegression()
clf_Ridge= linear_model.Ridge(alpha = 0.5)
clf_bayesian = linear_model.BayesianRidge()

selector = RFE(clf_bayesian,step =0.2)


def IndexFind(selector):
    FeatureIndex = []
    for i in range(len(selector.support_)):
        curr = selector.support_[i]
        if curr == True:
            FeatureIndex.append(i)
    print len(FeatureIndex)
    return FeatureIndex






testOutput = []
# Run the code
for i in range(target.shape[1]):
    print 'new one begin %d' % i
    Target = target[:,i]  
    targetTrain = Target[trainIndex[Index]]
    targetTest = Target[testIndex[Index]]
    
    #%% Feature selection
    selector = selector.fit(trainData_1, targetTrain)
    selectIndex = IndexFind(selector)
    
    trainData = trainData_1[:,selectIndex]
    testData = testData_1[:,selectIndex]
    TestData_2 = TestData_1[:,selectIndex]
    
   
    a = pattern(clf,trainData,testData,targetTrain,targetTest,output,error)
    a.Train()
    a.validation()
    
    b = pattern(clf_Ridge,trainData,testData,targetTrain,targetTest,output,error)
    b.Train()
    b.validation()
    
    c = pattern(clf_bayesian,trainData,testData,targetTrain,targetTest,output,error)
    c.Train()
    c.validation()
    
    
    plt.figure()
    plt.plot(np.arange(len(a.testData)),a.targetTest,color = 'r',label = 'True')
    plt.plot(np.arange(len(a.testData)),a.output,color = 'b',label='lineae')
    
    plt.plot(np.arange(len(b.testData)),b.output,color = 'k',label = 'Ridege',marker='*')
    plt.plot(np.arange(len(c.testData)),c.output,color = 'g',label = 'Byesian',marker='o')
    plt.legend()
    testOutput.append(c.clf.predict(TestData_2))

#%% get the output from the test data




import pandas as pd
data = pd.read_csv('C:/Users/SpectralMD2/Desktop/Kaggle/sample_submission.csv')
name= ['Ca', 'P', 'pH', 'SOC', 'Sand']
for i in range(5):
    data[name[i]] = testOutput[i]

data.to_csv("sample_submission1.csv",index=False)

data.to_excel('sample_submission1.xls')
