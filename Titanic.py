import enum
from pyexpat.errors import XML_ERROR_MISPLACED_XML_PI
from random import sample
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
import csv
from utilities import visualize_classifier
from sklearn.metrics import classification_report
from fancyimpute import KNN

traincsv = pd.read_csv('train.csv')
testcsv = pd.read_csv('test.csv')
# submission = pd.read_csv('sample_submission.csv')

traincsv.pop('Cabin')
traincsv.pop('Name')
traincsv.pop('Embarked')
traincsv.pop('Ticket')
testcsv.pop('Embarked')
testcsv.pop('Name')
testcsv.pop('Cabin')
testcsv.pop('Ticket')
testcsv.pop('PassengerId')
print(traincsv.head())

def isDigit(a):
    try:
        float(a)
        return True
    except ValueError:
        return False

def isNaN(string):
    return string != string


train = traincsv.iloc[:,1:].values
test = testcsv.iloc[:,:].values
surv = 0
died = 0
x = []
z = []

for i, item in enumerate(test):
    z.append(item)
    

for i, item in enumerate(train):
    if item[0] == 1:
        surv += 1
        x.append(item)
    if item[0] == 0:
        died += 1
        x.append(item)

x = np.array(x)
z = np.array(z)

for i, item in enumerate(z[:,2]):
    if isNaN(item):
        z[i,2] = random.randrange(0,69)
for i, item in enumerate(z[:,5]):
    if isNaN(item):
        z[i,5] = random.randrange(5,45)

x_encoded = np.empty(x.shape)
encoder = []   

for i,item in enumerate(x[0]):
    if isNaN(item):
        continue
    if isDigit(item):
        x_encoded[:,i] = x[:,i]
    else:
        encoder.append(preprocessing.LabelEncoder())
        x_encoded[:,i] = encoder[-1].fit_transform(x[:,i])
print(x_encoded.shape)

# for i, item in enumerate(x_encoded[:,8]):
#     if isNaN(item):
#         x_encoded[i,9] = 2.0

for i, item in enumerate(x_encoded[:,3]):
    if isNaN(item):
        x_encoded[i,3] = random.randrange(0,69)

x = x_encoded[:,1:].astype(int)
y = x_encoded[:,0].astype(int)

classifier = OneVsOneClassifier(LinearSVC(random_state=14))
classifier.fit(x,y)

class_names = ['Survived', 'Died']
print(classification_report(y, classifier.predict(x), target_names=class_names))

# print(int(float(31.275)))

# input_data = ['1','female','35','1','0','113803','53','S']
count = 0
z_encoded = np.empty(z.shape)
for i, item in enumerate(z[0]):
    if isDigit(item):
        z_encoded[:,i] = z[:,i]
    else:
        z_encoded[:,i] = encoder[count].transform(z[:,i])
        count += 1
input_data_encoded = np.array([z_encoded])

predicted_class = classifier.predict(z_encoded)
print(predicted_class)
# print(classifier.predict(sample_data_encoded))


submission = pd.read_csv('sample_submission.csv')
submission['Survived'] = classifier.predict(z_encoded)
submission.to_csv('submission.csv', index=False)

{#Попытка 3 и 2 
# input_data = 'train.csv'
# dataframe = pd.read_csv(input_data)
# x = dataframe.iloc[:, :].values

# x_encoded = []

# encoder = []
# for i, item in enumerate(input_data[0]):
#     if item.isdigit():
#         x_encoded[:,i] = input_data[:,i]
#     else:
#         encoder.append(preprocessing.LabelEncoder())
#         x_encoded[:,i] = encoder[-1].fit_transform(input_data[:,i])
# x = x_encoded.remove[:,2]
# print(x.head(2))


# df['Cabin'] = df['Cabin'].replace(' ','?')

# print(dataframe['Survived'].value_counts())
# print(dataframe['Cabin'].unique())
# print(dataframe.replace('NaN','?').head(2))

#ПОПЫТКА № 2
# xtrain, ytrain = df.drop('Survived', axis = 1), df['Survived']
# true_value = xtrain[:,:]
# print(true_value)


}

{#ПОПЫТКА № 1
#count_class1 = 0
#count_class2 = 0
#max_datapoints = 560
#цикл для заполнения
# # with open(input_data, 'r') as f:
# #     for line in f.readlines():
# #         if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
# #             break
# #         if "nan" in line:
# #             continue 
# #         data = line.split(',')
# #         if data[1] == '0' and count_class1 <= max_datapoints:
# #             x.append(data)
# #             count_class1 += 1
# #         if data[1] == '1' and count_class2 <= max_datapoints:
# #             x.append(data)
# #             count_class2 += 1
# }


# # x = np.array(x)



# label_encoder = []
# x_encoded = np.empty(x.shape)
# for i, item in enumerate(x[0]):
#     if item.isdigit():
#         x_encoded[:, i] = x[:, i]
#     else:
#         label_encoder.append(preprocessing.LabelEncoder())
#         x_encoded[:,i] = label_encoder[-1].fit_transform(x[:, i])

# x = x_encoded[:,:1:].astype(int)
# y = x_encoded[:,1].astype(int)

# svmclassifier = OneVsOneClassifier(LinearSVC(random_state=0))
# svmclassifier.fit(x,y)
# #cross_validation

# testing_data = 'test.csv'
# dataframe2 = pd.read_csv(testing_data)
# dataframe2['PassengerId'].value_counts()
# xtest = []



# svmclassifier = OneVsOneClassifier(LinearSVC(random_state=0))
# svmclassifier.fit(x, y)

# y_test_pred = svmclassifier.predict()
} 