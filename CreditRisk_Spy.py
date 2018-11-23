# -*- coding: utf-8 -*-
#import libraries 
import pandas as pd #for data visualisation 
#open the data file
with open("C:\\Users\\Prasad Matta\\Desktop\\Prasad_matta\\german.data.txt") as f:
    data = [x.strip().split('\t') for x in f]
#Convert the data file to Dataframe for data manipulation 
data_1=[]
for i in data:
    for k in i:
        data_1.append(k.split())
Check=[]
Duration=[]
Credit_history=[]
Purpose=[]
Credit_amount=[]
Savings_account=[]
Experience = []
Disposible_income=[]
Status_and_gender=[]
Other_debtors=[]
Residence=[]
Property=[]
Age=[]
Plan=[]
Housing=[]
No_of_cards=[]
Job=[]
No_of_people=[]
Telephone=[]
Foreign_worker=[]
Risk=[]
for i in data_1:
    Check.append(i[0])
    Duration.append(int(i[1]))
    Credit_history.append(i[2])
    Purpose.append(i[3])
    Credit_amount.append(int(i[4]))
    Savings_account.append(i[5])
    Experience.append(i[6])
    Disposible_income.append(int(i[7]))
    Status_and_gender.append(i[8])
    Other_debtors.append(i[9])
    Residence.append(int(i[10]))
    Property.append(i[11])
    Age.append(int(i[12]))
    Plan.append(i[13])
    Housing.append(i[14])
    No_of_cards.append(int(i[15]))
    Job.append(i[16])
    No_of_people.append(int(i[17]))
    Telephone.append(i[18])
    Foreign_worker.append(i[19])
    Risk.append(int(i[20]))
dataset = pd.DataFrame({'Checking_account':Check,
                        'Duration':Duration,
                        'Credit history':Credit_history,
                        'Purpose':Purpose,
                        'Credit_amount':Credit_amount,
                        'Savings account':Savings_account,
                        'Experience':Experience,
                        'Disposible_income':Disposible_income,
                        'Status and gender':Status_and_gender,
                        'Other debtors':Other_debtors,
                        'Residence':Residence,
                        'Property':Property,
                        'Age':Age,
                        'Plan':Plan,
                        'Housing':Housing,
                        'No_of_cards':No_of_cards,
                        'Job':Job,
                        'No_of_people':No_of_people,
                        'Telephone':Telephone,
                        'Forgien_worker':Foreign_worker,
                        'Risk':Risk})
#Split the dataframe from Target variable
data = dataset.copy()
X = data.iloc[:,:20]
y = data.iloc[:,-1]

#int value attributes
c_names = ['Duration','Credit_amount','Disposible_income','No_of_cards','No_of_people','Age','Residence','Risk']

#Encode categorical data to numerical data for further calculations 
from sklearn.preprocessing import LabelEncoder
for i in (X.columns):
    if i not in c_names:
        encode=LabelEncoder()
        X[i]=encode.fit_transform(X[i])

#Split data for train and test
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X,y, random_state = 0, test_size=0.3)

#feature scaling for simplification
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(x_test)

#Fitting random forest classifier for training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',n_estimators=15)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#find the accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)*100
