# Credit Risk
The dataset contains information of about a thousand individuals, we have to create a model that attempts to classify their credit risk.
  
 
Dataset:  https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29 
https://www.kaggle.com/uciml/german-credit  

PREPROCESSING :
STEP 1: This dataset contains no missing values there are some outliers these are left in the dataset because they are not due to measurement errors, and credit risk is part of the problem so the data not excluded.
STEP 2: I converted categorical variables to numeric by label encoding.
STEP 3: I normalised the data and divided into test and train by 30:70
STEP 4: Build the model using random forest and got accuracy 77.6%
 
