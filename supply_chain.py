#-------------------------------------------------------------------------------------------------------------------------------------------#
# Load & Overview of Data
#-------------------------------------------------------------------------------------------------------------------------------------------#

# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler # for RobustScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix # to create a confusion matrix
from sklearn.metrics import plot_confusion_matrix # to draw a confusion matrix

# read the train data
df = pd.read_csv('train.csv')
df 

# Check for datatypes 
df.dtypes

# Check for the null values
df.isna().sum()

# Check for duplicates 
df.duplicated().sum()

df.info()

df['Reached.on.Time_Y.N'].value_counts()

#-------------------------------------------------------------------------------------------------------------------------------------------#
# Data Visualisation
#-------------------------------------------------------------------------------------------------------------------------------------------#

g = sns.pairplot(df,hue="Reached.on.Time_Y.N", palette="husl")

# To perform EDA to answer the following questions:-

# 1) What was Customer Rating? And was the product delivered on time?

# To plot number of Customer Rating with labels from 1 (Worst) to 5 (Best). 
sns.countplot(x = df['Customer_rating']);

# To plot number of Reached On Time where 1 Indicates that the product has NOT reached on time and 0 indicates it has reached on time.
sns.countplot(x = df['Reached.on.Time_Y.N']);

# To plot Customer Rating versus Reached On Time
sns.countplot(x='Customer_rating', data=df, hue='Reached.on.Time_Y.N')
plt.show()

# 2) Is Customer query is being answered?
sns.countplot(x = df['Customer_care_calls']);

# To plot Customer_care_calls versus Reached.on.Time_Y.N
sns.countplot(x='Customer_care_calls', data=df, hue='Reached.on.Time_Y.N')
plt.show()

# 3) If Product importance is high. having highest rating or being delivered on time?
# To plot Product Importance (Low, Medium, High) versus Customer Rating [from 1 (Worst) to 5 (Best)]
sns.countplot(x='Product_importance', data=df, hue='Customer_rating')
plt.show()

# To plot Product Importance (Low, Medium, High) versus Reached.on.Time_Y.N (where 1=product has NOT reached on time; 0=product has reached on time)
sns.countplot(x='Product_importance', data=df, hue='Reached.on.Time_Y.N')
plt.show()

# Insights: Based on the barplots above, high product importance does not affect customer rating. However, when product importance is high, 
# the products have not reached on time.

sns.set(rc = {'figure.figsize': (10, 8)})
sns.heatmap(df.corr(), cmap = 'PuOr', annot = True, vmin = -1, vmax = 1, center = 0);
# annot = True -> label heatmap with correlation number
# center=0 -> white colour at the centre

#-------------------------------------------------------------------------------------------------------------------------------------------#
# Split the Data into Dependent and Indepedent Variables
#-------------------------------------------------------------------------------------------------------------------------------------------#

# Make a new copy of columns used to make predictions (ie. x)
X = df.drop('Reached.on.Time_Y.N', axis=1).copy() 
X.head()

# Make a new copy of the column of data we want to predict
y = df['Reached.on.Time_Y.N'].copy()
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#-------------------------------------------------------------------------------------------------------------------------------------------#
# Encode the categorical variables
#-------------------------------------------------------------------------------------------------------------------------------------------#

```
If perform the encoding before the split, it will lead to data leakage (train-test contamination) In the sense, it will introduce new data 
(integers of Label Encoders) and use it for the models thus it will affect the end predictions results (good validation scores but poor in deployment).

After the train and validation data category already matched up, we can perform fit_transform on the train data, then only transform for the 
validation data - based on the encoding maps from train data.

Almost all feature engineering like standarisation, Normalisation etc should be done after train testsplit.
```

pip install category_encoders

import category_encoders as ce # for category encoders

# Encode the categorical variables
# We convert the categorical features to numerical through the leave one out encoder in categorical_encoders. 
# This leaves a single numeric feature in the place of each existing categorical feature. This is needed to apply the scaler to all features in the training data.
encoder = ce.LeaveOneOutEncoder(return_df=True)

X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
X_train_encoded.shape

#-------------------------------------------------------------------------------------------------------------------------------------------#
# Standardisation of Data
#-------------------------------------------------------------------------------------------------------------------------------------------#
# apply robust scaler 
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded, y_train)
X_train_scaled.shape

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled_df.describe()

X_test_scaled = scaler.transform(X_test_encoded)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)
X_test_scaled_df.describe()

#-------------------------------------------------------------------------------------------------------------------------------------------#
# Classification Models
#-------------------------------------------------------------------------------------------------------------------------------------------#

#==============#
# (1) KNN 
#==============#
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled_df, y_train)

y_pred = knn.predict(X_test_scaled_df)

knn.score(X_test_scaled_df, y_test)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV

parameters = {"n_neighbors": range(1, 100)}

gridsearch = GridSearchCV(KNeighborsClassifier(), parameters)

gridsearch.fit(X_train_scaled_df, y_train)

gridsearch.best_params_

y_pred = gridsearch.predict(X_test_scaled_df)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

GD = GridSearchCV(estimator=KNeighborsClassifier(),
             param_grid={'n_neighbors': range(1, 100),
                         'weights': ['uniform', 'distance']},cv=5)

GD.fit(X_train_scaled_df, y_train)

y_pred = GD.predict(X_test_scaled_df)

GD.best_params_

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

#============================#
# (2) Logistics Regression
#============================#
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train_scaled_df, y_train)

y_pred=logreg.predict(X_test_scaled_df)
y_pred

# import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

# define metrics
y_pred_proba = logreg.predict_proba(X_test_scaled_df)[:,1]
fpr1, tpr1, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1) 

# create ROC curve
plt.plot(fpr1,tpr1,label="AUC="+str(auc))

plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)

plt.show()

#============================#
# (3) Support Vector Machine
#============================#

from sklearn.svm import SVC
model = SVC()

model.fit(X_train_scaled_df, y_train)

model.score(X_test_scaled_df, y_test)

#============================#
# (4) Decision Tree
#============================#
# Import library
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree 

# Create a decision tree and fit it to the training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train_scaled_df, y_train)

# plot decision tree
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["On Time", "Not On Time"],
          feature_names= X.columns);

y_pred = clf_dt.predict(X_test_scaled_df)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# plot_confusion_matrix() will run the test data down the tree and draw a confusion matrix. 
plot_confusion_matrix(clf_dt, X_test_scaled_df, y_test, display_labels=["On Time", "Not On Time"])

from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(), 
    n_estimators=5, 
    max_samples=0.8, 
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train_scaled_df, y_train)
bag_model.oob_score_

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RandomForestClassifier(n_estimators=5), X_train_scaled_df, y_train, cv=5)
scores.mean()





