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

# **Insights:** Based on the barplots above, high product importance does not affect customer rating. However, when product importance is high, the products have not reached on time.

