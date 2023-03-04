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


