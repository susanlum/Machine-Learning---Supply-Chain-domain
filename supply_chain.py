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
