from numpy import vectorize
from pygments.lexer import include

import processRB as prb
import affais_util as util
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# create csv file with headers from bibliography (https://fairmodel.econ.yale.edu/vote2012/affairs.txt) for redbook
# survey
prb.create_affairs_rb_csv()

print("====================")
print("1.   Load Data\n")
# Psychology Today Dataset
print("Loading dataframe for Psychology Today (PT) dataset from file: AffairsPT.csv\n")
df_PT = pd.read_csv("AffairsPT.csv")
# Redbook Dataset
print("Loading dataframe for Redbook (RB) dataset from file: ", prb.new_file_name, "\n")
df_RB = pd.read_csv(prb.new_file_name)
print("====================")
print()
print("====================")
print("2.   Describe Datasets")
# Psychology Today Dataset
print("\nPT Info:\n")
print(df_PT.info(verbose=True))
print("\nPT Describe:\n")
util.describe_all(df_PT)
# Redbook Dataset
print("\nRB Info:\n")
print(df_RB.info(verbose=True))
print("\nRB Describe:\n")
util.describe_all(df_RB)
print("====================")
print()
print("3.   Data Visualization\n")
print("\nA.   Histograms\n")
plt.figure(figsize=(25,25)).suptitle('Raw Data Histograms for PT survey')
plt.subplot(3,3,1)
sns.countplot(df_PT['affairs'],color="teal")
plt.subplot(3,3,2)
sns.countplot(df_PT['gender'])
plt.subplot(3,3,3)
sns.distplot(df_PT['age'],color="lime")
plt.subplot(3,3,4)
sns.distplot(df_PT['yearsmarried'],color="blue")
plt.subplot(3,3,5)
sns.countplot(df_PT['children'])
plt.subplot(3,3,6)
sns.countplot(df_PT['religiousness'],color="coral")
plt.subplot(3,3,7)
sns.countplot(df_PT['education'],color="darksalmon")
plt.subplot(3,3,8)
sns.countplot(df_PT['occupation'],color="c")
plt.subplot(3,3,9)
sns.countplot(df_PT['rating'],color="tomato")
plt.show()
print("\nB. Seaborn Regression with regression lines \n")
seaborn_regression_data_per_field = util.regression_plots_one_independent_variable(df_PT)
print("\nOne variable study regression using seaborn regplot tool:\n")
print(seaborn_regression_data_per_field)
print()
# TODO: embed regression line equation in plots
print("\nC. Field correlation study \n")
print()
df_PT = df_PT.drop(["ID"], axis=1)  # remove ID feature
util.draw_correlation_heatmap(df_PT)
print("Feature engineering: 'agemarried' (age that one got married) => age-ym")
print("Assists to remove false data/outliers")
df_PT["agemarried"] = df_PT["age"] - df_PT["yearsmarried"]
print(df_PT.shape)
df_PT = df_PT[df_PT['agemarried'] >= 15.0]  #remove people that were married extremely young
print(df_PT.shape)
print(df_PT.info())
util.describe_all(df_PT)
util.draw_correlation_heatmap(df_PT)
print("\n====================")
print("4.   Data Preprocessing")
print("\nA.   apply ohe for categorical data: \n")
#df_PT = pd.get_dummies(df_PT, columns=["gender", "children"])
df_PT.replace(["male"], 0, inplace=True)
df_PT.replace(["female"], 1, inplace=True)
util.draw_correlation_heatmap(df_PT)
print(df_PT.info())
util.describe_all(df_PT)
print(df_PT.corr())
print("\nB.   split data to test and training sets: \n")
X = df_PT.drop(["affairs"], axis=1)     #dataframe for independent variables
y = df_PT[["affairs"]]                  #dataframe for dependent variables
yy = y                                  #create copy for subsequent PCA operations
# use X, Y dataframes to split test and train data (80/20, consistent randomization)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
print("5.   Model Generation")
print("A.   one variable cascade linear regression analysis")
# util.perform_one_variable_regression_linear(df_PT, X_train, X_test, y_train, y_test)
print()
#util.perform_one_variable_regression_linear(X, X_train, X_test, y_train, y_test)
print("==========================================")
print("==========================================")
print("==========================================")
print()
