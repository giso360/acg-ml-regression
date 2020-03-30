import processRB as prb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def describe_all(df):
    with pd.option_context('display.max_columns', 40):
        print(df.describe(include='all'))


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
describe_all(df_PT)
# Redbook Dataset
print("\nRB Info:\n")
print(df_RB.info(verbose=True))
print("\nRB Describe:\n")
describe_all(df_RB)
print("====================")
print()
print("3.   Data Visualization\n")

XY_age_PT = df_PT.drop(['ID', 'gender', 'yearsmarried', 'children', 'religiousness',
                        'education', 'occupation', 'rating'], axis=1)
#
# dummy_df = df_PT.select_dtypes(exclude=['object'])
# dummy_df = dummy_df.drop(["ID"], axis=1)
# common_array_of_fields = np.array(dummy_df.columns).tolist()
#
# i = 0
# colors = ["", "teal", "lime", "blue", "coral", "c", "tomato", "darksalmon"]
# plt.figure(figsize=(25, 25)).suptitle('Scatter Plots for PT survey (attributes Vs target) with regression lines')
# plt.subplots_adjust(hspace=0.5)
# for field in common_array_of_fields:
#     copy = np.array(dummy_df.columns).tolist()
#     copy.remove("affairs")
#     i = i + 1
#     plhd = ["affairs"]
#     if field == "affairs":
#         i = 0
#         continue
#     else:
#         copy.remove(field)
#         XY_term_PT = dummy_df.drop(copy, axis=1)
#         plt.subplot(3, 3, i)
#         sns.regplot(field, "affairs", data=XY_term_PT, scatter_kws={"color": colors[i], "alpha": 0.05, "s": 200})
#
#
# plt.show()
# sns.regplot("age", "affairs", data=XY_age_PT, scatter_kws={"color":"lime","alpha":0.05,"s":1000})
# plt.xlim(15, 65)
#
# plt.show()
print("\n====================")
print("4.   Data Preprocessing")
# sns.distplot(df_PT['yearsmarried'],color="blue")
# sns.countplot(df_PT['children'])
# plt.subplot(3,3,6)
# sns.countplot(df_PT['religiousness'],color="coral")
# plt.subplot(3,3,7)
# sns.countplot(df_PT['education'],color="darksalmon")
# plt.subplot(3,3,8)
# sns.countplot(df_PT['occupation'],color="c")
# plt.subplot(3,3,9)
# sns.countplot(df_PT['rating'],color="tomato")



# notes
# notes
# notes
# notes


# plt.figure(figsize=(25,25)).suptitle('Scatter Plots for PT survey (attributes Vs target)')
# plt.subplot(3,3,1)
# sns.regplot(x=df_PT['age'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"darkred","alpha":0.05,"s":200})
# plt.subplot(3,3,2)
# sns.regplot(x=df_PT['yearsmarried'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"blue","alpha":0.05,"s":200})
# plt.subplot(3,3,3)
# sns.regplot(x=df_PT['religiousness'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"tomato","alpha":0.05,"s":200})
# plt.subplot(3,3,4)
# sns.regplot(x=df_PT['education'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"teal","alpha":0.05,"s":200})
# plt.subplot(3,3,5)
# sns.regplot(x=df_PT['occupation'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"lime","alpha":0.05,"s":200})
# plt.subplot(3,3,6)
# sns.regplot(x=df_PT['rating'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"darksalmon","alpha":0.05,"s":200})
# plt.subplot(3,3,7)
# sns.regplot(x=df_PT['children'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"color":"c","alpha":0.05,"s":200})
# plt.subplot(3,3,8)
# sns.regplot(x=df_PT['gender'], y=df_PT["affairs"], fit_reg=False, scatter_kws={"alpha":0.05,"s":200})
# plt.show()

# notes
# notes
# notes
# notes



