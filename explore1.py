import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import urllib.request
from PIL import Image
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from acquire import *
from prepare import *
import env

def dtypes_to_list(df):

    num_type_list, cat_type_list = [], []

    for column in df:

        col_type =  df[column].dtype

        if col_type == "object" : 
        
            cat_type_list.append(column)
    
        if np.issubdtype(df[column], np.number) and \
             ((df[column].max() + 1) / df[column].nunique())  == 1 :

            cat_type_list.append(column)

        if np.issubdtype(df[column], np.number) and \
            ((df[column].max() + 1) / df[column].nunique()) != 1 :

            num_type_list.append(column)

    return num_type_list, cat_type_list

def cat_vis(train, cat_type_list):

    sns.set_theme(style="whitegrid")

    df = train

    used_columns = cat_type_list
    df = df.loc[:, used_columns]

    plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
    heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='Spectral')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16)
    
def cat_test(train, target, cat_type_list):
    
    for col in cat_type_list:
        if col != target:
            α = 0.05
            null_hyp = col + " and " + target + " are independent."
            alt_hyp = "There appears to be a relationship between " + target + " and " + col + "."
            observed = pd.crosstab(train[target], train[col])
            chi2, p, degf, expected = stats.chi2_contingency(observed)
            if p < α:
                print("We reject the null hypothesis that", null_hyp)
                print(alt_hyp)
                print()
            else:
                print("We fail to reject the null hypothesis that", null_hyp)
                print("There appears to be no relationship between ", target, "and ", col, ".")
                print()

def cat_analysis(train, target, cat_type_list):

    cat_vis(train, cat_type_list)
    cat_test(train, target, cat_type_list)
    