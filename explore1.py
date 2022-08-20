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
from acquire1 import *
from prepare1 import *
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

def cat_vis(train, cat_type_list, resultant):

    # Plotting a heatmap
    mask = np.triu(np.ones_like(resultant, dtype=np.bool))    
    fig = plt.figure(figsize=(17,14))
    sns.heatmap(resultant, annot=True, mask=mask, center = .1,fmt = ".1f", cmap='cubehelix')
    plt.title('Chi-Square Test Results')
    plt.show()
    
    
def cat_test(train, target, cat_type_list):
    
    for col in cat_type_list:
        if col != target:
            α = 0.05
            null_hyp = col + " and " + target + " are independent."
            alt_hyp = "There appears to be a relationship between " + target + " and " + col + "."
            observed = pd.crosstab(train[target], train[col])
            chi_square, p, degf, expected = stats.chi2_contingency(observed)
            if p < α:
                print("We reject the null hypothesis that", null_hyp)
                print(alt_hyp)
                print()
            else:
                print("We fail to reject the null hypothesis that", null_hyp)
                print("There appears to be no relationship between ", target, "and ", col, ".")
                print()

    df = train

    # Resultant Dataframe will be a dataframe where the column names and Index will be the same
    # This is a matrix similar to correlation matrix which we get after df.corr()
    # Initialize the values in this matrix with 0
    resultant = pd.DataFrame(data=[(0 for i in range(len(cat_type_list))) for i in range(len(cat_type_list))], 
                            columns=cat_type_list)
    resultant.set_index(pd.Index(cat_type_list), inplace = True)

    # Finding p_value for all columns and putting them in the resultant matrix
    for i in cat_type_list:
        for j in cat_type_list:
            if i != j:
                chi2_val, p_val = chi2(np.array(df[i]).reshape(-1, 1), np.array(df[j]).reshape(-1, 1))
                resultant.loc[i,j] = p_val
    return resultant

def cat_analysis(train, target, cat_type_list):

    resultant = cat_test(train, target, cat_type_list)
    cat_vis(train, target, resultant)
    