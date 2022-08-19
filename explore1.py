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
from acquire import *
from prepare import *
from explore import *
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

    df.columns = df.columns.map(" ".join)

    corr_mat = df.corr().stack().reset_index(name="correlation")

    g = sns.relplot(
        data=corr_mat,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=12, sizes=(100, 250), size_norm=(-.2, .8),
    )

    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
    
def cat_test(train, target, cat_type_list):
    
    for col in cat_type_list:

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
    