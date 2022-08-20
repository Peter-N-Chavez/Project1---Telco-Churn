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
from explore1 import *
import env
import os

def prep_telco(telco):

    df = telco

    telco = telco.drop(columns=["internet_service_type_id", "contract_type_id", "payment_type_id"])

    telco["gender_encoded"] = telco.gender.map({"Female": 1, "Male": 0})
    telco["partner_encoded"] = telco.partner.map({"Yes": 1, "No": 0})
    telco["dependents_encoded"] = telco.dependents.map({"Yes": 1, "No": 0})
    telco["phone_service_encoded"] = telco.phone_service.map(\
                                                                {"Yes": 1, "No": 0})
    telco["paperless_billing_encoded"] = telco.paperless_billing.map(\
                                                                {"Yes": 1, "No": 0})
    telco["churn_encoded"] = telco.churn.map({"Yes": 1, "No": 0})
    telco["total_charges"] = telco.total_charges.str.replace(" ", "0", 
                                                                regex = True).\
                                        str.replace("$", "", regex = True).\
                                        str.replace(",", "_", regex = True).\
                                        str.replace("'", "", regex = True).\
                                        astype(float)

    dummy_df = pd.get_dummies(telco[["multiple_lines", \
                            "online_security", \
                            "online_backup", \
                            "device_protection", \
                            "tech_support", \
                            "streaming_tv", \
                            "streaming_movies", \
                            "contract_type", \
                            "internet_service_type", \
                            "payment_type"
                        ]],
                            drop_first=True)
    telco = pd.concat([telco, dummy_df], axis=1)
    return telco

def prep_split(df, target):
    
    train, test = train_test_split(df, test_size=.2, random_state=269, 
    stratify = df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=269, stratify = train[target])
    
    return train, validate, test

def impute_col_nan(df, column, strat):
    imp_col = SimpleImputer(missing_values= np.nan, strategy=(strat))
    df[[column]] = imp_col.fit_transform(df[[column]])
    return df