import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report
import matplotlib.pyplot as plt 

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

df = pd.read_csv(
    "/workspace/adult-all.csv",
    header=None,
    names = columns 
)

df = df.apply(
    lambda x: x.str.strip() if x.dtype == "object" else x 
)

df.replace("?",np.nan,inplace = True)
df.dropnaa(inplace = True)

categorical_columns = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income"
]


