import pandas as pd
import numpy as np

# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# EDA
import seaborn as sns
import matplotlib.pyplot as plt

# Feature Engineering
from sklearn.feature_selection import SelectKBest, f_classif


# Training
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

df = pd.read_csv('hospital_readmissions.csv')

import pandas as pd
import numpy as np

# Data processing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# EDA
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('hospital_readmissions.csv')
df.isnull().sum()
df.fillna(df.median(), inplace=True)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

X = df_scaled.drop('readmitted_yes', axis=1)
Y = df_scaled['readmitted_yes']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)

corr_matrix = df.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

strong_corr = [column for column in upper.columns if any(upper[column] > 0.5)]
print("Strongly correlated features:", strong_corr)