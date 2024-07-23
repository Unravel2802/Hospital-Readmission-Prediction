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
threshold = 0.5  # Adjust based on your dataset and problem
Y_binary = (Y > threshold).astype(int)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y_binary, test_size = 0.2, random_state=42)

df['new_feature'] = df['glucose_test_normal'] / df['A1Ctest_normal']  # Example: create a new feature


selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X,Y)

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

for name, model in models.items():
    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print(f'{name}: {scores.mean()}')



best_model = models['Random Forest']
best_model.fit(X_train, Y_train)
y_pred = best_model.predict(X_test)

print('Accuracy:', accuracy_score(Y_test, y_pred))
print('ROC-AUC:', roc_auc_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
