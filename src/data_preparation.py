import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def preprocess(df):
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X = pd.get_dummies(X, drop_first=True)
    return X, y
