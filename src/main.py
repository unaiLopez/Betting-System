import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

def full_time_result_to_class(result):
    if result == 'H':
        return 1
    elif result == 'D':
        return 0
    elif result == 'A':
        return 2

if __name__ == '__main__':
    df = pd.read_csv('../data/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])

    df = df.dropna(axis=0, how='any')
    df = df[df['Date'] > '2009-05-06']

    df.drop(['Date', 'HomeTeam', 'AwayTeam'], axis=1, inplace=True)

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']

    model = RandomForestClassifier(random_state=42)
    score = cross_val_score(model, X, y, scoring=make_scorer(accuracy_score, greater_is_better=True), cv=10)

    print(score)
    print(np.mean(score))