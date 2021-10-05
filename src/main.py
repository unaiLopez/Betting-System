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
    df = pd.read_csv('../data/All_Matches.csv', parse_dates=['Date'])

    columns_to_mantain = [
                          'HomeTeam', 'AwayTeam', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
                          'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score' , 'Away Budget'
                         ]
    df = df[columns_to_mantain]
    df = df.dropna(axis=0, how='any')
    df.Full_Time_Result = df.Full_Time_Result.apply(full_time_result_to_class)
    df['Difference_Overall_Score'] = df['Home Overall Score'] - df['Away Overall Score']
    df['Difference_Attack_Score'] = df['Home Attack Score'] - df['Away Attack Score']
    df['Difference_Middle_Score'] = df['Home Middle Score'] - df['Away Middle Score']
    df['Difference_Defensive_Score'] = df['Home Defensive Score'] - df['Away Defensive Score']
    df['Difference_Budget'] = df['Home Budget'] - df['Away Budget']

    df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']
    print(X.columns)
    print(X.shape)
    print(y.shape)

    model = RandomForestClassifier(random_state=42)
    score = cross_val_score(model, X, y, scoring=make_scorer(accuracy_score, greater_is_better=True), cv=10)

    print(np.mean(score))