import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def full_time_result_to_class(result):
    if result == 'H':
        return 1
    elif result == 'D':
        return 0
    elif result == 'A':
        return 2

if __name__ == '__main__':
    df = pd.read_csv('../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])

    for column in df.columns:
        print(column)

    df = df.dropna(axis=0, how='any')
    df = df[df['Date'] > '2009-05-06']
    columns_to_mantain = [
        'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO',
        'Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds'
    ]

    df = df[columns_to_mantain]
    print(df.isna().sum())

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']

    model = RandomForestClassifier(random_state=42)
    #model = Pipeline([
    #    ('scaler', StandardScaler()),
    #    ('logreg', LogisticRegression(random_state=42))
    #])
    #model = Pipeline([
    #    ('scaler', StandardScaler()),
    #    ('svc', SVC(random_state=42))
    #])
    score = cross_val_score(model, X, y, scoring=make_scorer(accuracy_score, greater_is_better=True), cv=10)

    print(score)
    print(np.mean(score))