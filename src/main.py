import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def full_time_result_to_class(result: pd.Series) -> int:
    if result == 'H':
        return 1
    elif result == 'D':
        return 0
    elif result == 'A':
        return 2

def calculate_profit(y_true, y_pred, odds):
    odds = odds.loc[
        y_true.index, [
                        'Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
                        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
                        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds'
                      ]
                   ]

    profit_vc = list()
    profit_bet365 = list()
    profit_bet_and_win = list()
    profit_interwetten = list()
    profit_william_hill = list()

    for i, (true, pred) in enumerate(zip(y_true.values, y_pred)):
        odd = odds.iloc[i]
        if true == pred:
            if true == 1:
                profit_vc.append(odd['VCBet_Home_Win_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Home_Win_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Home_Win_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Home_Win_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Home_Win_Odds'] - 1)
            elif true == 0:
                profit_vc.append(odd['VCBet_Draw_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Draw_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Draw_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Draw_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Draw_Odds'] - 1)
            else:
                profit_vc.append(odd['VCBet_Away_Win_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Away_Win_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Away_Win_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Away_Win_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Away_Win_Odds'] - 1)
        else:
            profit_vc.append(-1)
            profit_bet365.append(-1)
            profit_bet_and_win.append(-1)
            profit_interwetten.append(-1)
            profit_william_hill.append(-1)
    
    print(f'PROFIT VCBET -> {np.mean(profit_vc) * 100}')
    print(f'PROFIT BET365 -> {np.mean(profit_bet365) * 100}')
    print(f'PROFIT BET AND WIN -> {np.mean(profit_bet_and_win) * 100}')
    print(f'PROFIT INTERWETTEN -> {np.mean(profit_interwetten) * 100}')
    print(f'PROFIT WILLIAM HILL -> {np.mean(profit_william_hill) * 100}')

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
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

    return df

if __name__ == '__main__':
    df = pd.read_csv('../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])
    df = prepare_data(df)

    odds = df[['Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds']]
    
    df = df[['Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO']]

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(len(y_test))

    #model = RandomForestClassifier(random_state=42)
    model = Pipeline([
        ('cluster', KMeans(n_clusters=5)),
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(random_state=42))
    ])
    #model = Pipeline([
    #    ('scaler', StandardScaler()),
    #    ('svc', SVC(random_state=42))
    #])
    score = cross_val_score(model, X_train, y_train, scoring=make_scorer(accuracy_score, greater_is_better=True), cv=10)
    print(score)
    print(np.mean(score))

    #print(score)
    #print(np.mean(score))


    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(accuracy_score(y_test.values, preds))

    calculate_profit(y_test, preds, odds)

    index = y_test.index