import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from typing import List, Tuple

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

    profit_vcbet = list()
    profit_bet365 = list()
    profit_bet_and_win = list()
    profit_interwetten = list()
    profit_william_hill = list()

    for i, (true, pred) in enumerate(zip(y_true.values, y_pred)):
        odd = odds.iloc[i]
        if true == pred:
            if true == 1:
                profit_vcbet.append(odd['VCBet_Home_Win_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Home_Win_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Home_Win_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Home_Win_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Home_Win_Odds'] - 1)
            elif true == 0:
                profit_vcbet.append(odd['VCBet_Draw_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Draw_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Draw_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Draw_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Draw_Odds'] - 1)
            else:
                profit_vcbet.append(odd['VCBet_Away_Win_Odds'] - 1)
                profit_bet365.append(odd['Bet365_Away_Win_Odds'] - 1)
                profit_bet_and_win.append(odd['BetAndWin_Away_Win_Odds'] - 1)
                profit_interwetten.append(odd['Interwetten_Away_Win_Odds'] - 1)
                profit_william_hill.append(odd['WilliamHill_Away_Win_Odds'] - 1)
        else:
            profit_vcbet.append(-1)
            profit_bet365.append(-1)
            profit_bet_and_win.append(-1)
            profit_interwetten.append(-1)
            profit_william_hill.append(-1)
    
    vcbet = np.round(np.nanmean(profit_vcbet) * 100, 3)
    bet365 = np.round(np.nanmean(profit_bet365) * 100, 3)
    bet_and_win = np.round(np.nanmean(profit_bet_and_win) * 100, 3)
    interwetten = np.round(np.nanmean(profit_interwetten) * 100, 3)
    william_hill = np.round(np.nanmean(profit_william_hill) * 100, 3)

    print(f'PROFIT VCBET PER MATCH -> {vcbet}%')
    print(f'PROFIT BET365 PER MATCH -> {bet365}%')
    print(f'PROFIT BET AND WIN PER MATCH -> {bet_and_win}%')
    print(f'PROFIT INTERWETTEN PER MATCH -> {interwetten}%')
    print(f'PROFIT WILLIAM HILL PER MATCH -> {william_hill}%\n')

    return vcbet, bet365, bet_and_win, interwetten, william_hill

def _create_folds(df: pd.DataFrame, n_folds: int) -> List[Tuple]:
    ss = ShuffleSplit(n_splits=n_folds, test_size=0.1, random_state=42)

    folds = list()
    for train_index, test_index in ss.split(df):
        folds.append((train_index, test_index))
    
    return folds

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Date'] > '2009-05-06']

    columns_to_mantain = ['Date',
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
    
    df = df[['Date', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO']]

    df['Mes'] = df.Date.dt.month
    df['Dia'] = df.Date.dt.dayofweek

    df.drop('Date', axis=1, inplace=True)

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']

    folds = _create_folds(df, n_folds=20)

    #model = RandomForestClassifier(random_state=42)
    model = Pipeline([
        #('cluster', KMeans(n_clusters=10)),
        #('scaler', StandardScaler()),
        ('logreg', LGBMClassifier(n_estimators=500, random_state=42))
    ])
    #model = Pipeline([
    #    ('scaler', StandardScaler()),
    #    ('svc', SVC(random_state=42))
    #])

    accuracies = list()
    profit_vcbet = list()
    profit_bet365 = list()
    profit_bet_and_win = list()
    profit_interwetten = list()
    profit_william_hill = list()
    for i, (train, test) in enumerate(folds):
        X = df.drop('Full_Time_Result', axis=1)
        y = df['Full_Time_Result']

        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = np.round(accuracy_score(y_test.values, preds), 3)
        accuracies.append(accuracy)

        vcbet, bet365, bet_and_win, interwetten, william_hill = calculate_profit(y_test, preds, odds)
        
        profit_vcbet.append(vcbet)
        profit_bet365.append(bet365)
        profit_bet_and_win.append(bet_and_win)
        profit_interwetten.append(interwetten)
        profit_william_hill.append(william_hill)

        vcbet = np.round(np.nanmean(profit_vcbet), 3)
        bet365 = np.round(np.nanmean(profit_bet365), 3)
        bet_and_win = np.round(np.nanmean(profit_bet_and_win), 3)
        interwetten = np.round(np.nanmean(profit_interwetten), 3)
        william_hill = np.round(np.nanmean(profit_william_hill), 3)

    print(f'MEAN FOLDS PROFIT VCBET PER MATCH -> {vcbet}%')
    print(f'MEAN FOLDS PROFIT BET365 PER MATCH -> {bet365}%')
    print(f'MEAN FOLDS PROFIT BET AND WIN PER MATCH -> {bet_and_win}%')
    print(f'MEAN FOLDS PROFIT INTERWETTEN PER MATCH -> {interwetten}%')
    print(f'MEAN FOLDS PROFIT WILLIAM HILL PER MATCH -> {william_hill}%\n')

    print(f'ALL FOLD ACCURACIES {accuracies}')
    print(f'MEAN FOLDS ACCURACY {np.round(np.mean(accuracies), 3)}%')

    #from tpot import TPOTClassifier
    #model = TPOTClassifier(generations=5, population_size=50, cv=folds, scoring='accuracy', verbosity=2, random_state=42, n_jobs=-1)
    #model.fit(X, y)
