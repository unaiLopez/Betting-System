import optuna
import joblib
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from typing import List, Tuple

from create_folds import create_folds
from metrics import calculate_profit, calculate_mean_mean_profit

def full_time_result_to_class(result: pd.Series) -> int:
    if result == 'H':
        return 1
    elif result == 'D':
        return 0
    elif result == 'A':
        return 2

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Date'] > '2009-05-06']

    columns_to_mantain = ['Date', 'Season',
        'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO',
        'Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds'
    ]

    df = df[columns_to_mantain]

    return df

def objective(trial: object, model: object, X: pd.DataFrame, y: pd.Series, odds: pd.DataFrame, folds: List[Tuple]) -> float:
    params = {
        "lgbm__verbosity": -1,
        "lgbm__boosting_type": "gbdt",
        "lgbm__n_estimators": trial.suggest_int("lgbm__n_estimators", 200, 800),
        "lgbm__lambda_l1": trial.suggest_float("lgbm__lambda_l1", 1e-8, 10.0, log=True),
        "lgbm__lambda_l2": trial.suggest_float("lgbm__lambda_l2", 1e-8, 10.0, log=True),
        "lgbm__num_leaves": trial.suggest_int("lgbm__num_leaves", 2, 256),
        "lgbm__feature_fraction": trial.suggest_float("lgbm__feature_fraction", 0.4, 1.0),
        "lgbm__bagging_fraction": trial.suggest_float("lgbm__bagging_fraction", 0.4, 1.0),
        "lgbm__bagging_freq": trial.suggest_int("lgbm__bagging_freq", 1, 7),
        "lgbm__min_child_samples": trial.suggest_int("lgbm__min_child_samples", 5, 100),
    }

    model.set_params(**params)
    
    profit = list()
    for (train, test) in folds:
        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        my_profit = calculate_mean_mean_profit(y_test, preds, odds)
        
        profit.append(my_profit)

    return np.round(np.nanmean(profit), 3)

def fine_tune(model, X_train: pd.DataFrame, y_train: pd.Series, folds: List[Tuple], odds: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    func = lambda trial: objective(trial, model, X_train, y_train, odds, folds)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, timeout=800, n_jobs=-1)

    model.set_params(**study.best_params)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    profit = calculate_mean_mean_profit(y_test, preds, odds)

    print(f'BEST MEAN PROFIT PER MATCH: {study.best_value}%')
    print(f'PREDICTED NUMBER OF MATCHES {len(y_test)}')
    print(f'PROFIT TEST PER MATCH {profit}%')
    print(f'ACCURACY TEST {np.round(accuracy_score(y_test.values, preds), 3) * 100}%')

    trials_dataframe = study.trials_dataframe().sort_values(by='value', ascending=False)
    trials_dataframe.to_csv('trials_dataframe.csv', index=False)

    os.makedirs('../../models', exist_ok=True)
    joblib.dump(model, '../../models/best_model.bin')

def train_with_optuna():
    df = pd.read_csv('../../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])
    df = prepare_data(df)

    odds_to_mantain = ['Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds']
    odds = df[odds_to_mantain]
    
    odds['Home_Win_Odds'] = np.nanmean(odds[['Bet365_Home_Win_Odds', 'BetAndWin_Home_Win_Odds', 'Interwetten_Home_Win_Odds', 'WilliamHill_Home_Win_Odds', 'VCBet_Home_Win_Odds']], axis=1)
    odds['Away_Win_Odds'] = np.mean(odds[['Bet365_Away_Win_Odds', 'BetAndWin_Away_Win_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Away_Win_Odds', 'VCBet_Away_Win_Odds']], axis=1)
    odds['Draw_Odds'] = np.mean(odds[['Bet365_Draw_Odds', 'BetAndWin_Draw_Odds', 'Interwetten_Draw_Odds', 'WilliamHill_Draw_Odds', 'VCBet_Draw_Odds']], axis=1)
    odds.drop(odds_to_mantain, axis=1, inplace=True)
    
    df = df[['Date', 'Season', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO']]

    df['Mes'] = df.Date.dt.month
    df['Dia'] = df.Date.dt.dayofweek

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, shuffle=False)

    df_train = pd.concat([X_train, y_train], axis=1)

    folds = create_folds(df_train)

    model = Pipeline([
        ('lgbm', LGBMClassifier(random_state=42))
    ])

    X_train.drop(['Date', 'Season'], axis=1, inplace=True)
    X_test.drop(['Date', 'Season'], axis=1, inplace=True)
    
    fine_tune(model, X_train, y_train, folds, odds, X_test, y_test)

def train_without_optuna():
    df = pd.read_csv('../../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])
    df = prepare_data(df)

    odds = df[['Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds']]
    
    df = df[['Date', 'Season', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO']]

    df['Mes'] = df.Date.dt.month
    df['Dia'] = df.Date.dt.dayofweek

    X = df.drop('Full_Time_Result', axis=1)
    y = df['Full_Time_Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    folds = create_folds(df_train)

    df_train.drop(['Date', 'Season'], axis=1, inplace=True)
    df_test.drop(['Date', 'Season'], axis=1, inplace=True)

    model = Pipeline([
        #('sampling', SMOTE(k_neighbors=3, random_state=42)),
        ('lgbm', LGBMClassifier(n_estimators=500, random_state=42))
    ])

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

if __name__ == '__main__':
    train_with_optuna()
