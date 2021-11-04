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
from prepare_data import prepare_data
from metrics import calculate_profit_per_booker, calculate_profit_all_bookers_mean_profit, get_all_booker_odd_values

def inference(model: object, X_test: pd.DataFrame, y_test: pd.Series, odds: pd.DataFrame, threshold: float):
    preds = model.predict(X_test)        
    chosen_preds = np.where(model.predict_proba(X_test)[:, 1] > threshold, 1, 0)
    
    index = y_test.index
    preds = pd.Series(preds, index=index)
    chosen_preds = pd.Series(chosen_preds, index=index)
    preds = pd.concat([preds, chosen_preds], axis=1)
    preds.columns = ['PREDICTION', 'MASK']
    preds = preds[preds.MASK != 0]
    
    if preds.empty:
        my_profit = 0.0
    else:
        y_test = y_test[y_test.index.isin(preds.index)]
        preds = preds.PREDICTION.values
        my_profit = calculate_profit_all_bookers_mean_profit(y_test, preds, odds)
    
    return y_test, preds, my_profit

def objective(trial: object, model: object, X: pd.DataFrame, y: pd.Series, odds: pd.DataFrame, folds: List[Tuple]) -> float:
    threshold = trial.suggest_float("threshold", 0.4, 0.7)

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
        
        _, _, my_profit = inference(model, X_test, y_test, odds, threshold)
        
        profit.append(my_profit)

    return np.nanmean(profit)

def fine_tune(model, X_train: pd.DataFrame, y_train: pd.Series, folds: List[Tuple], odds: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    func = lambda trial: objective(trial, model, X_train, y_train, odds, folds)
    study = optuna.create_study(direction='maximize')
    study.optimize(func, timeout=360, n_jobs=-1)
    #study.optimize(func, n_trials=1, n_jobs=6)

    trials_dataframe = study.trials_dataframe().sort_values(['value'], ascending=False)
    trials_dataframe.to_csv('trials_dataframe.csv', index=False)

    best_params = study.best_params
    threshold = best_params['threshold']
    del best_params['threshold']

    model.set_params(**best_params)

    model.fit(X_train, y_train)
    y_test, preds, profit = inference(model, X_test, y_test, odds, threshold)

    print(f'BEST MEAN PROFIT PER MATCH: {study.best_value}%')
    print(f'PREDICTED NUMBER OF MATCHES {len(y_test)}')
    print(f'PROFIT TEST PER MATCH {profit}%')
    print(f'ACCURACY TEST {np.round(accuracy_score(y_test.values, preds), 3) * 100}%')

    os.makedirs('../../models', exist_ok=True)
    joblib.dump(model, '../../models/best_model.bin')
    joblib.dump(threshold, '../../models/best_threshold.bin')

def train_with_optuna():
    df = pd.read_csv('../../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])
    df = prepare_data(df)
    odds = get_all_booker_odd_values(df)
    
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
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO',
        'HOME_TRUESKILL_MU_NO_RESET', 'AWAY_TRUESKILL_MU_NO_RESET', 'HOME_TRUESKILL_SIGMA_NO_RESET', 'AWAY_TRUESKILL_SIGMA_NO_RESET', 'DRAW_CHANCE_NO_RESET',
        'HOME_TRUESKILL_MU_SEASON', 'AWAY_TRUESKILL_MU_SEASON', 'HOME_TRUESKILL_SIGMA_SEASON', 'AWAY_TRUESKILL_SIGMA_SEASON', 'DRAW_CHANCE_SEASON']]

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
        X = df.drop(['Full_Time_Result', 'Date', 'Season'], axis=1)
        y = df['Full_Time_Result']

        X_train, y_train = X.iloc[train], y.iloc[train]
        X_test, y_test = X.iloc[test], y.iloc[test]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = np.round(accuracy_score(y_test.values, preds), 3)
        accuracies.append(accuracy)

        vcbet, bet365, bet_and_win, interwetten, william_hill = calculate_profit_per_booker(y_test, preds, odds)
        
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
    # train_with_optuna()
    train_without_optuna()