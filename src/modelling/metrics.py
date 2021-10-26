import numpy as np
import pandas as pd

from typing import Tuple

def calculate_profit(y_true: pd.Series, y_pred: np.array, odds: pd.DataFrame) -> Tuple[float, float, float, float, float]:
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

def calculate_mean_mean_profit(y_true: pd.Series, y_pred: np.array, odds: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    odds = odds.loc[y_true.index, ['Home_Win_Odds', 'Draw_Odds', 'Away_Win_Odds']]


    mean_profit = list()
    for i, (true, pred) in enumerate(zip(y_true.values, y_pred)):
        odd = odds.iloc[i]
        if true == pred:
            if true == 1:
                mean_profit.append(odd['Home_Win_Odds'] - 1)
            elif true == 0:
                mean_profit.append(odd['Draw_Odds'] - 1)
            else:
                mean_profit.append(odd['Away_Win_Odds'] - 1)
        else:
            mean_profit.append(-1)
    
    mean_mean_profit = np.round(np.nanmean(mean_profit) * 100, 3)

    #print(f'MEAN MEAN PROFIT PER MATCH -> {mean_mean_profit}%')

    return mean_mean_profit