import pandas as pd

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Date'] > '2009-05-06']

    columns_to_mantain = [
        'Date', 'Season', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
        'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
        'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_ELO', 'AWAY_ELO', 'DIFFERENCE_ELO',
        'Bet365_Home_Win_Odds', 'Bet365_Draw_Odds', 'Bet365_Away_Win_Odds', 'BetAndWin_Home_Win_Odds', 'BetAndWin_Draw_Odds', 'BetAndWin_Away_Win_Odds',
        'Interwetten_Home_Win_Odds', 'Interwetten_Draw_Odds', 'Interwetten_Away_Win_Odds', 'WilliamHill_Home_Win_Odds', 'WilliamHill_Draw_Odds',
        'WilliamHill_Away_Win_Odds', 'VCBet_Home_Win_Odds', 'VCBet_Draw_Odds', 'VCBet_Away_Win_Odds', 'HOME_TRUESKILL_MU_NO_RESET',
        'AWAY_TRUESKILL_MU_NO_RESET','HOME_TRUESKILL_SIGMA_NO_RESET', 'AWAY_TRUESKILL_SIGMA_NO_RESET','DRAW_CHANCE_NO_RESET',
        'HOME_TRUESKILL_MU_SEASON', 'AWAY_TRUESKILL_MU_SEASON', 'HOME_TRUESKILL_SIGMA_SEASON', 'AWAY_TRUESKILL_SIGMA_SEASON',
        'DRAW_CHANCE_SEASON', 'HOME_ID', 'AWAY_ID', 'HOME_WINS_HOME', 'HOME_DRAWS_HOME', 'HOME_LOSSES_HOME', 'AWAY_WINS_HOME',
        'AWAY_DRAWS_HOME', 'AWAY_LOSSES_HOME', 'HOME_WINS_AWAY', 'HOME_DRAWS_AWAY', 'HOME_LOSSES_AWAY', 'AWAY_WINS_AWAY', 'AWAY_DRAWS_AWAY', 'AWAY_LOSSES_AWAY'
    ]

    df = df[columns_to_mantain]

    return df