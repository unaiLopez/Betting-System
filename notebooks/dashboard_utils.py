from os import RTLD_DEEPBIND
import pandas as pd
import random

def get_elo_df():
    df = pd.read_csv('../inputs/ready_data/preprocessed_all_matches.csv')
    elo_list = []
    date_list = []
    team_list = []
    season_list = []
    for _, row in df.iterrows():
        home_team = row.HomeTeam
        away_team = row.AwayTeam

        home_elo = row.HOME_TRUESKILL_MU_SEASON
        away_elo = row.AWAY_TRUESKILL_MU_SEASON
        
        match_date = row.Date
        season = row.Season

        date_list.append(match_date)
        date_list.append(match_date)
        
        team_list.append(home_team)
        team_list.append(away_team)

        elo_list.append(home_elo)
        elo_list.append(away_elo)

        season_list.append(season)
        season_list.append(season)

    df = pd.DataFrame({"Team": team_list, "ELO": elo_list, "Date": date_list, "Season": season_list})
    df.sort_values(by=['Date'], inplace=True)
    return df    

def get_profit_df():
    df = pd.read_csv('../inputs/ready_data/preprocessed_all_matches.csv')
    print(df.shape)
    df_home = get_only_home_df(df.copy())
    df_away = get_only_away_df(df.copy())
    df_draw = get_only_draw_df(df.copy())
    df_random = get_random_df(df.copy())
    df_elo = get_elo_profit_df(df.copy())
    final_df = df_home.append(df_away).append(df_draw).append(df_random).append(df_elo)
    print(final_df.METHOD.unique())
    print(final_df.shape)
    final_df.sort_values(by=['Date'], inplace=True)
    return final_df

def get_only_home_df(df):
    df_home=df
    df_home['METHOD'] = 'Home'
    profit_list = []
    for _, row in df_home.iterrows():
        if row.Full_Time_Result==1:
            profit_list.append(row.BetAndWin_Home_Win_Odds)
        else:
            profit_list.append(-1)
    
    df_home['PROFIT'] = profit_list
    df_home['ACC_PROFIT'] = df_home.PROFIT.cumsum()
    return df_home


def get_only_away_df(df):
    df_away=df
    df_away['METHOD'] = 'Away'
    profit_list = []
    for _, row in df_away.iterrows():
        if row.Full_Time_Result==2:
            profit_list.append(row.BetAndWin_Away_Win_Odds)
        else:
            profit_list.append(-1)
    
    df_away['PROFIT'] = profit_list
    df_away['ACC_PROFIT'] = df_away.PROFIT.cumsum()
    return df_away

def get_only_draw_df(df):
    df_draw=df
    df_draw['METHOD'] = 'Draw'
    profit_list = []
    for _, row in df_draw.iterrows():
        if row.Full_Time_Result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        else:
            profit_list.append(-1)
    
    df_draw['PROFIT'] = profit_list
    df_draw['ACC_PROFIT'] = df_draw.PROFIT.cumsum()
    return df_draw

def get_random_df(df):
    df_random=df
    df_random['METHOD'] = 'Random'
    profit_list = []
    for _, row in df_random.iterrows():
        pred = random.randint(0,2)
        # BetAndWin_Home_Win_Odds,BetAndWin_Draw_Odds,BetAndWin_Away_Win_Odds
        result = row.Full_Time_Result
        if pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        elif pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        elif pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        else:
            profit_list.append(-1)
    
    df_random['PROFIT'] = profit_list
    df_random['ACC_PROFIT'] = df_random.PROFIT.cumsum()
    return df_random

def get_elo_profit_df(df):
    df_elo_profit = df
    df_elo_profit['METHOD'] = 'ELO'
    profit_list = []
    for _, row in df_elo_profit.iterrows():
        result = row.Full_Time_Result
        pred = 0
        if row.DRAW_CHANCE_SEASON > 0.95:
            pred = 0
        elif row.HOME_TRUESKILL_MU_SEASON > row.AWAY_TRUESKILL_MU_SEASON:
            pred = 1
        elif row.HOME_TRUESKILL_MU_SEASON < row.AWAY_TRUESKILL_MU_SEASON:
            pred = 2


        if pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        elif pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        elif pred == result and result==0:
            profit_list.append(row.BetAndWin_Draw_Odds)
        else:
            profit_list.append(-1)
    
    df_elo_profit['PROFIT'] = profit_list
    df_elo_profit['ACC_PROFIT'] = df_elo_profit.PROFIT.cumsum()
    return df_elo_profit


    