import pandas as pd
import numpy as np
from trueskill import Rating, quality_1vs1, rate_1vs1

def full_time_result_to_class(result):
    if result == 'H':
        return 1
    elif result == 'D':
        return 0
    elif result == 'A':
        return 2

def calculate_elo_ratings(df):
    df['Goal_Difference'] = np.abs((df['Full_Time_Away_Team_Goals'] - df['Full_Time_Home_Team_Goals']))
    df['HOME_ELO'] = 0
    df['AWAY_ELO'] = 0

    elo_formula_dict = {}

    for i, row in df.iterrows():
        if row['Goal_Difference'] == 2:
            k = 30 * 1.5
        elif row['Goal_Difference'] == 3:
            k = 30 * 1.75
        elif row['Goal_Difference'] >= 4:
            k = 30 * 1.75 + ((row['Goal_Difference'] - 3)/8)
        else:
            k = 30

        if row['HomeTeam'] not in elo_formula_dict:
            elo_formula_dict[row['HomeTeam']] = {}
            elo_formula_dict[row['HomeTeam']]['ELO'] = 1500

        if row['AwayTeam'] not in elo_formula_dict:
            elo_formula_dict[row['AwayTeam']] = {}
            elo_formula_dict[row['AwayTeam']]['ELO'] = 1500

        rating_difference = elo_formula_dict[row['HomeTeam']]['ELO'] - elo_formula_dict[row['AwayTeam']]['ELO'] + 100
        elo_formula_dict[row['HomeTeam']]['Win_Expectancy'] = 1 / (10 ** (-rating_difference / 400) + 1)

        if row['Full_Time_Result'] == 1:
            elo_formula_dict[row['HomeTeam']]['Result_Game'] = 1
        elif row['Full_Time_Result'] == 0:
            elo_formula_dict[row['HomeTeam']]['Result_Game'] = 0.5
        else:
            elo_formula_dict[row['HomeTeam']]['Result_Game'] = 0

        elo_formula_dict[row['HomeTeam']]['ELO'] = elo_formula_dict[row['HomeTeam']]['ELO'] + k * (elo_formula_dict[row['HomeTeam']]['Result_Game'] - elo_formula_dict[row['HomeTeam']]['Win_Expectancy'])

        rating_difference = elo_formula_dict[row['AwayTeam']]['ELO'] - elo_formula_dict[row['HomeTeam']]['ELO'] + 100
        elo_formula_dict[row['AwayTeam']]['Win_Expectancy'] = 1 - elo_formula_dict[row['HomeTeam']]['Win_Expectancy']

        if row['Full_Time_Result'] == 1:
            elo_formula_dict[row['AwayTeam']]['Result_Game'] = 0
        elif row['Full_Time_Result'] == 0:
            elo_formula_dict[row['AwayTeam']]['Result_Game'] = 0.5
        else:
            elo_formula_dict[row['AwayTeam']]['Result_Game'] = 1

        elo_formula_dict[row['AwayTeam']]['ELO'] = elo_formula_dict[row['AwayTeam']]['ELO'] + k * (elo_formula_dict[row['AwayTeam']]['Result_Game'] - elo_formula_dict[row['AwayTeam']]['Win_Expectancy'])

        df.loc[i, 'AWAY_ELO'] = elo_formula_dict[row['AwayTeam']]['ELO']
        df.loc[i, 'HOME_ELO'] = elo_formula_dict[row['HomeTeam']]['ELO']
    
    df.drop(['Goal_Difference', 'Full_Time_Away_Team_Goals', 'Full_Time_Home_Team_Goals'], axis=1, inplace=True)
    
    return df


def shift_elo(df):
    elo_dict = dict()
    for index, row in df.iterrows():
        home_team = row.HomeTeam
        away_team = row.AwayTeam
        home_elo = row.HOME_ELO
        away_elo = row.AWAY_ELO

        if home_team not in elo_dict:
            elo_dict[home_team] = home_elo
        else:
            df.loc[index, 'HOME_ELO'] = elo_dict[home_team]
            elo_dict[home_team] = home_elo

        if away_team not in elo_dict:

            elo_dict[away_team] = away_elo
        else:
            df.loc[index, 'AWAY_ELO'] = elo_dict[away_team]
            elo_dict[away_team] = away_elo
    
    return df


def calculate_stat_differences(df):
    df['Difference_Overall_Score'] = df['Home Overall Score'] - df['Away Overall Score']
    df['Difference_Attack_Score'] = df['Home Attack Score'] - df['Away Attack Score']
    df['Difference_Middle_Score'] = df['Home Middle Score'] - df['Away Middle Score']
    df['Difference_Defensive_Score'] = df['Home Defensive Score'] - df['Away Defensive Score']
    df['Difference_Budget'] = df['Home Budget'] - df['Away Budget']
    df['DIFFERENCE_ELO'] = df['HOME_ELO'] - df['AWAY_ELO']

    return df

def calculate_history(df: pd.DataFrame) -> pd.DataFrame:
    team_history = dict()
    home_win = []
    away_win = []
    draws = []
    matches = []
    for _, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        teams = sorted([home_team, away_team], key=str.lower)
        teams = teams[0]+teams[1]
        result = row["Full_Time_Result"]
        if teams not in team_history:
            team_history[teams] = {home_team: 0, away_team: 0, 'Draw': 0, 'Matches': 0}
        
        if result == 0:
            value = team_history[teams]['Draw'] 
            draws.append(value)
            team_history[teams]['Draw'] = value + 1

            home_win.append(team_history[teams][home_team])
            away_win.append(team_history[teams][away_team])
            
        elif result == 1:
            value = team_history[teams][home_team] 
            home_win.append(value)
            team_history[teams][home_team] = value + 1

            draws.append(team_history[teams]['Draw'])
            away_win.append(team_history[teams][away_team])

        elif result == 2:
            value = team_history[teams][away_team]
            away_win.append(value)
            team_history[teams][away_team] = value + 1
            
            draws.append(team_history[teams]['Draw'])
            home_win.append(team_history[teams][home_team])

        matches.append(team_history[teams]['Matches'] + 1)
        team_history[teams]['Matches'] = team_history[teams]['Matches'] + 1

    print(len(home_win), len(away_win), len(draws), len(matches))
    df['HISTORICAL_HOME_WINS'] = home_win
    df['HISTORICAL_AWAY_WINS'] = away_win
    df['HISTORICAL_DRAWS'] = draws
    df['HISTORICAL_MATCHES'] = matches

    return df


def calculate_trueskill(df: pd.DataFrame) -> pd.DataFrame:

    trueskill_dict = {}
    trueskill_season_dict = {}

    home_elos = []
    away_elos = []

    home_mu = []
    home_sigma = []
    away_mu = []
    away_sigma = []

    draw_chance = []

    home_elos_reset = []
    away_elos_reset = []

    home_mu_reset = []
    home_sigma_reset = []
    away_mu_reset = []
    away_sigma_reset = []

    draw_chance_reset = []

    for _, row in df.iterrows():
        home_team = row.HomeTeam
        away_team = row.AwayTeam
        result = row.Full_Time_Result
        season = row.Season

        if season not in trueskill_season_dict:
            trueskill_season_dict[season] = {}

        if home_team not in trueskill_dict:
            trueskill_dict[home_team] = Rating()
        
        if away_team not in trueskill_dict:
            trueskill_dict[away_team] = Rating()
        
        if home_team not in trueskill_season_dict[season]:
            trueskill_season_dict[season][home_team] = Rating()
        
        if away_team not in trueskill_season_dict[season]:
            trueskill_season_dict[season][away_team] = Rating()
        

        home_elos.append(trueskill_dict[home_team])
        away_elos.append(trueskill_dict[away_team])

        home_mu.append(trueskill_dict[home_team].mu)
        home_sigma.append(trueskill_dict[home_team].sigma)

        away_mu.append(trueskill_dict[away_team].mu)
        away_sigma.append(trueskill_dict[away_team].sigma)
        
        draw_chance.append(quality_1vs1(trueskill_dict[home_team], trueskill_dict[away_team]))


        home_elos_reset.append(trueskill_season_dict[season][home_team])
        away_elos_reset.append(trueskill_season_dict[season][away_team])

        home_mu_reset.append(trueskill_season_dict[season][home_team].mu)
        home_sigma_reset.append(trueskill_season_dict[season][home_team].sigma)

        away_mu_reset.append(trueskill_season_dict[season][away_team].mu)
        away_sigma_reset.append(trueskill_season_dict[season][away_team].sigma)
        
        draw_chance_reset.append(quality_1vs1(trueskill_season_dict[season][home_team], trueskill_season_dict[season][away_team]))



        if result == 0:
            trueskill_dict[home_team], trueskill_dict[away_team] = rate_1vs1(trueskill_dict[home_team], trueskill_dict[away_team], drawn=True)
            trueskill_season_dict[season][home_team], trueskill_season_dict[season][away_team] = rate_1vs1(trueskill_season_dict[season][home_team], trueskill_season_dict[season][away_team], drawn=True)
        elif result == 1:
            trueskill_dict[home_team], trueskill_dict[away_team] = rate_1vs1(trueskill_dict[home_team], trueskill_dict[away_team])
            trueskill_season_dict[season][home_team], trueskill_season_dict[season][away_team] = rate_1vs1(trueskill_season_dict[season][home_team], trueskill_season_dict[season][away_team])
        elif result == 2:
            trueskill_dict[away_team], trueskill_dict[home_team] = rate_1vs1(trueskill_dict[away_team], trueskill_dict[home_team])
            trueskill_season_dict[season][away_team], trueskill_season_dict[season][home_team] = rate_1vs1(trueskill_season_dict[season][away_team], trueskill_season_dict[season][home_team])
        else:
            raise Exception('This match has no result, please check the data.')

    # df['HOME_TRUESKILL_NO_RESET'] = home_elos
    # df['AWAY_TRUESKILL_NO_RESET'] = away_elos
    df['HOME_TRUESKILL_MU_NO_RESET'] = home_mu
    df['AWAY_TRUESKILL_MU_NO_RESET'] = away_mu
    df['HOME_TRUESKILL_SIGMA_NO_RESET'] = home_sigma
    df['AWAY_TRUESKILL_SIGMA_NO_RESET'] = away_sigma
    df['DRAW_CHANCE_NO_RESET'] = draw_chance

    # df['HOME_TRUESKILL_SEASON'] = home_elos_reset
    # df['AWAY_TRUESKILL_SEASON'] = away_elos_reset
    df['HOME_TRUESKILL_MU_SEASON'] = home_mu_reset
    df['AWAY_TRUESKILL_MU_SEASON'] = away_mu_reset
    df['HOME_TRUESKILL_SIGMA_SEASON'] = home_sigma_reset
    df['AWAY_TRUESKILL_SIGMA_SEASON'] = away_sigma_reset
    df['DRAW_CHANCE_SEASON'] = draw_chance_reset
    return df


def team_names_to_numeric(df):
    team_list = df.HomeTeam.unique().tolist()
    team_dict = {x: team_list.index(x) for x in team_list}
    df['HOME_ID'] = df.HomeTeam.map(team_dict)
    df['AWAY_ID'] = df.AwayTeam.map(team_dict)
    return df


def result_count(df):
    seasons = df.Season.unique()
    home_wins_playing_home = []
    home_draws_playing_home = []
    home_losses_playing_home = []

    away_wins_playing_home = []
    away_draws_playing_home = []
    away_losses_playing_home = []

    home_wins_playing_away = []
    home_draws_playing_away = []
    home_losses_playing_away = []

    away_wins_playing_away = []
    away_draws_playing_away = []
    away_losses_playing_away = []
    teams = {}
    for season in seasons:
        df_aux = df.query('Season==@season')
        teams = {}
        for _, row in df_aux.iterrows():
            home_team = row.HomeTeam
            away_team = row.AwayTeam
            if home_team not in teams:
                teams[home_team] = {"Home_Wins": [], "Home_Draws": [], "Home_Losses": [],
                                    "Away_Wins": [], "Away_Draws": [], "Away_Losses": []}
            if away_team not in teams:
                teams[away_team] = {"Home_Wins": [], "Home_Draws": [], "Home_Losses": [],
                                    "Away_Wins": [], "Away_Draws": [], "Away_Losses": []}

            result = row.Full_Time_Result
            
            
            home_wins_playing_home.append(sum(teams[home_team]["Home_Wins"]))
            home_draws_playing_home.append(sum(teams[home_team]["Home_Draws"]))
            home_losses_playing_home.append(sum(teams[home_team]["Home_Losses"]))
            away_wins_playing_home.append(sum(teams[away_team]["Home_Wins"]))
            away_draws_playing_home.append(sum(teams[away_team]["Home_Draws"]))
            away_losses_playing_home.append(sum(teams[away_team]["Home_Losses"]))

            home_wins_playing_away.append(sum(teams[home_team]["Away_Wins"]))
            home_draws_playing_away.append(sum(teams[home_team]["Away_Draws"]))
            home_losses_playing_away.append(sum(teams[home_team]["Away_Losses"]))
            away_wins_playing_away.append(sum(teams[away_team]["Away_Wins"]))
            away_draws_playing_away.append(sum(teams[away_team]["Away_Draws"]))
            away_losses_playing_away.append(sum(teams[away_team]["Away_Losses"]))

            if result == 0:
                # empate
                teams[home_team]["Home_Wins"].append(0)
                teams[home_team]["Home_Draws"].append(1)
                teams[home_team]["Home_Losses"].append(0)
                teams[home_team]["Away_Wins"].append(0)
                teams[home_team]["Away_Draws"].append(0)
                teams[home_team]["Away_Losses"].append(0)

                teams[away_team]["Home_Wins"].append(0)
                teams[away_team]["Home_Draws"].append(0)
                teams[away_team]["Home_Losses"].append(0)
                teams[away_team]["Away_Wins"].append(0)
                teams[away_team]["Away_Draws"].append(1)
                teams[away_team]["Away_Losses"].append(0)

            elif result == 1:
                teams[home_team]["Home_Wins"].append(1)
                teams[home_team]["Home_Draws"].append(0)
                teams[home_team]["Home_Losses"].append(0)
                teams[home_team]["Away_Wins"].append(0)
                teams[home_team]["Away_Draws"].append(0)
                teams[home_team]["Away_Losses"].append(0)

                teams[away_team]["Home_Wins"].append(0)
                teams[away_team]["Home_Draws"].append(0)
                teams[away_team]["Home_Losses"].append(0)
                teams[away_team]["Away_Wins"].append(0)
                teams[away_team]["Away_Draws"].append(0)
                teams[away_team]["Away_Losses"].append(1)

            else:
                teams[home_team]["Home_Wins"].append(0)
                teams[home_team]["Home_Draws"].append(0)
                teams[home_team]["Home_Losses"].append(1)
                teams[home_team]["Away_Wins"].append(0)
                teams[home_team]["Away_Draws"].append(0)
                teams[home_team]["Away_Losses"].append(0)

                teams[away_team]["Home_Wins"].append(0)
                teams[away_team]["Home_Draws"].append(0)
                teams[away_team]["Home_Losses"].append(0)
                teams[away_team]["Away_Wins"].append(1)
                teams[away_team]["Away_Draws"].append(0)
                teams[away_team]["Away_Losses"].append(0)

    df['HOME_WINS_HOME'] = home_wins_playing_home
    df['HOME_DRAWS_HOME'] = home_draws_playing_home
    df['HOME_LOSSES_HOME'] = home_losses_playing_home
    df['AWAY_WINS_HOME'] = away_wins_playing_home
    df['AWAY_DRAWS_HOME'] = away_draws_playing_home
    df['AWAY_LOSSES_HOME'] = away_losses_playing_home
    df['HOME_WINS_AWAY'] = home_wins_playing_away
    df['HOME_DRAWS_AWAY'] = home_draws_playing_away
    df['HOME_LOSSES_AWAY'] = home_losses_playing_away
    df['AWAY_WINS_AWAY'] = away_wins_playing_away
    df['AWAY_DRAWS_AWAY'] = away_draws_playing_away
    df['AWAY_LOSSES_AWAY'] = away_losses_playing_away
    return df

def clean_data(df: pd.DataFrame)->pd.DataFrame:
    df.drop(['Half_Time_Home_Team_Goals','Half_Time_Away_Team_Goals', 'Half_Time_Result', 'Home_Team_Red_Cards.1'], axis=1, inplace=True)
    return df


if __name__ == '__main__':
    df = pd.read_csv('../../inputs/raw_data/all_matches.csv', parse_dates=['Date'])
    df = df[df.HomeTeam.notna()]

    df.Full_Time_Result = df.Full_Time_Result.apply(full_time_result_to_class)

    df = calculate_elo_ratings(df)
    df = shift_elo(df)
    df = calculate_stat_differences(df)
    df = calculate_history(df)
    df = calculate_trueskill(df)
    df = team_names_to_numeric(df)
    df = result_count(df)
    df = clean_data(df)
    df.to_csv('../../inputs/ready_data/preprocessed_all_matches.csv', index=False)