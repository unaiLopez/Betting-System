import pandas as pd
import numpy as np

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

if __name__ == '__main__':
    df = pd.read_csv('../../inputs/raw_data/all_matches.csv', parse_dates=['Date'])
    df = df[df.HomeTeam.notna()]

    df.Full_Time_Result = df.Full_Time_Result.apply(full_time_result_to_class)

    df = calculate_elo_ratings(df)
    df = shift_elo(df)
    df = calculate_stat_differences(df)
    df = calculate_history(df)
    
    df.to_csv('../../inputs/ready_data/preprocessed_all_matches.csv', index=False)