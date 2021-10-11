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
            elo_formula_dict[row['HomeTeam']]['ELO'] = 1000

        if row['AwayTeam'] not in elo_formula_dict:
            elo_formula_dict[row['AwayTeam']] = {}
            elo_formula_dict[row['AwayTeam']]['ELO'] = 1000

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

def calculate_stat_differences(df):
    df['Difference_Overall_Score'] = df['Home Overall Score'] - df['Away Overall Score']
    df['Difference_Attack_Score'] = df['Home Attack Score'] - df['Away Attack Score']
    df['Difference_Middle_Score'] = df['Home Middle Score'] - df['Away Middle Score']
    df['Difference_Defensive_Score'] = df['Home Defensive Score'] - df['Away Defensive Score']
    df['Difference_Budget'] = df['Home Budget'] - df['Away Budget']
    df['DIFFERENCE_ELO'] = df['HOME_ELO'] - df['AWAY_ELO']

    return df

if __name__ == '__main__':
    df = pd.read_csv('../inputs/raw_data/all_matches.csv', parse_dates=['Date'])
    
    df.Full_Time_Result = df.Full_Time_Result.apply(full_time_result_to_class)

    df = calculate_elo_ratings(df)
    df = calculate_stat_differences(df)

    df.to_csv('../inputs/ready_data/preprocessed_all_matches.csv', index=False)