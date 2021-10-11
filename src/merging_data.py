import pandas as pd
from utils import get_odds_header, team_name_changer, check_team_names
from tqdm import tqdm


def format_date(match_date: str) -> str:
    splitted_date = match_date.split('/')
    match_date = splitted_date[0] + '/' + splitted_date[1] + '/' + '20' + splitted_date[2]
    return match_date


def get_season_df(year: int) -> pd.DataFrame:
    league_file = '../data/Spanish-League-Data/Season_' + str(year) + '_' + str(year+1) + '.csv'
    fifa_file = '../data/Web-Scraping-FIFA/Teams-Season/Teams Season ' + str(year) + '-' + str(year+1)
    df_league = pd.read_csv(league_file)
    df_fifa = pd.read_csv(fifa_file)

    df_league.rename(columns=get_odds_header(), inplace=True)

    # Transform scraping date format and team names
    if year < 2018:
        df_league['Date'] = df_league['Date'].apply(format_date)
    df_league['Date'] = pd.to_datetime(df_league['Date'])

    df_fifa['Team'] = df_fifa['Team'].apply(team_name_changer)
    df_fifa['Date'] = pd.to_datetime(df_fifa['Date'])

    teams_league = df_league['HomeTeam'].unique()
    teams_fifa = df_fifa['Team'].unique()
    if len(teams_league) != len(teams_fifa):
        raise Exception("The number of teams is different between the two datasets.")

    check_team_names(teams_league, teams_fifa)  # Checks for different names between the two dataframes

    # Creating df columns
    stat_columns = [col for col in df_fifa.columns if col != 'Date' and col != 'Team']

    home_dict = {col: "Home " + col for col in stat_columns}
    away_dict = {col: "Away " + col for col in stat_columns}

    home_df = pd.DataFrame(columns = stat_columns)
    away_df = pd.DataFrame(columns = stat_columns)

    fifa_dates = df_fifa['Date'].unique()
    last_update = fifa_dates.min()
    first_iteration = True
    for _, row in df_league.iterrows():
        current_date = row['Date']
        df_aux = df_fifa.query('@last_update == Date')
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        update_home_stats = df_aux.query('Team == @home_team')
        update_away_stats = df_aux.query('Team == @away_team')

        home_df = home_df.append(update_home_stats)
        away_df = away_df.append(update_away_stats)

        if current_date >= last_update and len(fifa_dates) > 0:
            if first_iteration:
                first_iteration = False
            else:
                last_update = fifa_dates.min()
                fifa_dates = fifa_dates[fifa_dates > min(fifa_dates)]

    home_df.drop(['Team', 'Date'], axis=1, inplace=True)
    away_df.drop(['Team', 'Date'], axis=1, inplace=True)

    home_df.rename(columns=home_dict, inplace=True)
    away_df.rename(columns=away_dict, inplace=True)

    home_df = home_df.reset_index(drop=True)
    away_df = away_df.reset_index(drop=True)

    season_df = pd.concat([home_df, away_df], axis=1)
    df_league = pd.concat([df_league, season_df], axis=1)
    df_comp = df_league[['Date', 'HomeTeam', 'AwayTeam', 'Home Overall Score', 'Away Overall Score']]
    return df_league


def main():
    matches_df = pd.DataFrame()
    for season_year in tqdm(range(2006, 2020)):
        season_matches = get_season_df(year=season_year)
        if matches_df.shape[0] == 0:
            matches_df = season_matches
            matches_df.drop(['SBD', 'SBA', 'SBH',
                             'GBA', 'GBH', 'GBD',
                             'GBD', 'GBH',
                             'SJH', 'SJD', 'SJA',
                             'LBD', 'LBH', 'LBA',
                             'B365CH', 'Market_Avg_Under_2.5_Goals', 'B365CA', 'Market_Maximum_Odds', 'B365CAHH', 'Market_Size_Handicap', 'Market_Avg_Asian_Size_Handicap_Away_Odds', 'MaxCH', 'Market_Avg_Win_Odds', 'Market_Maximum_Under_2.5_Goals', 'Market_Avg_Over_2.5_Goals', 'Market_Avg_Asian_Size_Handicap_Home_Odds', 'MaxC<2.5', 'Bet365_Asian_Size_Handicap_Home_Odds', 'Pinnacle_Under_2.5_Goals', 'IWCA', 'Market_Avg_Odds', 'Pinnacle_Over_2.5_Goals', 'MaxCD', 'Market_Maximum_Asian_Size_Handicap_Home_Odds', 'BWCD', 'PC>2.5', 'AvgCAHH', 'MaxCA', 'B365C>2.5', 'Market_Maximum_Asian_Size_Handicap_Away_Odds', 'B365CD', 'IWCD', 'MaxCAHA', 'Time', 'PCAHA', 'Market_Maximum_Over_2.5_Goals', 'MaxC>2.5', 'AvgCAHA', 'BWCA', 'Bet365_Under_2.5_Goals', 'VCCH', 'AvgCA', 'AHCh', 'B365C<2.5', 'Market_Maximum_Win_Odds', 'VCCD', 'PC<2.5', 'IWCH', 'AvgC>2.5', 'Pinnacle_Asian_Size_Handicap_Away_Odds', 'WHCD', 'WHCA', 'AvgC<2.5', 'Pinnacle_Asian_Size_Handicap_Home_Odds', 'VCCA', 'Bet365_Asian_Size_Handicap_Away_Odds', 'BWCH', 'Bet365_Over_2.5_Goals', 'AvgCH', 'WHCH', 'PCAHH', 'AvgCD', 'B365CAHA', 'MaxCAHH',
                             'BbAv<2.5', 'BbAHh', 'BbAv>2.5', 'BbAvA', 'BbMx<2.5', 'Bb1X2', 'BbMxA', 'BbAvAHA', 'BbMxH', 'BbAvAHH', 'BbMxAHA', 'BbMxD', 'BbMx>2.5', 'BbAvD', 'BbAvH', 'BbMxAHH', 'BbAH', 'BbOU'], axis=1, inplace=True, errors='ignore')
        else:
            season_matches.drop(['BSD', 'BSA', 'BSH',
                                 'PSCH', 'PSCD', 'Pinacle_Away_Win_Odds', 'Pinacle_Home_Win_Odds', 'Pinacle_Draw_Odds', 'PSCA',
                                 'SBD', 'SBA', 'SBH',
                                 'GBH', 'GBA', 'GBD',
                                 'SJD', 'SJA', 'SJH',
                                 'LBD', 'LBH', 'LBA',

                                 'B365CH', 'Market_Avg_Under_2.5_Goals', 'B365CA', 'Market_Maximum_Odds', 'B365CAHH',
                                 'Market_Size_Handicap', 'Market_Avg_Asian_Size_Handicap_Away_Odds', 'MaxCH',
                                 'Market_Avg_Win_Odds', 'Market_Maximum_Under_2.5_Goals', 'Market_Avg_Over_2.5_Goals',
                                 'Market_Avg_Asian_Size_Handicap_Home_Odds', 'MaxC<2.5',
                                 'Bet365_Asian_Size_Handicap_Home_Odds', 'Pinnacle_Under_2.5_Goals', 'IWCA',
                                 'Market_Avg_Odds', 'Pinnacle_Over_2.5_Goals', 'MaxCD',
                                 'Market_Maximum_Asian_Size_Handicap_Home_Odds', 'BWCD', 'PC>2.5', 'AvgCAHH', 'MaxCA',
                                 'B365C>2.5', 'Market_Maximum_Asian_Size_Handicap_Away_Odds', 'B365CD', 'IWCD',
                                 'MaxCAHA', 'Time', 'PCAHA', 'Market_Maximum_Over_2.5_Goals', 'MaxC>2.5', 'AvgCAHA',
                                 'BWCA', 'Bet365_Under_2.5_Goals', 'VCCH', 'AvgCA', 'AHCh', 'B365C<2.5',
                                 'Market_Maximum_Win_Odds', 'VCCD', 'PC<2.5', 'IWCH', 'AvgC>2.5',
                                 'Pinnacle_Asian_Size_Handicap_Away_Odds', 'WHCD', 'WHCA', 'AvgC<2.5',
                                 'Pinnacle_Asian_Size_Handicap_Home_Odds', 'VCCA',
                                 'Bet365_Asian_Size_Handicap_Away_Odds', 'BWCH', 'Bet365_Over_2.5_Goals', 'AvgCH',
                                 'WHCH', 'PCAHH', 'AvgCD', 'B365CAHA', 'MaxCAHH',

                                 'BbAv<2.5', 'BbAHh', 'BbAv>2.5', 'BbAvA', 'BbMx<2.5', 'Bb1X2', 'BbMxA', 'BbAvAHA',
                                 'BbMxH', 'BbAvAHH', 'BbMxAHA', 'BbMxD', 'BbMx>2.5', 'BbAvD', 'BbAvH', 'BbMxAHH',
                                 'BbAH', 'BbOU'
                                 ], axis=1, inplace=True, errors='ignore')
            print('')
            #print(season_year)
            #print('FIFA: ', list(set(season_matches).difference(set(matches_df))))
            #print('Matches:', list(set(matches_df).difference(set(season_matches))))
            matches_df = matches_df.append(season_matches, ignore_index=True)

    matches_df.to_csv('../data/raw_data/all_matches.csv', index=False, encoding='UTF-8')


if __name__ == '__main__':
    main()
