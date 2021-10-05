def get_odds_header():
    """
    Description: Returns a dictionary to make more legible the headers of CO-UK data
    :return: Dictionary of legible headers
    """

    headers = {
        "Div": "Division",
        "FTHG": "Full_Time_Home_Team_Goals",
        "FTAG": "Full_Time_Away_Team_Goals",
        "FTR": "Full_Time_Result",
        "HTHG": "Half_Time_Home_Team_Goals",
        "HTAG": "Half_Time_Away_Team_Goals",
        "HTR": "Half_Time_Result",
        "HS": "Home_Team_Shots",
        "AS": "Away_Team_Shots",
        "HST": "Home_Team_Target_Shots",
        "AST": "Away_Team_Target_Shots",
        "HF": "Home_Team_Fouls_Commited",
        "AF": "Away_Team_Fouls_Commited",
        "HC": "Home_Team_Corners",
        "AC": "Away_Team_Corners",
        "HO": "Home_Team_Offsides",
        "AO": "Away_Team_Offsides",
        "HY": "Home_Team_Yellow_Cards",
        "AY": "Away_Team_Yellow_Cards",
        "HR": "Home_Team_Red_Cards",
        "AR": "Home_Team_Red_Cards",
        "B365H": "Bet365_Home_Win_Odds",
        "B365D": "Bet365_Draw_Odds",
        "B365A": "Bet365_Away_Win_Odds",
        "BWH": "BetAndWin_Home_Win_Odds",
        "BWD": "BetAndWin_Draw_Odds",
        "BWA": "BetAndWin_Away_Win_Odds",
        "IWH": "Interwetten_Home_Win_Odds",
        "IWD": "Interwetten_Draw_Odds",
        "IWA": "Interwetten_Away_Win_Odds",
        "PSH": "Pinacle_Home_Win_Odds",
        "PSD": "Pinacle_Draw_Odds",
        "PSA": "Pinacle_Away_Win_Odds",
        "WHH": "WilliamHill_Home_Win_Odds",
        "WHD": "WilliamHill_Draw_Odds",
        "WHA": "WilliamHill_Away_Win_Odds",
        "VCH": "VCBet_Home_Win_Odds",
        "VCD": "VCBet_Draw_Odds",
        "VCA": "VCBet_Away_Win_Odds",
        "MaxH": "Market_Maximum_Win_Odds",
        "MaxD": "Market_Maximum_Odds",
        "MaxA": "Market_Maximum_Win_Odds",
        "AvgH": "Market_Avg_Win_Odds",
        "AvgD": "Market_Avg_Odds",
        "AvgA": "Market_Avg_Win_Odds",
        "B365>2.5": "Bet365_Over_2.5_Goals",
        "B365<2.5": "Bet365_Under_2.5_Goals",
        "P>2.5": "Pinnacle_Over_2.5_Goals",
        "P<2.5": "Pinnacle_Under_2.5_Goals",
        "Max>2.5": "Market_Maximum_Over_2.5_Goals",
        "Max<2.5": "Market_Maximum_Under_2.5_Goals",
        "Avg>2.5": "Market_Avg_Over_2.5_Goals",
        "Avg<2.5": "Market_Avg_Under_2.5_Goals",
        "AHh": "Market_Size_Handicap",
        "B365AHH": "Bet365_Asian_Size_Handicap_Home_Odds",
        "B365AHA": "Bet365_Asian_Size_Handicap_Away_Odds",
        "PAHH": "Pinnacle_Asian_Size_Handicap_Home_Odds",
        "PAHA": "Pinnacle_Asian_Size_Handicap_Away_Odds",
        "MaxAHH": "Market_Maximum_Asian_Size_Handicap_Home_Odds",
        "MaxAHA": "Market_Maximum_Asian_Size_Handicap_Away_Odds",
        "AvgAHH": "Market_Avg_Asian_Size_Handicap_Home_Odds",
        "AvgAHA": "Market_Avg_Asian_Size_Handicap_Away_Odds",
        "B365CH": "B365CH",
        "B365CD": "B365CD",
        "B365CA": "B365CA",
        "BWCH": "BWCH",
        "BWCD": "BWCD",
        "BWCA": "BWCA",
        "IWCH": "IWCH",
        "IWCD": "IWCD",
        "IWCA": "IWCA",
        "PSCH": "PSCH",
        "PSCD": "PSCD",
        "PSCA": "PSCA",
        "WHCH": "WHCH",
        "WHCD": "WHCD",
        "WHCA": "WHCA",
        "VCCH": "VCCH",
        "VCCD": "VCCD",
        "VCCA": "VCCA",
        "MaxCH": "MaxCH",
        "MaxCD": "MaxCD",
        "MaxCA": "MaxCA",
        "AvgCH": "AvgCH",
        "AvgCD": "AvgCD",
        "AvgCA": "AvgCA",
        "B365C>2.5": "B365C>2.5",
        "B365C<2.5": "B365C<2.5",
        "PC>2.5": "PC>2.5",
        "PC<2.5": "PC<2.5",
        "MaxC>2.5": "MaxC>2.5",
        "MaxC<2.5": "MaxC<2.5",
        "AvgC>2.5": "AvgC>2.5",
        "AvgC<2.5": "AvgC<2.5",
        "AHCh": "AHCh",
        "B365CAHH": "B365CAHH",
        "B365CAHA": "B365CAHA",
        "PCAHH": "PCAHH",
        "PCAHA": "PCAHA",
        "MaxCAHH": "MaxCAHH",
        "MaxCAHA": "MaxCAHA",
        "AvgCAHH": "AvgCAHH",
        "AvgCAHA": "AvgCAHA"
    }
    return headers


def check_team_names(fifa_names: list, league_names: list):
    fifa = list(set(fifa_names).difference(set(league_names)))
    league = list(set(league_names).difference(set(fifa_names)))

    if len(league) > 0:
        print('League names that are not in FIFA df', league)
        print('League names that are not in League df', fifa)
        raise Exception("There are teams with a different name")


def team_name_changer(name: str) -> str:
    team_dict = {
        'FC Barcelona': 'Barcelona',
        'Real Madrid': 'Real Madrid',
        'Valencia CF': 'Valencia',
        'Sevilla FC': 'Sevilla',
        'Atlético Madrid': 'Ath Madrid',
        'CA Osasuna': 'Osasuna',
        'Villarreal CF': 'Villarreal',
        'Athletic Club de Bilbao': 'Ath Bilbao',
        'Real Betis': 'Betis',
        'RCD Espanyol': 'Espanol',
        'Real Zaragoza': 'Zaragoza',
        'Deportivo de La Coruña': 'La Coruna',
        'Levante UD': 'Levante',
        'Getafe CF': 'Getafe',
        'RC Celta': 'Celta',
        'RC Recreativo de Huelva': 'Recreativo',
        'Gimnàstic de Tarragona': 'Gimnastic',
        'RCD Mallorca': 'Mallorca',
        'Racing Santander': 'Santander',
        'Real Sociedad': 'Sociedad',
        'Real Valladolid CF': 'Valladolid',
        'Real Murcia Club de Fútbol': 'Murcia',
        'UD Almería': 'Almeria',
        'Málaga CF': 'Malaga',
        'CD Numancia': 'Numancia',
        'Real Sporting de Gijón': 'Sp Gijon',
        'CD Tenerife': 'Tenerife',
        'Xerez Club Deportivo': 'Xerez',
        'Hércules CF': 'Hercules',
        'Granada CF': 'Granada',
        'Rayo Vallecano': 'Vallecano',
        'Elche CF': 'Elche',
        'Córdoba CF': 'Cordoba',
        'SD Eibar': 'Eibar',
        'UD Las Palmas': 'Las Palmas',
        'CD Leganés': 'Leganes',
        'Deportivo Alavés': 'Alaves',
        'Girona FC': 'Girona',
        'SD Huesca': 'Huesca'



    }
    if name not in team_dict:
        raise Exception("{} team name is not in the dictionary, please add it in the utils file.".format(name))

    return team_dict[name]


