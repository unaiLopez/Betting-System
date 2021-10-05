import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from os import listdir

FOLDER_PATH = "Spanish_League_Data"
COLUMNS_TO_MANTAIN = [
                      "Div", "Date", "HomeTeam", "AwayTeam", 
                      "FTHG", "FTAG", "FTR", "HTHG", "HTAG",
                      "HTR", "HS", "AS", "HST", "AST", "HF",
                      "AF", "HC", "AC", "HY", "AY", "HR", "AR",
                      "B365H", "B365D", "B365A", "BWH", "BWD",
                      "BWA", "IWH", "IWD", "IWA",
                      "WHH", "WHD", "WHA", "VCH", "VCD", "VCA"
                     ]
READABLE_HEADER = [
                   "League Division", "Match Date", "Home Team", "Away Team", 
                   "Home Team Goals", "Away Team Goals", "Match Result", 
                   "Half Time Home Team Goals", "Half Time Away Team Goals", 
                   "Half Time Result", "Home Team Shots", "Away Team Shots", 
                   "Home Team Shots Target", "Away Team Shots Target",
                   "Home Team Faults Commited", "Away Team Faults Commited",
                   "Home Team Corners", "Away Team Corners", "Home Team Yellow Cards", 
                   "Away Team Yellow Cards", "Home Team Red Cards", "Away Team Red Cards",
                   "Bet365 Home Win Odds", "Bet365 Draw Odds", "Bet365 Away Win Odds", 
                   "Bet&Win Home Win Odds", "Bet&Win Draw Odds", "Bet&Win Away Win Odds",
                   "Interwetten Home Win Odds", "Interwetten Draw Odds", "Interwetten Away Win Odds",  
                   "William Hill Home Win Odds", "William Hill Draw Odds", "William Hill Away Win Odds",
                   "VC Bet Home Win Odds", "VC Bet Draw Odds", "VC Bet Away Win Odds"
                  ]


def match_result_to_numeric(x):
    if x == 'H':
        return 1
    if x == 'D':
        return 2
    if x == 'A':
        return 3

def fill_na_values(data):
    features_to_fill = [
                    "Interwetten Home Win Odds", "Interwetten Away Win Odds", "Interwetten Draw Odds", 
                    "William Hill Home Win Odds", "William Hill Away Win Odds", "William Hill Draw Odds", 
                    "VC Bet Home Win Odds", "VC Bet Away Win Odds", "VC Bet Draw Odds"
                   ]
    for feature in features_to_fill:
        temp_mean = data[feature].mean()
        data[feature] = data[feature].fillna(temp_mean)
    
    return data

def initialize_accumulated_statistics(data):
    features_to_initialize = [
                          "Home Team Accumulated Scored Goals", "Away Team Accumulated Scored Goals",
                    	  "Home Team Accumulated Received Goals", "Away Team Accumulated Received Goals",
                    	  "Home Team Accumulated Yellow Cards", "Away Team Accumulated Yellow Cards",
		          "Home Team Accumulated Red Cards", "Away Team Accumulated Red Cards",
		          "Home Team Accumulated Thrown Shots", "Away Team Accumulated Thrown Shots",
		          "Home Team Accumulated Received Shots", "Away Team Accumulated Received Shots",
		          "Home Team Accumulated Thrown Shots Target", "Away Team Accumulated Thrown Shots Target",
		          "Home Team Accumulated Received Shots Target", "Away Team Accumulated Received Shots Target",
		          "Home Team Accumulated Thrown Corners", "Away Team Accumulated Thrown Corners",
		          "Home Team Accumulated Received Corners", "Away Team Accumulated Received Corners",
		          "Home Team Accumulated Commited Faults", "Away Team Accumulated Commited Faults", 
		          "Home Team Accumulated Received Faults", "Away Team Accumulated Received Faults"
                   	 ]
    for feature in features_to_initialize:
        data[feature] = 0
        
    return data

def compute_accumulated_received_faults(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][12]) == 0):
                    statistics_per_team[i][12].append(0)
                else:
                    statistics_per_team[i][12].append(data.loc[j, "Away Team Faults Commited"]+statistics_per_team[i][12][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][12]) == 0):
                    statistics_per_team[i][12].append(0)
                else:
                    statistics_per_team[i][12].append(data.loc[j, "Home Team Faults Commited"]+statistics_per_team[i][12][-1])
    return data, statistics_per_team

def compute_accumulated_commited_faults(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][11]) == 0):
                    statistics_per_team[i][11].append(0)
                else:
                    statistics_per_team[i][11].append(data.loc[j, "Home Team Faults Commited"]+statistics_per_team[i][11][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][11]) == 0):
                    statistics_per_team[i][11].append(0)
                else:
                    statistics_per_team[i][11].append(data.loc[j, "Away Team Faults Commited"]+statistics_per_team[i][11][-1])
    return data, statistics_per_team

def compute_accumulated_received_cornerss(tatistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][10]) == 0):
                    statistics_per_team[i][10].append(0)
                else:
                    statistics_per_team[i][10].append(data.loc[j, "Away Team Corners"]+statistics_per_team[i][10][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][10]) == 0):
                    statistics_per_team[i][10].append(0)
                else:
                    statistics_per_team[i][10].append(data.loc[j, "Home Team Corners"]+statistics_per_team[i][10][-1])
    return data, statistics_per_team

def compute_accumulated_thrown_corners(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][9]) == 0):
                    statistics_per_team[i][9].append(0)
                else:
                    statistics_per_team[i][9].append(data.loc[j, "Home Team Corners"]+statistics_per_team[i][9][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][9]) == 0):
                    statistics_per_team[i][9].append(0)
                else:
                    statistics_per_team[i][9].append(data.loc[j, "Away Team Corners"]+statistics_per_team[i][9][-1])
    return data, statistics_per_team

def compute_accumulated_received_shots_target(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][8]) == 0):
                    statistics_per_team[i][8].append(0)
                else:
                    statistics_per_team[i][8].append(data.loc[j, "Away Team Shots Target"]+statistics_per_team[i][8][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][8]) == 0):
                    statistics_per_team[i][8].append(0)
                else:
                    statistics_per_team[i][8].append(data.loc[j, "Home Team Shots Target"]+statistics_per_team[i][8][-1])
    return data, statistics_per_team

def compute_accumulated_thrown_shots_target(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][7]) == 0):
                    statistics_per_team[i][7].append(0)
                else:
                    statistics_per_team[i][7].append(data.loc[j, "Home Team Shots Target"]+statistics_per_team[i][7][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][7]) == 0):
                    statistics_per_team[i][7].append(0)
                else:
                    statistics_per_team[i][7].append(data.loc[j, "Away Team Shots Target"]+statistics_per_team[i][7][-1])
    return data, statistics_per_team

def compute_accumulated_received_shots(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][6]) == 0):
                    statistics_per_team[i][6].append(0)
                else:
                    statistics_per_team[i][6].append(data.loc[j, "Away Team Shots"]+statistics_per_team[i][6][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][6]) == 0):
                    statistics_per_team[i][6].append(0)
                else:
                    statistics_per_team[i][6].append(data.loc[j, "Home Team Shots"]+statistics_per_team[i][6][-1])
    return data, statistics_per_team

def compute_accumulated_thrown_shots(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][5]) == 0):
                    statistics_per_team[i][5].append(0)
                else:
                    statistics_per_team[i][5].append(data.loc[j, "Home Team Shots"]+statistics_per_team[i][5][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][5]) == 0):
                    statistics_per_team[i][5].append(0)
                else:
                    statistics_per_team[i][5].append(data.loc[j, "Away Team Shots"]+statistics_per_team[i][5][-1])
    return data, statistics_per_team

def compute_accumulated_received_red_cards(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][4]) == 0):
                    statistics_per_team[i][4].append(0)
                else:
                    statistics_per_team[i][4].append(data.loc[j, "Home Team Red Cards"]+statistics_per_team[i][4][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][4]) == 0):
                    statistics_per_team[i][4].append(0)
                else:
                    statistics_per_team[i][4].append(data.loc[j, "Away Team Red Cards"]+statistics_per_team[i][4][-1])
    return data, statistics_per_team

def compute_accumulated_received_yellow_cards(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][3]) == 0):
                    statistics_per_team[i][3].append(0)
                else:
                    statistics_per_team[i][3].append(data.loc[j, "Home Team Yellow Cards"]+statistics_per_team[i][3][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][3]) == 0):
                    statistics_per_team[i][3].append(0)
                else:
                    statistics_per_team[i][3].append(data.loc[j, "Away Team Yellow Cards"]+statistics_per_team[i][3][-1])
    return data, statistics_per_team

def compute_accumulated_received_goals(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][2]) == 0):
                    statistics_per_team[i][2].append(0)
                else:
                    statistics_per_team[i][2].append(data.loc[j, "Away Team Goals"]+statistics_per_team[i][2][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][2]) == 0):
                    statistics_per_team[i][2].append(0)
                else:
                    statistics_per_team[i][2].append(data.loc[j, "Home Team Goals"]+statistics_per_team[i][2][-1])
    return data, statistics_per_team

def compute_accumulated_scored_goals(statistics_per_team, data):
    for i in range(len(statistics_per_team)):
        for j in range(len(data)):
            if(data.loc[j, "Home Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][1]) == 0):
                    statistics_per_team[i][1].append(0)
                else:
                    statistics_per_team[i][1].append(data.loc[j, "Home Team Goals"]+statistics_per_team[i][1][-1])
            if(data.loc[j, "Away Team"] == statistics_per_team[i][0]):
                if(len(statistics_per_team[i][1]) == 0):
                    statistics_per_team[i][1].append(0)
                else:
                    statistics_per_team[i][1].append(data.loc[j, "Away Team Goals"]+statistics_per_team[i][1][-1])
    return data, statistics_per_team

def get_team_statistics_season(data):
    statistics_per_team = [] #Temporal cube for each team [["Alaves", [0,3,5]], ["Barcelona", [0,3,9]]]
    #Team name, goals scored, goals received, yellow cards, red cards, shots thrown, shots received,
    #shots target thron, shots target received, corners thrown, corners received, faults commited, faults received
    teams = []
    #data["Accumulated Total Goals Home"] = 0
    unique_teams = list(dict.fromkeys(data["Home Team"].values)) # Get unique teams
    
    for i in range(len(unique_teams)):
        teams.append(unique_teams[i])
        statistics_per_team.append(teams)
        #Initializing the arrays with 0 accumulated values
        statistics_per_team[i].append([0]) #Goals scored --> 1 index
        statistics_per_team[i].append([0]) #Goals received --> 2 index
        statistics_per_team[i].append([0]) #Yellow cards --> 3 index
        statistics_per_team[i].append([0]) #Red cards --> 4 index
        statistics_per_team[i].append([0]) #Shots thrown --> 5 index
        statistics_per_team[i].append([0]) #Shots received --> 6 index
        statistics_per_team[i].append([0]) #Shots target thrown --> 7 index
        statistics_per_team[i].append([0]) #Shots target received --> 8 index
        statistics_per_team[i].append([0]) #Corners thrown --> 9 index
        statistics_per_team[i].append([0]) #Corners received --> 10 index
        statistics_per_team[i].append([0]) #Faults commited --> 11 index
        statistics_per_team[i].append([0]) #Faults received --> 12 index
        teams = []
        
    data, statistics_per_team = compute_accumulated_scored_goals(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_goals(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_yellow_cards(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_red_cards(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_thrown_shots(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_shots(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_thrown_shots_target(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_shots_target(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_thrown_corners(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_corners(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_commited_faults(statistics_per_team, data)
    data, statistics_per_team = compute_accumulated_received_faults(statistics_per_team, data)
                    
    return data, statistics_per_team

def integrate_computed_accumulated_data_per_team(statistics_per_team, data):
    ACCUMULATED_COLUMNS_HOME_TEAM = ["Home Team Accumulated Scored Goals", "Home Team Accumulated Received Goals",
                                     "Home Team Accumulated Yellow Cards", "Home Team Accumulated Red Cards",
                                     "Home Team Accumulated Thrown Shots", "Home Team Accumulated Received Shots",
                                     "Home Team Accumulated Thrown Shots Target",
                                     "Home Team Accumulated Received Shots Target",
                                     "Home Team Accumulated Thrown Corners",
                                     "Home Team Accumulated Received Corners",
                                     "Home Team Accumulated Commited Faults",
                                     "Home Team Accumulated Received Faults"]
    ACCUMULATED_COLUMNS_AWAY_TEAM = ["Away Team Accumulated Scored Goals", "Away Team Accumulated Received Goals",
                                     "Away Team Accumulated Yellow Cards", "Away Team Accumulated Red Cards",
                                     "Away Team Accumulated Thrown Shots", "Away Team Accumulated Received Shots",
                                     "Away Team Accumulated Thrown Shots Target",
                                     "Away Team Accumulated Received Shots Target",
                                     "Away Team Accumulated Thrown Corners",
                                     "Away Team Accumulated Received Corners",
                                     "Away Team Accumulated Commited Faults",
                                     "Away Team Accumulated Received Faults"]
    counter_index = 0
    counter = 0
    start_counting = False
    for i in range(len(data)):
        for team_statistics in statistics_per_team:
            if(data.at[i, "Home Team"] == team_statistics[0]):
                feature_index = 1
                for accumulated_column_home_team in ACCUMULATED_COLUMNS_HOME_TEAM:
                    data.at[i, accumulated_column_home_team] = team_statistics[feature_index][counter_index]
                    feature_index += 1
            if(data.at[i, "Away Team"] == team_statistics[0]):
                feature_index = 1
                for accumulated_column_away_team in ACCUMULATED_COLUMNS_AWAY_TEAM:
                    data.at[i, accumulated_column_away_team] = team_statistics[feature_index][counter_index]
                    feature_index += 1
        counter += 1
        if counter == 9:
            start_counting = True
        if start_counting:
            if i%10 == 0:
                counter_index += 1
            
    return data

def preprocess_data(data):
    data = data[COLUMNS_TO_MANTAIN]
    data.columns = READABLE_HEADER
    data["Match Result"] = data["Match Result"].apply(match_result_to_numeric)
    data = fill_na_values(data)
    data = initialize_accumulated_statistics(data)
    data, statistics_per_team = get_team_statistics_season(data)
    data = integrate_computed_accumulated_data_per_team(statistics_per_team, data)
    
    return data

def read_preprocess_folder(folder):
    pd.set_option('display.max_columns', None)
    files = [f for f in listdir(folder)]
    sorted_files = []
    sorted_files = files.sort()
    count = 0
    data = pd.read_csv(folder + "/" + files[0])
    data = preprocess_data(data)
    for file in files:
        if count > 0:
            temp_data = pd.read_csv(FOLDER_PATH + "/" + file)
            temp_data = preprocess_data(temp_data)
            data = pd.concat([data, temp_data], axis = 0)
        count += 1
    
    return data

def main():
    data = read_preprocess_folder(FOLDER_PATH)
    actual_date = date.today().strftime("%b-%d-%Y")
    data.to_csv("Preprocessed-Data/"+actual_date, index=False)
    return data


if __name__ == "__main__":
    main()
