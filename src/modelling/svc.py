from math import nan
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
# PLS partial least squares
# Feature tools python --> genera x agregadas 

def knn_clusters(df: pd.DataFrame):
    pass

def model_ratios(df: pd.DataFrame):
    X = df.drop(['Full_Time_Result'], axis=1)
    X = StandardScaler().fit_transform(X)
    Y = df.Full_Time_Result
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=False)
    model = SVC()
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))

def main():
    df = pd.read_csv('../../inputs/ready_data/preprocessed_all_matches.csv', parse_dates=['Date'])
    df.dropna(subset=['HomeTeam', 'AwayTeam'], inplace=True)
    df = df[['Date', 'Season', 'Full_Time_Result', 'Home Overall Score', 'Home Attack Score', 'Home Middle Score', 'Home Defensive Score', 'Home Budget',
            'Away Overall Score', 'Away Attack Score', 'Away Middle Score', 'Away Defensive Score', 'Away Budget', 'Difference_Overall_Score',
            'Difference_Attack_Score', 'Difference_Middle_Score', 'Difference_Defensive_Score', 'Difference_Budget', 'HOME_TRUESKILL_MU_NO_RESET',
            'AWAY_TRUESKILL_MU_NO_RESET','HOME_TRUESKILL_SIGMA_NO_RESET', 'AWAY_TRUESKILL_SIGMA_NO_RESET','DRAW_CHANCE_NO_RESET',
            'HOME_TRUESKILL_MU_SEASON', 'AWAY_TRUESKILL_MU_SEASON', 'HOME_TRUESKILL_SIGMA_SEASON', 'AWAY_TRUESKILL_SIGMA_SEASON',
            'DRAW_CHANCE_SEASON', 'HOME_ID', 'AWAY_ID',
            'HOME_WINS_HOME', 'HOME_DRAWS_HOME', 'HOME_LOSSES_HOME', 'AWAY_WINS_HOME', 'AWAY_DRAWS_HOME', 'AWAY_LOSSES_HOME', 'HOME_WINS_AWAY', 'HOME_DRAWS_AWAY', 
            'HOME_LOSSES_AWAY', 'AWAY_WINS_AWAY', 'AWAY_DRAWS_AWAY', 'AWAY_LOSSES_AWAY']]
    df.drop(['Date'], axis=1, inplace=True)

    knn_clusters(df)
    # model_ratios(df)


if __name__ == '__main__':
    main()