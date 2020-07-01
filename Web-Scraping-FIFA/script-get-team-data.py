import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

ROOT_PATH = "https://sofifa.com"
SPANISH_FIRST_DIVISION_TEAMS = "/teams?type=all&lg[0]=53"
FOLDER_TEAMS_PER_SEASON = 'Teams-Season'

def change_date_format(date):
    month_day_year = date.split(" ")

    month = month_day_year[0]
    day = month_day_year[1].strip(',')
    year = month_day_year[2]

    switcher = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12"
    }

    month = switcher.get(month)

    if len(day) == 1:
        day = "0" + day
    
    new_date_format = day + "/" + month + "/" + year

    return new_date_format

def get_season_links(soap):
    season_links = []
    divs = soap.find_all('div', {"class": "bp3-menu"})
    menu_items = divs[0].find_all('a')
    for menu_item in list(reversed(menu_items)):
        season_links.append(menu_item.attrs['href'])

    return season_links

def get_update_links(soap):
    update_links = []
    update_dates = []

    divs = soap.find_all('div', {"class": "bp3-menu"})
    menu_items = divs[1].find_all('a')
    for menu_item in list(reversed(menu_items)):
        update_links.append(menu_item.attrs['href'])
        update_dates.append(menu_item.text)

    return update_links, update_dates

def get_season_name(fifa_year):
    if fifa_year[0] == '0':
        season = int(fifa_year[1])
        season_name = 'Teams Season ' + '200' + str(season-1) + '-200' + str(season)
    else:
        season = int(fifa_year)
        if fifa_year == '10':
            season_name = 'Teams Season ' + '2009-20' + str(season)
        else:
            season_name = 'Teams Season ' + '20' + str(season-1) + '-20' + str(season)
    
    return season_name

if __name__ == "__main__":

    scraped_teams_data = pd.DataFrame({
        'Date' : [],
        'Team' : [],
        'Overall Score' : [],
        'Attack Score' : [],
        'Middle Score' : [],
        'Defensive Score' : [],
        'Budget' : []
    })

    result = requests.get(ROOT_PATH + "" + SPANISH_FIRST_DIVISION_TEAMS)

    if result.status_code == 200:
        source = result.content
        soup = BeautifulSoup(source)
        season_links = get_season_links(soup)
        for season in season_links:
            fifa_year = season[31:33]
            result = requests.get(ROOT_PATH + "" + season)
            if result.status_code == 200:
                source = result.content
                soup = BeautifulSoup(source)
                update_links, update_dates = get_update_links(soup)
                for update_link, update_date in list(zip(update_links, update_dates)):
                    result = requests.get(ROOT_PATH + "" + update_link)
                    if result.status_code == 200:
                        source = result.content
                        soup = BeautifulSoup(source)

                        team_names = []
                        overall_scores = []
                        attack_scores = []
                        middle_scores = []
                        defensive_scores = []
                        budgets = []

                        teams_table = soup.table
                        teams_data = teams_table.find_all('tr')

                        for team_data in teams_data:
                            stats = team_data.find_all('td')
                            for stat in stats:
                                if stat['class'][0] == 'col-name-wide':
                                    team_names.append(stat.a.div.text)
                                elif len(stat['class']) > 1 and stat['class'][1] == 'col-oa':
                                    overall_scores.append(stat.span.text)
                                elif len(stat['class']) > 1 and stat['class'][1] == 'col-at':
                                    attack_scores.append(stat.span.text)
                                elif len(stat['class']) > 1 and stat['class'][1] == 'col-md':
                                    middle_scores.append(stat.span.text)
                                elif len(stat['class']) > 1 and stat['class'][1] == 'col-df':
                                    defensive_scores.append(stat.span.text)
                                elif len(stat['class']) > 1 and stat['class'][1] == 'col-tb':
                                    if 'K' in stat.text:
                                        budgets.append('0.' + stat.text.strip('€K'))
                                    else:
                                        budgets.append(stat.text.strip('€M'))

                        update_date = change_date_format(update_date)
                        print(update_link + "    " + update_date)

                        teams_stats = pd.DataFrame({
                            'Date' : [update_date for i in range(len(team_names))],
                            'Team' : team_names,
                            'Overall Score' : overall_scores,
                            'Attack Score' : attack_scores,
                            'Middle Score' : middle_scores,
                            'Defensive Score' : defensive_scores,
                            'Budget' : budgets
                        })

                        scraped_teams_data =  pd.concat([scraped_teams_data, teams_stats], axis=0)

                    else:
                        print("Unable to connect to the website. Error " + result.status_code)
                    
                if not os.path.exists(FOLDER_TEAMS_PER_SEASON):
                    os.mkdir(FOLDER_TEAMS_PER_SEASON)

                scraped_teams_data.to_csv(FOLDER_TEAMS_PER_SEASON + '/' + get_season_name(fifa_year), index=False)

                scraped_teams_data = pd.DataFrame({
                        'Date' : [],
                        'Team' : [],
                        'Overall Score' : [],
                        'Attack Score' : [],
                        'Middle Score' : [],
                        'Defensive Score' : [],
                        'Budget' : []
                })

            else:
                print("Unable to connect to the website. Error " + result.status_code)

    else:
        print("Unable to connect to the website. Error " + result.status_code)