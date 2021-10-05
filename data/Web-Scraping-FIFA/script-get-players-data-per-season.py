import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

ROOT_PATH = "https://sofifa.com"
SPANISH_FIRST_DIVISION_PLAYERS_STATS = "/players?type=all&lg[0]=53&showCol[]=ae&showCol[]=oa&showCol[]=pt&showCol[]=bp&showCol[]=ta&showCol[]=ts&showCol[]=to&showCol[]=tp&showCol[]=te&showCol[]=td&showCol[]=tg&showCol[]=tt"
FOLDER_PLAYERS_PER_SEASON = 'Players-Season'

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
        season_name = 'Players Season ' + '200' + str(season-1) + '-200' + str(season)
    else:
        season = int(fifa_year)
        if fifa_year == '10':
            season_name = 'Players Season ' + '2009-20' + str(season)
        else:
            season_name = 'Players Season ' + '20' + str(season-1) + '-20' + str(season)
    
    return season_name

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
        day = "0" + day.strip(',')
    
    new_date_format = day + "/" + month + "/" + year

    return new_date_format

def get_pagination_players(players_data, update_date):
    player_names = []
    ages = []
    overall_ratings = []
    potentials = []
    team_contracts = []
    best_positions = []
    total_attackings = []
    total_skills = []
    total_movements = []
    total_powers = []
    total_mentallities = []
    total_defendings = []
    total_goalkeepings = []
    total_stats = []

    for player_data in players_data:
        stats = player_data.find_all('td')
        for stat in stats:
            if stat['class'][0] == 'col-name':
                a_tags = stat.find_all('a', {"class": "tooltip"})
                try:
                    player_names.append(a_tags[0].div.text)
                except:
                    pass
            if stat['class'][0] == 'col-name':
                try:
                    team_div = stat.find_all('div', {"class": "bp3-text-overflow-ellipsis"})
                    team_contracts.append(team_div[0].a.text)
                except:
                    pass
            if len(stat['class']) > 1 and stat['class'][1] == 'col-ae':
                ages.append(stat.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-oa':
                overall_ratings.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-pt':
                potentials.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-bp':
                best_positions.append(stat.a.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-ta':
                total_attackings.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-ts':
                total_skills.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-to':
                total_movements.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-tp':
                total_powers.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-te':
                total_mentallities.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-td':
                total_defendings.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-tg':
                total_goalkeepings.append(stat.span.text)
            elif len(stat['class']) > 1 and stat['class'][1] == 'col-tt':
                total_stats.append(stat.span.text)

    update_date = change_date_format(update_date)

    players_stats = pd.DataFrame({
        'Date' : [update_date for i in range(len(player_names))],
        'Player' : player_names,
        'Age' : ages,
        'Overall Rating' : overall_ratings,
        'Potential' : potentials,
        'Team Contract' : team_contracts,
        'Best Position' : best_positions,
        'Total Attacking' : total_attackings,
        'Total Skill' : total_skills,
        'Total Movement' : total_movements,
        'Total Power' : total_powers,
        'Total Mentality' : total_mentallities,
        'Total Defending' : total_defendings,
        'Total Goalkeeping' : total_goalkeepings,
        'Total Stats' : total_stats
    })

    return players_stats

if __name__ == "__main__":

    scraped_players_data = pd.DataFrame({
        'Date' : [],
        'Player' : [],
        'Age' : [],
        'Overall Rating' : [],
        'Potential' : [],
        'Team Contract' : [],
        'Best Position' : [],
        'Total Attacking' : [],
        'Total Skill' : [],
        'Total Movement' : [],
        'Total Power' : [],
        'Total Mentality' : [],
        'Total Defending' : [],
        'Total Goalkeeping' : [],
        'Total Stats' : []
    })

    result = requests.get(ROOT_PATH + "" + SPANISH_FIRST_DIVISION_PLAYERS_STATS)

    if result.status_code == 200:
        source = result.content
        soup = BeautifulSoup(source)
        season_links = get_season_links(soup)
        for season in season_links:
            fifa_year = season[251:253]
            result = requests.get(ROOT_PATH + "" + season)
            if result.status_code == 200:
                source = result.content
                soup = BeautifulSoup(source)
                update_links, update_dates = get_update_links(soup)
                for update_link, update_date in list(zip(update_links, update_dates)):
                    print(update_link)
                    result = requests.get(ROOT_PATH + "" + update_link)
                    if result.status_code == 200:
                        source = result.content
                        soup = BeautifulSoup(source)

                        keep_going = True
                        while keep_going:
                            players_table = soup.table
                            players_data = players_table.find_all('tr')
                            pagination = soup.find_all('div', {"class": "pagination"})
                            pagination_buttons = pagination[0].find_all('a')
                            if len(pagination_buttons) == 1:
                                if pagination_buttons[0].span.text == 'Next':
                                    pagination_link = pagination_buttons[0].attrs['href']
                                    print("Pagination link " + pagination_link)
                                else:
                                    keep_going = False
                            else:
                                pagination_link = pagination_buttons[1].attrs['href']
                                print("Pagination link " + pagination_link)

                            players_stats = get_pagination_players(players_data, update_date)
                            pagination_request = requests.get(ROOT_PATH + "" + pagination_link)

                            if pagination_request.status_code == 200:
                                source = pagination_request.content
                                soup = BeautifulSoup(source)

                                scraped_players_data =  pd.concat([scraped_players_data, players_stats], axis=0)
                            else:
                                keep_going = False
                                print("Unable to connect to the website. Error " + result.status_code)

                    else:
                        print("Unable to connect to the website. Error " + result.status_code)

                if not os.path.exists(FOLDER_PLAYERS_PER_SEASON):
                    os.mkdir(FOLDER_PLAYERS_PER_SEASON)

                scraped_players_data.to_csv(FOLDER_PLAYERS_PER_SEASON + '/' + get_season_name(fifa_year), index=False)

                scraped_players_data = pd.DataFrame({
                    'Date' : [],
                    'Player' : [],
                    'Age' : [],
                    'Overall Rating' : [],
                    'Potential' : [],
                    'Team Contract' : [],
                    'Best Position' : [],
                    'Total Attacking' : [],
                    'Total Skill' : [],
                    'Total Movement' : [],
                    'Total Power' : [],
                    'Total Mentality' : [],
                    'Total Defending' : [],
                    'Total Goalkeeping' : [],
                    'Total Stats' : []
                })

            else:
                print("Unable to connect to the website. Error " + result.status_code)

    else:
        print("Unable to connect to the website. Error " + result.status_code)