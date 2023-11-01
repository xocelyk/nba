import time
from sportsipy.nba.teams import Teams
from sportsipy.nba.schedule import Schedule
from sportsipy.nba.boxscore import Boxscore
import pandas as pd
import numpy as np
import utils
import csv


def get_team_names(year=2024):
    '''
    Returns a dictionary of team names to abbreviations
    '''
    year = 2024
    names_to_abbr = {}
    names_to_abbr['Houston Rockets'] = 'HOU'
    schedule = Schedule('HOU', year=year)
    for game in schedule:
        game = game.dataframe
        opponent_name = game['opponent_name'].iloc[0]
        if opponent_name not in names_to_abbr:
            names_to_abbr[opponent_name] = game['opponent_abbr'].iloc[0]
    return names_to_abbr

def load_year_data(year=2024):
    '''
    pre-loader for update_data function
    '''
    data = []
    filename = 'data/games/year_data_{}.csv'.format(year)
    df = pd.read_csv(filename)
    df = df[df['completed'] == True]
    for idx, row in df.iterrows():
        string = row['boxscore_id']
        row['date_string'] = string[4:6] + '/' + string[6:8] + '/' + string[0:4]
        row['date'] = pd.to_datetime(row['date_string'])
        data.append([row['boxscore_id'], row['date'], row['team'], row['opponent'], row['team_score'], row['opponent_score'], 'Home', row['pace'], row['completed'], year])
    return data

def update_data(names_to_abbr, year=2024, preload=True):
    '''
    Returns a dataframe of all the data for the given year
    '''
    if preload:
        data = load_year_data(year)
        boxscore_tracked = [row[0] for row in data]
    else:
        data = []
        boxscore_tracked = []
    abbr_to_name = {v: k for k, v in names_to_abbr.items()}
    abbrs = list(names_to_abbr.values())

    # update data
    for abbr in abbrs:
        schedule = Schedule(abbr, year=year)
        time.sleep(5)
        for game in schedule:
            if game.boxscore_index in boxscore_tracked:
                continue
            location = game.location
            if location == 'Home':
                row = [game.boxscore_index, game.date, abbr, game.opponent_abbr, game.points_scored, game.points_allowed, location]
            else:
                if location == 'Away':
                    location = 'Home'
                row = [game.boxscore_index, game.date, game.opponent_abbr, abbr, game.points_allowed, game.points_scored, location]
            if game.points_scored is None:
                pace = None
            else:
                pace = Boxscore(game.boxscore_index).pace
                time.sleep(5)
            row.append(pace)
            if str(row[-3]).isdigit() and str(row[-4]).isdigit():
                row.append(True)
            else:
                row.append(False)
            if not isinstance(game.boxscore_index, str):
                continue
            row.append(year)
            data.append(row)
            boxscore_tracked.append(game.boxscore_index)

    # create dataframe
    data = pd.DataFrame(data, columns=['boxscore_id', 'date', 'team', 'opponent', 'team_score', 'opponent_score', 'location', 'pace', 'completed', 'year'])
    data['team_name'] = data['team'].apply(lambda x: abbr_to_name[x])
    data['opponent_name'] = data['opponent'].apply(lambda x: abbr_to_name[x])
    data['margin'] = data['team_score'] - data['opponent_score']
    data = data[['boxscore_id', 'date', 'team', 'opponent', 'team_name', 'opponent_name', 'team_score', 'opponent_score', 'margin', 'location', 'pace', 'completed', 'year']]
    data.set_index('boxscore_id', inplace=True)
    data.to_csv('data/games/year_data_{}.csv'.format(year))
    return data

def load_regular_season_win_totals_futures():
    res = {}
    filename = 'data/regular_season_win_totals_odds_archive.csv'
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    first_row = True
    header = []
    for row in data:
        if first_row:
            header = row
            first_row = False
            continue
        else:
            team = row[0]
            res[team] = {}
            for i in range(1, len(row)):
                res[team][header[i]] = float(row[i]) if row[i] != '' else np.nan
    return res

def load_training_data(names, update=True, reset=False, start_year=2010, stop_year=2024):
    '''
    Loads the data from start_year to stop_year and returns a dataframe with the data
    Data includes each game with data, team rating, opp rating, team last year rating, opp last year rating, and num games into season

    Current Features:
    - home/away rating
    - home/away last year rating
    - number of games into season
    - home/away last year rating and number of games into season interaction
    - Adjusted margin of victory for last 10 games
    - Adjusted margin of victory for last 5 games
    - Adjusted margin of victory for last 3 games
    - Adjusted margin of victory for last 1 game

    Future Features:
    - Number of days of rest since last game
    '''

    all_data_archive = pd.read_csv(f'data/train_data.csv')
    all_data_archive.drop([col for col in all_data_archive.columns if 'Unnamed' in col], axis=1, inplace=True)
    win_totals_futures = load_regular_season_win_totals_futures()
    
    if update == True:
        all_data = []
        end_year_ratings_dct = {}
        first_year = True
        for year in range(start_year, stop_year+1):
            year_data = pd.read_csv(f'data/games/year_data_{year}.csv')
            year_data = year_data.sort_values('date')
            if 'team_abbr' in year_data.columns and 'team' not in year_data.columns:
                year_data['team'] = year_data['team_abbr']
                year_data['opponent'] = year_data['opponent_abbr']
            year_data['num_games_into_season'] = range(1, len(year_data) + 1)
            year_data = year_data[year_data['year'] == year]
            if 'team_abbr' in year_data.columns and 'team' not in year_data.columns:
                year_data.rename(columns={'team_abbr': 'team', 'opponent_abbr': 'opponent'}, inplace=True)
            year_data['date'] = pd.to_datetime(year_data['date'], format='mixed')
            end_year_ratings_dct[year] = {}
            abbrs = list(set(year_data['team']).union(set(year_data['opponent'])))
            games_to_date = year_data[year_data['completed'] == True]
            if year == stop_year:
                year_names = names
            else:
                year_names = None
            year_ratings = utils.get_em_ratings(games_to_date[games_to_date['completed'] == True], names=year_names)
            print(year)
            print('Year Ratings:', sorted(year_ratings.items(), key=lambda x: x[1], reverse=True))
            print()
            print()
            for team, rating in year_ratings.items():
                end_year_ratings_dct[year][team] = rating
            if first_year:
                first_year = False
                continue
            else:
                if reset or year == stop_year:
                    for team in abbrs:
                        if team not in end_year_ratings_dct[year - 1].keys():
                            # Some teams have changed names over the seasons--hard coding the changes for now
                            if team == 'BRK':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['NJN']
                                print('Linking NJN to BRK')
                            elif team == 'NOP':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['NOH']
                                print('Linking NOH to NOP')
                            elif team == 'CHO':
                                end_year_ratings_dct[year - 1][team] = end_year_ratings_dct[year - 1]['CHA']
                                print('Linking CHA to CHO')
                            else:
                                print('No Link Found: ', team)
                                end_year_ratings_dct[year - 1][team] = np.mean(list(end_year_ratings_dct[year - 1].values()))
                    end_year_ratings_df = pd.DataFrame(end_year_ratings_dct[year - 1].items(), columns=['team', 'rating'])
                    end_year_ratings_df['year'] = year - 1
                    end_year_ratings_df.to_csv(f'data/end_year_ratings/{year - 1}.csv', index=False)
                    year_data['last_year_team_rating'] = year_data.apply(lambda x: end_year_ratings_dct[year - 1][x['team']], axis=1)
                    year_data['last_year_opp_rating'] = year_data.apply(lambda x: end_year_ratings_dct[year - 1][x['opponent']], axis=1)
                    year_data['num_games_into_season'] = year_data.apply(lambda x: len(year_data[year_data['date'] < x['date']]), axis=1)
                    year_data['team_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(year)][x['team']], axis=1)
                    year_data['opp_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(year)][x['opponent']], axis=1)
                    year_data['margin'] = year_data['team_score'] - year_data['opponent_score']

                    year_data_temp = []
                    for i, date in enumerate(sorted(year_data['date'].unique())):
                        print('Progress:', i+1, '/', len(year_data['date'].unique()), end='\r')
                        games_to_date = year_data[year_data['date'] < date]
                        games_on_date = year_data[year_data['date'] == date]
                        if len(games_to_date[games_to_date['completed'] == True]) > 100:
                            cur_ratings = utils.get_em_ratings(games_to_date[games_to_date['completed'] == True])
                        else:
                            # If not enough data to get EM ratings for every team, ratings default to 0
                            cur_ratings = {team: 0 for team in end_year_ratings_dct[year-1].keys()}

                        games_on_date['team_rating'] = games_on_date.apply(lambda x: cur_ratings[x['team']], axis=1)
                        games_on_date['opp_rating'] = games_on_date.apply(lambda x: cur_ratings[x['opponent']], axis=1)
                        games_on_date = games_on_date[['team', 'opponent', 'team_rating', 'opp_rating', 'last_year_team_rating', 'last_year_opp_rating', 'margin', 'pace', 'num_games_into_season', 'date', 'year']]
                        year_data_temp += games_on_date.values.tolist()
                    
                    year_data = pd.DataFrame(year_data_temp, columns=['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'pace', 'num_games_into_season', 'date', 'year'])
                    year_data = utils.last_n_games(year_data, 10)
                    year_data = utils.last_n_games(year_data, 5)
                    year_data = utils.last_n_games(year_data, 3)
                    year_data = utils.last_n_games(year_data, 1)

                    year_data['completed'] = year_data['margin'].apply(lambda x: True if not np.isnan(x) else False)
                    year_data['date'] = pd.to_datetime(year_data['date']).dt.date
                    year_data['team_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(x['year'])][x['team']], axis=1).astype(float)
                    year_data['opponent_win_total_future'] = year_data.apply(lambda x: win_totals_futures[str(x['year'])][x['opponent']], axis=1).astype(float)
                    # year_data to list of dictionaries
                    year_data = year_data.to_dict('records')
                    all_data += year_data

                else:
                    year_data = all_data_archive[all_data_archive['year'] == year]
                    year_data = year_data.to_dict('records')
                    all_data += year_data

        # all_data = pd.DataFrame(all_data, columns=['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin','pace', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game'])
        all_data = pd.DataFrame(all_data)
        all_data.to_csv(f'data/train_data.csv', index=False)
    else:
        all_data = pd.read_csv(f'data/train_data.csv')
        all_data.drop([col for col in all_data.columns if 'Unnamed' in col], axis=1, inplace=True)
        all_data['team_win_total_future'] = all_data.apply(lambda x: win_totals_futures[str(x['year'])][x['team']], axis=1).astype(float)
        all_data['opponent_win_total_future'] = all_data.apply(lambda x: win_totals_futures[str(x['year'])][x['opponent']], axis=1).astype(float)
        all_data.to_csv(f'data/train_data.csv')
    
    all_data = add_days_since_most_recent_game(all_data)
    all_data.to_csv(f'data/train_data.csv', index=False)
    return all_data

def add_days_since_most_recent_game(df, cap=10):
    # TODO: only apply this to recently completed games
    df['team_days_since_most_recent_game'] = cap
    df['opponent_days_since_most_recent_game'] = cap
    df['date'] = pd.to_datetime(df['date']).dt.date
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        year_data = year_data.sort_values('date')
        team_most_recent_game_date = {team: None for team in year_data['team'].unique()}
        # team_most_recent_game_time_diff = {team: None for team in year_data['team'].unique()}
        for i, row in year_data.iterrows():
            team = row['team']
            if team_most_recent_game_date[team] is None:
                team_most_recent_game_date[team] = row['date']
                df.loc[i, 'team_days_since_most_recent_game'] = cap
            else:
                df.loc[i, 'team_days_since_most_recent_game'] = min((row['date'] - team_most_recent_game_date[team]).days, cap)
                team_most_recent_game_date[team] = row['date']
            opponent = row['opponent']
            if team_most_recent_game_date[opponent] is None:
                team_most_recent_game_date[opponent] = row['date']
                df.loc[i, 'opponent_days_since_most_recent_game'] = cap
            else:
                df.loc[i, 'opponent_days_since_most_recent_game'] = min((row['date'] - team_most_recent_game_date[opponent]).days, cap)
                team_most_recent_game_date[opponent] = row['date']
    return df
            
            

            

            


