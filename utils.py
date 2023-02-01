import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression
import time

x_features = 'team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game'

# TODO: build out model to retrive this number
# copying from Sagarin 1/25/23
HCA = 3.13

def calc_rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())

def get_em_ratings(df, cap=30, break_point=1e-10, max_iter=10):
    # TODO: make pace/efficiency based
    ratings = {team: 0 for team in df['team'].unique()}

    for _ in range(max_iter):
        prev_ratings = ratings.copy()
        ratings_temp = {team: [] for team in ratings.keys()}
        for boxscore_id, row in df.iterrows():
            home_margin = row['margin']
            if home_margin > cap:
                home_margin = cap + np.log(home_margin - cap + 1)
            elif home_margin < -cap:
                home_margin = -cap - np.log(-home_margin - cap + 1)
            home_game_score = home_margin + ratings[row['opponent']] - HCA
            away_game_score = -home_margin + ratings[row['team']] + HCA
            ratings_temp[row['team']].append(home_game_score)
            ratings_temp[row['opponent']].append(away_game_score)
        ratings_temp = {team: np.mean(ratings_temp[team]) for team in ratings_temp.keys()}
        # print data frame of ratings
        ratings = ratings_temp.copy()


        # # get l2 distance between ratings
        # l2_dist = 0
        # for team, rating in ratings.items():
        #     l2_dist += (rating - prev_ratings[team]) ** 2
        # l2_dist = np.sqrt(l2_dist)
        # print(l2_dist)
        # if l2_dist < break_point:
        #     break
    ratings_df = pd.DataFrame.from_dict(ratings_temp, orient='index', columns=['rating'])
    ratings_df = ratings_df.sort_values(by='rating', ascending=False)
    
    return ratings

def get_adjacency_matrix(df):
    '''
    df needs only three features: team, opponent, and margin
    creates the adjacency matrix for pagerank
    each team a node, each MOV an edge weight
    '''
    adjacency_matrix = np.zeros((30, 30))
    abbr_to_index = {}
    abbr_to_index = {abbr: i for i, abbr in enumerate(df['team'].unique())}
    for idx, game_data in df.iterrows():
        team_score = sigmoid_margin(game_data['margin'])
        opponent_score = sigmoid_margin(-game_data['margin'])
        team_idx = abbr_to_index[game_data['team']]
        opponent_idx = abbr_to_index[game_data['opponent']]
        adjacency_matrix[team_idx][opponent_idx] = team_score
        adjacency_matrix[opponent_idx][team_idx] = opponent_score
    adjacency_matrix = normalize_matrix(adjacency_matrix)
    return adjacency_matrix, abbr_to_index

def normalize_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            num_games_played = matrix[i][j] + matrix[j][i]
            if num_games_played == 0:
                continue
            matrix[i][j] /= num_games_played
            matrix[j][i] /= num_games_played
    for row_index in range(matrix.shape[0]):
        num_teams_played = sum(matrix[row_index] != 0)
        if num_teams_played == 0:
            continue
        matrix[row_index] /= num_teams_played

    return matrix

def eigenrank(df):
    adj_mat, abbr_to_index = get_adjacency_matrix(df)
    index_to_abbr = {v: k for k, v in abbr_to_index.items()}
    val, vec = eigs(adj_mat, which='LM', k=1)
    vec = np.ndarray.flatten(abs(vec))
    sorted_indices = vec.argsort()
    ranked = {index_to_abbr[i]: vec[i] for i in sorted_indices}
    return ranked

def sigmoid_margin(margin, k=.05):
    return 1 / (1 + np.exp(-k * margin))

def get_em_ratings_from_eigenratings(df, ratings):
    em_ratings_dct = {}
    df_with_ratings = df.copy()
    df_with_ratings['team_rating'] = df_with_ratings['team'].apply(lambda x: ratings[x])
    df_with_ratings['opponent_rating'] = df_with_ratings['opponent'].apply(lambda x: ratings[x])
    X = df_with_ratings[['team_rating', 'opponent_rating']].values
    y = df_with_ratings['margin'].values
    model = LinearRegression()
    model.fit(X, y)
    mean_rating = np.mean(list(ratings.values()))
    for team, rating in ratings.items():
        em_ratings_dct[team] = model.predict([[rating, mean_rating]])[0]
    mean_em_rating = np.mean(list(em_ratings_dct.values()))
    for team, rating in em_ratings_dct.items():
        em_ratings_dct[team] = rating - mean_em_rating
    return em_ratings_dct

def last_n_games(year_data, n):
    year_data = year_data.sort_values(by='date', ascending=True)
    year_data['last_{}_team_rating'.format(n)] = np.nan
    year_data['last_{}_opponent_rating'.format(n)] = np.nan
    for team in list(set(year_data['team'].unique().tolist() + year_data['opponent'].unique().tolist())):
        # team data is where team is team or opponent is team
        team_data = year_data[(year_data['team'] == team) | (year_data['opponent'] == team)]
        # adj_margin = margin + opp_rating - HCA if team == team else margin + team_rating + HCA
        team_data['team_adj_margin'] = team_data.apply(lambda x: x['margin'] + x['opponent_rating'] - HCA if x['team'] == team else x['margin'] + x['team_rating'] + HCA, axis=1)
        team_data['last_{}_rating'.format(n)] = team_data['team_adj_margin'].rolling(n, closed='left').mean()
        # fillna with 0
        team_data['last_{}_rating'.format(n)] = team_data['last_{}_rating'.format(n)].fillna(0)
        team_data['last_{}_team_rating'.format(n)] = team_data.apply(lambda x: x['last_{}_rating'.format(n)] if x['team'] == team else np.nan, axis=1)
        team_data['last_{}_opponent_rating'.format(n)] = team_data.apply(lambda x: x['last_{}_rating'.format(n)] if x['opponent'] == team else np.nan, axis=1)

        # merge team data with year data, only replace if na
        year_data['last_{}_team_rating'.format(n)] = year_data['last_{}_team_rating'.format(n)].combine_first(team_data['last_{}_team_rating'.format(n)])
        year_data['last_{}_opponent_rating'.format(n)] = year_data['last_{}_opponent_rating'.format(n)].combine_first(team_data['last_{}_opponent_rating'.format(n)])
    return year_data

def get_last_n_games_dict(completed_games, n_lst, teams_on_date=None):
    res = {n: {} for n in n_lst}
    completed_games.sort_values(by='date', ascending=False, inplace=True)
    for team in list(set(completed_games['team'].unique().tolist() + completed_games['opponent'].unique().tolist())):
        if teams_on_date:
            if team not in teams_on_date:
                continue
        
        team_data = completed_games[(completed_games['team'] == team) | (completed_games['opponent'] == team)].sort_values(by='date', ascending=False).iloc[:max(n_lst)]
        team_data = duplicate_games(team_data)
        team_data = team_data[team_data['team'] == team]

        for n in n_lst:
            team_vals = {}
            team_data = team_data.iloc[:n]
            team_data['team_adj_margin'] = team_data.apply(lambda x: x['margin'] + x['opponent_rating'] - HCA, axis=1)
            if len(team_data) < n:
                team_val = 0
            else:
                vals = []
                for idx, row in team_data.iterrows():
                    vals.append(row['toeam_adj_margin'])
                team_val = np.mean(vals)
            team_vals[team] = team_val
            res[n][team] = team_val
    return res

def add_days_since_most_recent_game_to_df(df):
    for year in df['year'].unique():
        print('adding most recent game: {}'.format(year))
        year_data = df[df['year'] == year]
        for date in year_data['date'].unique():
            date_data = year_data[year_data['date'] == date]
            for team in date_data['team'].unique():
                team_days_since_most_recent_game = days_since_most_recent_game(team, date, year_data)
                df.loc[(df['team'] == team) & (df['date'] == date), 'team_days_since_most_recent_game'] = team_days_since_most_recent_game
            for opponent in date_data['opponent'].unique():
                opponent_days_since_most_recent_game = days_since_most_recent_game(opponent, date, year_data)
                df.loc[(df['opponent'] == opponent) & (df['date'] == date), 'opponent_days_since_most_recent_game'] = opponent_days_since_most_recent_game
    return df


def days_since_most_recent_game(team, date, games):
    '''
    returns the number of days since the most recent game for the team on the given date
    '''
    cap = 10
    team_data = games[(games['team'] == team) | (games['opponent'] == team)]
    team_data = duplicate_games_training_data(team_data)
    team_data = team_data[team_data['date'] < date]
    date = pd.to_datetime(date)

    team_data['date'] = pd.to_datetime(team_data['date'])
    if len(team_data) == 0:
        return cap
    else:
        return min(cap, (date - team_data.iloc[0]['date']).days)

def duplicate_games(df):
    '''
    duplicates the games in the dataframe so that the team and opponent are switched
    '''
    features = ['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future', 'pace', 'team_win']
    def reverse_game(row):
        team = row['opponent']
        opponent = row['team']
        team_rating = row['opponent_rating']
        opponent_rating = row['team_rating']
        last_year_team_rating = row['last_year_opponent_rating']
        last_year_opponent_rating = row['last_year_team_rating']
        margin = -row['margin'] + 2 * HCA
        num_games_into_season = row['num_games_into_season']
        date = row['date']
        year = row['year']
        team_last_10_rating = row['opponent_last_10_rating']
        opponent_last_10_rating = row['team_last_10_rating']
        team_last_5_rating = row['opponent_last_5_rating']
        opponent_last_5_rating = row['team_last_5_rating']
        team_last_3_rating = row['opponent_last_3_rating']
        opponent_last_3_rating = row['team_last_3_rating']
        team_last_1_rating = row['opponent_last_1_rating']
        opponent_last_1_rating = row['team_last_1_rating']
        completed = row['completed']
        team_win_total_future = row['opponent_win_total_future']
        opponent_win_total_future = row['team_win_total_future']
        pace = row['pace']
        team_win = int(not (bool(row['team_win'])))
        return [team, opponent, team_rating, opponent_rating, last_year_team_rating, last_year_opponent_rating, margin, num_games_into_season, date, year, team_last_10_rating, opponent_last_10_rating, team_last_5_rating, opponent_last_5_rating, team_last_3_rating, opponent_last_3_rating, team_last_1_rating, opponent_last_1_rating, completed, team_win_total_future, opponent_win_total_future, pace, team_win]

    duplicated_games = []
    for idx, game in df.iterrows():
        duplicated_games.append(reverse_game(game))
    duplicated_games = pd.DataFrame(duplicated_games, columns=features)
    df = pd.concat([df, duplicated_games])
    return df

def duplicate_games_training_data(df):
    features = ['team', 'opponent', 'team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'margin', 'num_games_into_season', 'date', 'year', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'completed', 'team_win_total_future', 'opponent_win_total_future']
    def reverse_game(row):
        team = row['opponent']
        opponent = row['team']
        team_rating = row['opponent_rating']
        opponent_rating = row['team_rating']
        last_year_team_rating = row['last_year_opponent_rating']
        last_year_opponent_rating = row['last_year_team_rating']
        margin = -row['margin'] + 2 * HCA
        num_games_into_season = row['num_games_into_season']
        date = row['date']
        year = row['year']
        team_last_10_rating = row['opponent_last_10_rating']
        opponent_last_10_rating = row['team_last_10_rating']
        team_last_5_rating = row['opponent_last_5_rating']
        opponent_last_5_rating = row['team_last_5_rating']
        team_last_3_rating = row['opponent_last_3_rating']
        opponent_last_3_rating = row['team_last_3_rating']
        team_last_1_rating = row['opponent_last_1_rating']
        opponent_last_1_rating = row['team_last_1_rating']
        completed = row['completed']
        team_win_total_future = row['opponent_win_total_future']
        opponent_win_total_future = row['team_win_total_future']
        return [team, opponent, team_rating, opponent_rating, last_year_team_rating, last_year_opponent_rating, margin, num_games_into_season, date, year, team_last_10_rating, opponent_last_10_rating, team_last_5_rating, opponent_last_5_rating, team_last_3_rating, opponent_last_3_rating, team_last_1_rating, opponent_last_1_rating, completed, team_win_total_future, opponent_win_total_future]
    
    duplicated_games = []
    for idx, game in df.iterrows():
        duplicated_games.append(reverse_game(game))
    duplicated_games = pd.DataFrame(duplicated_games, columns=features)
    df = pd.concat([df, duplicated_games])
    return df