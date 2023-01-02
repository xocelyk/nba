import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression


# TODO: build out model to retrive this number
HCA = 3

def calc_rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())

def get_em_ratings(df, depth=1000):
    # TODO: factor in pace
    cap = .2
    try:
        seed = get_em_ratings_from_eigenratings(df, eigenrank(df))
        assert len(seed.keys()) == 30
    except:
        seed = {team: 0 for team in df['team'].unique()}
    adj_em_ratings = seed.copy()
    for team, rating in adj_em_ratings.items():
        adj_em_ratings[team] = rating / 100

    team_list = list(set(list(team for team in df['team'].unique()) + list(team for team in df['opponent'].unique())))
    for _ in range(depth):
        prev_ratings = adj_em_ratings.copy()
        diffs = {team: [] for team in team_list}
        for boxscore_id, row in df.iterrows():
            pred_home_ppp_margin = (adj_em_ratings[row['team']] - adj_em_ratings[row['opponent']])
            pred_away_ppp_margin = (adj_em_ratings[row['opponent']] - adj_em_ratings[row['team']])
            home_ppp_margin = row['margin'] / row['pace']
            if home_ppp_margin > cap:
                home_ppp_margin = cap + np.log(home_ppp_margin - cap + 1)
            elif home_ppp_margin < -cap:
                home_ppp_margin = -cap - np.log(-home_ppp_margin - cap + 1)
            away_ppp_margin = -home_ppp_margin
            home_diff = home_ppp_margin - pred_home_ppp_margin
            away_diff = away_ppp_margin - pred_away_ppp_margin
            diffs[row['team']].append(home_diff)
            diffs[row['opponent']].append(away_diff)
        for team, diff in diffs.items():
            adj_em_ratings[team] += np.mean(diff)

        # get l2 distance between ratings
        l2_dist = 0
        for team, rating in adj_em_ratings.items():
            l2_dist += (rating - prev_ratings[team]) ** 2
        l2_dist = np.sqrt(l2_dist)
        if l2_dist < 1e-6:
            break

    for team, rating in adj_em_ratings.items():
        adj_em_ratings[team] = rating * 100
    return adj_em_ratings

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
    # TODO: can duplicate games and use groupby to speed this up

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

def get_last_n_games_dict(completed_games, n):
    team_vals = {}
    completed_games = completed_games[completed_games['year'] == completed_games['year'].max()]
    completed_games = completed_games.sort_values(by='date', ascending=False)

    for team in list(set(completed_games['team'].unique().tolist() + completed_games['opponent'].unique().tolist())):
        team_data = completed_games[completed_games['team'] == team]
        # team_data = team_data.append(completed_games[completed_games['opponent'] == team])
        # rewrite the above with concat instead of append
        team_data = pd.concat([team_data, completed_games[completed_games['opponent'] == team]])
        # get the n most recent games
        team_data = team_data.iloc[:n]
        team_data = team_data.sort_values(by='date', ascending=False)
        if len(team_data) < n:
            team_val = 0
        else:
            vals = []
            for idx, row in team_data.iterrows():
                if row['team'] == team:
                    vals.append(row['team_rating'])
                else:
                    vals.append(row['opponent_rating'])
            team_val = np.mean(vals)
        team_vals[team] = team_val
    return team_vals

