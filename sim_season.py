import pandas as pd
import numpy as np
import datetime
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression
import utils
import time
import data_loader
from random import choice

# need to import adjust_em_ratings

'''

# TODO: update with days since most recent game

This is the simulation flow:
for each day:
    1. Run the get most recent data function over the completed games and impute future games with that data
        - Ratings, win totals, and last n game ratings
    2. Run the simulation over the next block of games
        - Games are blocked by day, so we simulate each day
    3. Add simulated games to completed games and remove simulated games from future games

'''

class MarginModel:
    def __init__(self, margin_model, margin_model_resid_mean, margin_model_resid_std):
        self.margin_model = margin_model
        self.resid_std = margin_model_resid_std
        self.resid_mean = margin_model_resid_mean

class Season:
    def __init__(self, year, completed_games, future_games, margin_model, mean_pace, std_pace):
        self.year = year
        self.completed_games = completed_games
        self.completed_games['winner_name'] = self.completed_games.apply(lambda row: row['team'] if row['margin'] > 0 else row['opponent'], axis=1)
        self.completed_games['team_win'] = self.completed_games.apply(lambda row: 1 if row['margin'] > 0 else 0, axis=1)
        self.future_games = future_games
        self.future_games['winner_name'] = np.nan
        self.margin_model = margin_model
        self.teams = self.teams()
        self.mean_pace = mean_pace
        self.std_pace = std_pace
        # TODO: fix this
        self.future_games['pace'] = [np.random.normal(self.mean_pace, self.std_pace) for _ in range(len(self.future_games))]
        self.completed_games['pace'] = [np.random.normal(self.mean_pace, self.std_pace) for _ in range(len(self.completed_games))]
        self.time = time.time()
        self.win_total_futures = self.get_win_total_futures()
        self.last_year_ratings = self.get_last_year_ratings()
        self.last_n_games_adj_margins = self.init_last_n_games_adj_margins()
        self.team_last_adj_margin_dict = {team: np.mean(self.last_n_games_adj_margins[team][:1]) if len(self.last_n_games_adj_margins[team]) >= 1 else 0 for team in self.teams}
        self.std_drift_dict = self.get_std_drift(['team_rating', 'team_last_10_rating', 'team_last_5_rating', 'team_last_3_rating', 'team_last_1_rating'])
        # self.drift_feature_fns = self.get_std_drift(features = ['team_rating', 'team_last_10_rating', 'team_last_5_rating', 'team_last_3_rating', 'team_last_1_rating'], new=True)
        # self.drift_fns = self.get_drift_fns(['team_rating', 'team_last_10_rating', 'team_last_5_rating', 'team_last_3_rating', 'team_last_1_rating'])
        self.last_game_stats_dict = None
        self.sim_date_increment = 3

    def get_drift_fns(self, features):
        feature_fns_dict = {}
        completed_games_copy = self.completed_games.copy()
        completed_games = utils.duplicate_games(completed_games_copy)
        completed_games['team_adj_margin'] = completed_games.apply(lambda x: x['margin'] + x['opponent_rating'] - utils.HCA, axis=1)
        for feature in features:
            # I want to find the average change in rating as a function of team_last_1_game_rating and num_games_into_season
            completed_games[feature + '_drift'] = completed_games[feature].diff()
            completed_games[feature + '_drift'] = completed_games[feature + '_drift'].fillna(0)
            completed_games[feature + '_drift'] = completed_games[feature + '_drift'].apply(lambda x: 0 if x == np.inf else x)
            feature_lm = LinearRegression()
            feature_lm.fit(completed_games[['team_last_1_rating', 'num_games_into_season']], completed_games[feature + '_drift'])
            # find std of residuals
            feature_drift_resid_std = np.std(completed_games[feature + '_drift'] - feature_lm.predict(completed_games[['team_last_1_rating', 'num_games_into_season']]))
            feature_fn = lambda x: x[feature] + feature_lm.predict([[x['team_last_1_rating'], x['num_games_into_season']]])[0] + np.random.normal(0, feature_drift_resid_std)
            feature_fns_dict[feature] = feature_fn
        return feature_fns_dict
        
    

    def init_last_n_games_adj_margins(self):
        # earliest games first, most recent games last
        completed_games = self.completed_games.copy()
        res = {}
        completed_games.sort_values(by='date', ascending=True, inplace=True)
        for team in list(set(completed_games['team'].unique().tolist() + completed_games['opponent'].unique().tolist())):
            team_data = completed_games[(completed_games['team'] == team) | (completed_games['opponent'] == team)].sort_values(by='date', ascending=True)
            team_data = utils.duplicate_games(team_data)
            team_data = team_data[team_data['team'] == team]
            team_vals = []
            team_data['team_adj_margin'] = team_data.apply(lambda x: x['margin'] + x['opponent_rating'] - utils.HCA, axis=1)
            if len(team_data) == 0:
                team_adj_margins = []
            else:
                team_adj_margins = team_data['team_adj_margin'].tolist()
            res[team] = team_adj_margins
        return res
    
    def get_std_drift(self, features, new=False):
        '''
        assuming autocorrelation is normally distributed
        '''
        if not new:
            team_drifts = {feature: [] for feature in features}
            for team in self.teams:
                team_data = utils.duplicate_games(self.completed_games)
                team_data = team_data.loc[team_data['team'] == team]
                team_data.sort_values(by='date', inplace=True, ascending=True)
                for feature in features:
                    team_data[feature + '_drift'] = team_data[feature].diff()
                    team_drifts[feature].append(team_data[feature + '_drift'].mean())
            team_drifts = {feature: np.std(team_drifts[feature]) for feature in features}
            for feature in features:
                if feature.startswith('team'):
                    opponent_feature = feature.replace('team', 'opponent')
                    team_drifts[opponent_feature] = team_drifts[feature]
            
            return team_drifts
        
        else:
            # TODO: hold some out
            drift_fns = {}
            train_data = self.completed_games.copy()
            X = train_data['team_last_1_rating']
            X = X.values.reshape(-1, 1)
            for feature in features:
                y = train_data[feature].diff(1).fillna(0)
                model = LinearRegression().fit(X, y)
                pred_y = model.predict(X)
                resid = pred_y - y
                std_resid = np.std(resid)

                gen_feature_fn = lambda x: model.predict([[x]])[0] + np.random.normal(0, std_resid)
                drift_fns[feature] = gen_feature_fn
        return drift_fns


    def get_random_pace(self):
        return np.random.normal(self.mean_pace, self.std_pace)

    def get_win_total_futures(self):
        win_total_futures = {}
        all_games = pd.concat([self.completed_games, self.future_games])
        # get dict of team to win total futures
        win_total_futures = {}
        for team in self.teams:
            team_win_total_futures = all_games.loc[all_games['team'] == team, 'team_win_total_future'].iloc[0]
            win_total_futures[team] = team_win_total_futures
        return win_total_futures

    def get_last_year_ratings(self):
        """Create a dict of team to last year's team rating."""
        last_year_ratings = {}
        # Concatenate completed games with future games
        all_games = pd.concat([self.completed_games, self.future_games])
        # get dict of team to win total futures
        for team in self.teams:
            team_last_year_ratings = all_games.loc[all_games['team'] == team, 'last_year_team_rating'].iloc[0]
            last_year_ratings[team] = team_last_year_ratings
        return last_year_ratings

    def teams(self):
        return sorted(list(set(self.completed_games['team'].unique().tolist() + self.future_games['team'].unique().tolist())))

    def simulate_season(self, date_increment=1):
        '''
        TODO:
        - make sure games completed and games future are imputed with all the correct features beforehand
        '''
        date_increment = self.sim_date_increment
        min_date = self.future_games['date'].min()
        max_date = self.future_games['date'].max()
        daterange = [min_date]
        while daterange[-1] <= max_date:
            daterange.append(daterange[-1] + datetime.timedelta(days=1))
        
        for date in daterange[::date_increment]:
            start_date = date
            end_date = date + datetime.timedelta(days=date_increment)
            self.simulate_day(start_date, end_date, date_increment)

    def get_team_last_games(self):
        teams_last_games_dict = {}
        for team in self.teams:
            team_last_games = utils.duplicate_games(self.completed_games)
            team_last_games = team_last_games.loc[team_last_games['team'] == team]
            team_last_games.sort_values(by='date', ascending=False, inplace=True)
            team_last_game = team_last_games.iloc[0]
            teams_last_games_dict[team] = team_last_game
        return teams_last_games_dict

    def update_data(self, drift=False, games_on_date=None):
        '''
        after playing a series of games (e.g. a day), update the ratings for each team
        '''
        if self.future_games.empty:
            return
        if games_on_date is None:
            # teams = self.teams
            games_on_date = self.future_games[self.future_games['completed'] == True]
        # else:
        teams = list(set(games_on_date['team'].unique().tolist() + games_on_date['opponent'].unique().tolist()))
        team_freq_dict = games_on_date['team'].value_counts().to_dict()
        for team in self.teams:
            team_freq_dict[team] = team_freq_dict.get(team, 0)
        
        # PROBLEM with drift: drift in features are treated as independent, but they are not
        # this reduces variance in team projected records
        # get all drift features as a function of adjusted game margin and num games into season + random variance
        if drift:
            # # TODO: broken
            # for idx, row in games_on_date.sort_values(by='date', ascending=True).iterrows():
            #     team_adj_margin = row['margin'] + row['opponent_rating'] - utils.HCA
            #     opponent_adj_margin = -row['margin'] + row['team_rating'] + utils.HCA
            #     self.team_last_adj_margin_dict[row['team']] = team_adj_margin
            #     self.team_last_adj_margin_dict[row['opponent']] = opponent_adj_margin
            
            # for feature , feature_fn in self.drift_fns.items():
            #     for team in teams:
            #         self.future_games['team_last_1_rating'] = self.future_games.apply(lambda row:
            #          self.team_last_adj_margin_dict.get(row['team'], 0), axis=1)
            #         self.future_games.apply(lambda row: team_freq_dict[row['team']] * feature_fn(row), axis=1)


            drift_features = self.std_drift_dict.keys()
            drift_feature_fns_dict = self.drift_feature_fns
            if not self.last_game_stats_dict:
                last_game_stats_dict = self.get_team_last_games()
                self.last_game_stats_dict = last_game_stats_dict
            else:
                team_last_game_scores = {}
                last_game_stats_dict = self.last_game_stats_dict
                for team in teams:
                    team_data = games_on_date[(games_on_date['team'] == team) | (games_on_date['opponent'] == team)]
                    team_data = utils.duplicate_games(team_data)
                    if len(team_data) > 0:
                        team_last_game_scores[team] = np.mean(team_data['margin'] - utils.HCA + team_data['opponent_rating'])

                    else:
                        team_last_game_scores[team] = 0
                    print()
                    print(team)
                    print('mean last game scores:', team_last_game_scores[team])
                    for feature in drift_features:
                        if feature.startswith('opponent'):
                            team_feature = feature.replace('opponent', 'team')
                            if team_feature in last_game_stats_dict[team].keys():
                                last_game_stats_dict[team][feature] = last_game_stats_dict[team][team_feature]
                            else:
                                last_game_stats_dict[team][feature] = last_game_stats_dict[team][team_feature] + drift_feature_fns_dict[team_feature](team_last_game_scores[team])
                        else:
                            last_game_stats_dict[team][feature] = last_game_stats_dict[team][feature] + drift_feature_fns_dict[feature](team_last_game_scores[team])
                        print(feature, ':', last_game_stats_dict[team][feature])

                    # for feature in drift_features:
                    #     last_game_stats_dict[team][feature] = last_game_stats_dict[team][feature] + team_freq_dict[team] * np.random.normal(0, self.std_drift_dict[feature])
                    #     self.last_game_stats_dict = last_game_stats_dict
                for feature in drift_features:
                    # if X has stdev sigma, then c * X has stdev c * sigma
                    if feature.startswith('team'):
                        self.future_games[feature] = self.future_games.apply(lambda row: last_game_stats_dict[row['team']][feature], axis=1)
                    else:
                        self.future_games[feature] = self.future_games.apply(lambda row: last_game_stats_dict[row['opponent']][feature], axis=1)


        else:
            # this should be an object attribute that is updated with each simulate_game
            # last_n_games_dict = utils.get_last_n_games_dict(self.completed_games, [10, 5, 3, 1])
            # last_10_games_dict, last_5_games_dict, last_3_games_dict, last_1_games_dict = last_n_games_dict[10], last_n_games_dict[5], last_n_games_dict[3], last_n_games_dict[1]

            last_10_games_dict = {team: np.mean(self.last_n_games_adj_margins[team][:10]) if len(self.last_n_games_adj_margins[team]) >= 10 else 0 for team in self.teams}
            last_5_games_dict = {team: np.mean(self.last_n_games_adj_margins[team][:5]) if len(self.last_n_games_adj_margins[team]) >= 5 else 0 for team in self.teams}
            last_3_games_dict = {team: np.mean(self.last_n_games_adj_margins[team][:3]) if len(self.last_n_games_adj_margins[team]) >= 3 else 0 for team in self.teams}
            last_1_games_dict = {team: np.mean(self.last_n_games_adj_margins[team][:1]) if len(self.last_n_games_adj_margins[team]) >= 1 else 0 for team in self.teams}

            self.future_games['team_last_10_rating'] = self.future_games['team'].map(last_10_games_dict)
            self.future_games['opponent_last_10_rating'] = self.future_games['opponent'].map(last_10_games_dict)

            self.future_games['team_last_5_rating'] = self.future_games['team'].map(last_5_games_dict)
            self.future_games['opponent_last_5_rating'] = self.future_games['opponent'].map(last_5_games_dict)

            self.future_games['team_last_3_rating'] = self.future_games['team'].map(last_3_games_dict)
            self.future_games['opponent_last_3_rating'] = self.future_games['opponent'].map(last_3_games_dict)

            self.future_games['team_last_1_rating'] = self.future_games['team'].map(last_1_games_dict)
            self.future_games['opponent_last_1_rating'] = self.future_games['opponent'].map(last_1_games_dict)

            em_ratings = utils.get_em_ratings(self.completed_games)

            self.future_games['team_rating'] = self.future_games['team'].map(em_ratings)
            self.future_games['opponent_rating'] = self.future_games['opponent'].map(em_ratings)

        if self.future_games['last_year_team_rating'].isnull().any():
            self.future_games['last_year_team_rating'] = self.future_games['team'].map(self.last_year_ratings)
            self.future_games['last_year_opponent_rating'] = self.future_games['opponent'].map(self.last_year_ratings)

        if self.future_games['team_win_total_future'].isnull().any():
            self.future_games['team_win_total_future'] = self.future_games['team'].map(self.win_total_futures)
            self.future_games['opponent_win_total_future'] = self.future_games['opponent'].map(self.win_total_futures)

        if self.future_games['num_games_into_season'].isnull().any():
            # this only works for playoffs
            self.future_games['num_games_into_season'].fillna(len(self.completed_games), inplace=True)

        if self.future_games['pace'].isnull().any():
            self.future_games['pace'] = [self.get_random_pace() for _ in range(len(self.future_games))]

        

    def simulate_day(self, start_date, end_date, date_increment=1):
        games_on_date = self.future_games[(self.future_games['date'] < end_date) & (self.future_games['date'] >= start_date)]
        if games_on_date.empty:
            return
        games_on_date = games_on_date.apply(self.simulate_game, axis=1)
        self.completed_games = self.completed_games.append(games_on_date)
        self.future_games = self.future_games[~self.future_games.index.isin(self.completed_games.index)]
        if self.future_games.empty:
            return
        self.update_data(games_on_date=games_on_date)


    def simulate_game(self, row):
        team = row['team']
        opponent = row['opponent']
        train_data = self.get_game_data(row)
        expected_margin = self.margin_model.margin_model.predict(train_data)[0]
        margin = np.random.normal(0, self.margin_model.resid_std) + self.margin_model.resid_mean + expected_margin
        team_win = int(margin > 0)
        row['completed'] = True
        row['team_win'] = team_win
        row['margin'] = margin
        row['winner_name'] = team if team_win else opponent
        team_adj_margin = row['margin'] + row['opponent_rating'] - utils.HCA
        opponent_adj_margin = -row['margin'] + row['team_rating'] + utils.HCA
        self.last_n_games_adj_margins[team].append(team_adj_margin)
        self.last_n_games_adj_margins[opponent].append(opponent_adj_margin)
        return row

    def get_game_data(self, row):
        team_rating = row['team_rating']
        opp_rating = row['opponent_rating']
        last_year_team_rating = row['last_year_team_rating']
        last_year_opp_rating = row['last_year_opponent_rating']
        num_games_into_season = row['num_games_into_season']
        team_last_10_rating = row['team_last_10_rating']
        opponent_last_10_rating = row['opponent_last_10_rating']
        team_last_5_rating = row['team_last_5_rating']
        opponent_last_5_rating = row['opponent_last_5_rating']
        team_last_3_rating = row['team_last_3_rating']
        opponent_last_3_rating = row['opponent_last_3_rating']
        team_last_1_rating = row['team_last_1_rating']
        opponent_last_1_rating = row['opponent_last_1_rating']
        team_win_total_future = row['team_win_total_future']
        opponent_win_total_future = row['opponent_win_total_future']

        data = pd.DataFrame([[team_rating, opp_rating, team_win_total_future, opponent_win_total_future, last_year_team_rating, last_year_opp_rating, num_games_into_season, team_last_10_rating, opponent_last_10_rating, team_last_5_rating, opponent_last_5_rating, team_last_3_rating, opponent_last_3_rating, team_last_1_rating, opponent_last_1_rating]], columns=['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])
        return data

    def get_win_loss_report(self):
        record_by_team = {team: [0, 0] for team in self.teams}
        for idx, game in self.completed_games.iterrows():
            if game['team_win']:
                record_by_team[game['team']][0] += 1
                record_by_team[game['opponent']][1] += 1
            else:
                record_by_team[game['team']][1] += 1
                record_by_team[game['opponent']][0] += 1
        return record_by_team

    def playoffs(self):
        playoff_results = {'playoffs': [], 'second_round': [], 'conference_finals': [], 'finals': [], 'champion': []}

        ec_standings, wc_standings = self.get_playoff_standings(self.get_win_loss_report())

        rank = 1
        print('Eastern Conference Standings')
        for idx, row in ec_standings.iterrows():
            print(f'{rank}. {row["team"]} ({row["wins"]}-{row["losses"]})')
            rank += 1
        print()
        rank = 1
        print('Western Conference Standings')
        for idx, row in wc_standings.iterrows():
            print(f'{rank}. {row["team"]} ({row["wins"]}-{row["losses"]})')
            rank += 1
    
        self.future_games['playoff_label'] = None
        self.future_games['winner_name'] = None

        self.completed_games['playoff_label'] = None
        self.completed_games['winner_name'] = None

        east_seeds, west_seeds = self.play_in(ec_standings, wc_standings)
        self.seeds = {}
        for seed, team in east_seeds.items():
            self.seeds[team] = seed
        for seed, team in west_seeds.items():
            self.seeds[team] = seed

        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values()) 
        assert len(set(east_alive).intersection(set(west_alive))) == 0
        assert len(set(west_alive + east_alive)) == len(west_alive + east_alive)
        playoff_results['playoffs'] = east_alive + west_alive
        
        for seed, team in east_seeds.items():
            print('{}. {} (E)'.format(seed, team))
        print()
        for seed, team in west_seeds.items():
            print('{}. {} (W)'.format(seed, team))
        print()

        # simulate first round
        east_seeds, west_seeds = self.first_round(east_seeds, west_seeds)
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results['second_round'] = east_alive + west_alive

        # simulate second round
        east_seeds, west_seeds = self.second_round(east_seeds, west_seeds)
        east_alive = list(east_seeds.values())
        west_alive = list(west_seeds.values())
        playoff_results['conference_finals'] = east_alive + west_alive

        # simulate conference finals
        e1, w1 = self.conference_finals(east_seeds, west_seeds)
        playoff_results['finals'] = [e1, w1]

        if choice([True, False]):
            team1 = e1
            team2 = w1
        else:
            team1 = w1
            team2 = e1
        
        # simulate finals
        champ = self.finals(team1, team2)
        finals = [team1, team2]
        playoff_results['champion'] = [champ]
        print('Champion: ' + champ)
        return playoff_results


    def first_round(self, east_seeds, west_seeds):

        game_1_date = self.get_next_date()
        game_2_date = game_1_date + datetime.timedelta(days=1)
        game_3_date = game_2_date + datetime.timedelta(days=1)
        game_4_date = game_3_date + datetime.timedelta(days=1)
        game_5_date = game_4_date + datetime.timedelta(days=1)
        game_6_date = game_5_date + datetime.timedelta(days=1)
        game_7_date = game_6_date + datetime.timedelta(days=1)

        matchups = {'E_1_8': [east_seeds[1], east_seeds[8]],
                    'E_4_5': [east_seeds[4], east_seeds[5]],
                    'E_2_7': [east_seeds[2], east_seeds[7]],
                    'E_3_6': [east_seeds[3], east_seeds[6]],
                    'W_1_8': [west_seeds[1], west_seeds[8]],
                    'W_4_5': [west_seeds[4], west_seeds[5]],
                    'W_2_7': [west_seeds[2], west_seeds[7]],
                    'W_3_6': [west_seeds[3], west_seeds[6]]}
        
        for label, [team1, team2] in matchups.items():
            self.append_future_game(self.future_games, game_1_date, team1, team2, label)
            self.append_future_game(self.future_games, game_2_date, team1, team2, label)
            self.append_future_game(self.future_games, game_3_date, team2, team1, label)
            self.append_future_game(self.future_games, game_4_date, team2, team1, label)
            self.append_future_game(self.future_games, game_5_date, team1, team2, label)
            self.append_future_game(self.future_games, game_6_date, team2, team1, label)
            self.append_future_game(self.future_games, game_7_date, team1, team2, label)
        self.update_data(games_on_date=self.future_games[:-56])
        
        for date in sorted([game_1_date, game_2_date, game_3_date, game_4_date, game_5_date, game_6_date, game_7_date]):
            self.simulate_day(date, date + datetime.timedelta(days=1), 1)

        e1 = self.get_series_winner('E_1_8')
        e2 = self.get_series_winner('E_2_7')
        e3 = self.get_series_winner('E_3_6')
        e4 = self.get_series_winner('E_4_5')

        w1 = self.get_series_winner('W_1_8')
        w2 = self.get_series_winner('W_2_7')
        w3 = self.get_series_winner('W_3_6')
        w4 = self.get_series_winner('W_4_5')

        if self.seeds[e1] > self.seeds[e4]:
            e1, e4 = e4, e1
        if self.seeds[e2] > self.seeds[e3]:
            e2, e3 = e3, e2
        
        if self.seeds[w1] > self.seeds[w4]:
            w1, w4 = w4, w1
        if self.seeds[w2] > self.seeds[w3]:
            w2, w3 = w3, w2

        east_seeds = {1: e1, 2: e2, 3: e3, 4: e4}
        west_seeds = {1: w1, 2: w2, 3: w3, 4: w4}

        return east_seeds, west_seeds

    def second_round(self, east_seeds, west_seeds):
        game_1_date = self.get_next_date()
        game_2_date = game_1_date + datetime.timedelta(days=1)
        game_3_date = game_2_date + datetime.timedelta(days=1)
        game_4_date = game_3_date + datetime.timedelta(days=1)
        game_5_date = game_4_date + datetime.timedelta(days=1)
        game_6_date = game_5_date + datetime.timedelta(days=1)
        game_7_date = game_6_date + datetime.timedelta(days=1)

        matchups = {'E_1_4': (east_seeds[1], east_seeds[4]),
                    'E_2_3': (east_seeds[2], east_seeds[3]),
                    'W_1_4': (west_seeds[1], west_seeds[4]),
                    'W_2_3': (west_seeds[2], west_seeds[3])}
        
        for label, (team1, team2) in matchups.items():
            self.append_future_game(self.future_games, game_1_date, team1, team2, label)
            self.append_future_game(self.future_games, game_2_date, team1, team2, label)
            self.append_future_game(self.future_games, game_3_date, team2, team1, label)
            self.append_future_game(self.future_games, game_4_date, team2, team1, label)
            self.append_future_game(self.future_games, game_5_date, team1, team2, label)
            self.append_future_game(self.future_games, game_6_date, team2, team1, label)
            self.append_future_game(self.future_games, game_7_date, team1, team2, label)
        self.update_data(games_on_date=self.future_games[:-28])
        
        for date in sorted([game_1_date, game_2_date, game_3_date, game_4_date, game_5_date, game_6_date, game_7_date]):
            self.simulate_day(date, date + datetime.timedelta(days=1), 1)

        e_1 = self.get_series_winner('E_1_4')
        e_2 = self.get_series_winner('E_2_3')
        w_1 = self.get_series_winner('W_1_4')
        w_2 = self.get_series_winner('W_2_3')

    # TODO: fix this for other rounds too
        if self.seeds[e_1] > self.seeds[e_2]:
            e_1, e_2 = e_2, e_1
        if self.seeds[w_1] > self.seeds[w_2]:
            w_1, w_2 = w_2, w_1

        east_seeds = {1: e_1, 2: e_2}
        west_seeds = {1: w_1, 2: w_2}

        return east_seeds, west_seeds

    
    def conference_finals(self, east_seeds, west_seeds):
        game_1_date = self.get_next_date()
        game_2_date = game_1_date + datetime.timedelta(days=1)
        game_3_date = game_2_date + datetime.timedelta(days=1)
        game_4_date = game_3_date + datetime.timedelta(days=1)
        game_5_date = game_4_date + datetime.timedelta(days=1)
        game_6_date = game_5_date + datetime.timedelta(days=1)
        game_7_date = game_6_date + datetime.timedelta(days=1)

        matchups = {'E_1_2': [east_seeds[1], east_seeds[2]],
                    'W_1_2': [west_seeds[1], west_seeds[2]]}
        
        for label, (team1, team2) in matchups.items():
            self.append_future_game(self.future_games, game_1_date, team1, team2, label)
            self.append_future_game(self.future_games, game_2_date, team1, team2, label)
            self.append_future_game(self.future_games, game_3_date, team2, team1, label)
            self.append_future_game(self.future_games, game_4_date, team2, team1, label)
            self.append_future_game(self.future_games, game_5_date, team1, team2, label)
            self.append_future_game(self.future_games, game_6_date, team2, team1, label)
            self.append_future_game(self.future_games, game_7_date, team1, team2, label)
        self.update_data(games_on_date=self.future_games[:-14])
        
        for date in sorted([game_1_date, game_2_date, game_3_date, game_4_date, game_5_date, game_6_date, game_7_date]):
            self.simulate_day(date, date + datetime.timedelta(days=1), 1)
        
        e_1 = self.get_series_winner('E_1_2')
        w_1 = self.get_series_winner('W_1_2')

        return e_1, w_1

    def finals(self, e_1, w_1):
        game_1_date = self.get_next_date()
        game_2_date = game_1_date + datetime.timedelta(days=1)
        game_3_date = game_2_date + datetime.timedelta(days=1)
        game_4_date = game_3_date + datetime.timedelta(days=1)
        game_5_date = game_4_date + datetime.timedelta(days=1)
        game_6_date = game_5_date + datetime.timedelta(days=1)
        game_7_date = game_6_date + datetime.timedelta(days=1)

        matchups = {'E_1_W_1': [e_1, w_1]}
        
        for label, (team1, team2) in matchups.items():
            self.append_future_game(self.future_games, game_1_date, team1, team2, label)
            self.append_future_game(self.future_games, game_2_date, team1, team2, label)
            self.append_future_game(self.future_games, game_3_date, team2, team1, label)
            self.append_future_game(self.future_games, game_4_date, team2, team1, label)
            self.append_future_game(self.future_games, game_5_date, team1, team2, label)
            self.append_future_game(self.future_games, game_6_date, team2, team1, label)
            self.append_future_game(self.future_games, game_7_date, team1, team2, label)
        self.update_data(games_on_date=self.future_games[:-7])
        
        for date in sorted([game_1_date, game_2_date, game_3_date, game_4_date, game_5_date, game_6_date, game_7_date]):
            self.simulate_day(date, date + datetime.timedelta(days=1), 1)
        
        winner = self.get_series_winner('E_1_W_1')

        return winner
    
    def get_series_winner(self, series_label):
        series = self.completed_games[self.completed_games['playoff_label'] == series_label]
        assert len(series) == 7
        value_counts = series['winner_name'].value_counts().sort_values(ascending=False)
        # get the team with the most wins
        teams = series.iloc[0][['team', 'opponent']].values.tolist()
        winner = value_counts.index[0]
        print('{}. {} vs {}. {} ({}. {})'.format(self.seeds[teams[0]], teams[0], self.seeds[teams[1]], teams[1], self.seeds[winner], winner))
        return winner
            
            
    def play_in(self, ec_standings, wc_standings):
    
        [e_1, e_2, e_3, e_4, e_5, e_6, e_7, e_8, e_9, e_10] = ec_standings['team'].values.tolist()[:10]
        [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9, w_10] = wc_standings['team'].values.tolist()[:10]


        # simulate play in round 1
        playin_round_1_date = self.get_next_date()
        self.append_future_game(self.future_games, date=playin_round_1_date, team=e_7, opponent=e_8, playoff_label='E_P_1')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=e_9, opponent=e_10, playoff_label='E_P_2')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=w_7, opponent=w_8, playoff_label='W_P_1')
        self.append_future_game(self.future_games, date=playin_round_1_date, team=w_9, opponent=w_10, playoff_label='W_P_2')
        self.update_data(games_on_date=self.future_games[:-4])
        self.simulate_day(playin_round_1_date, playin_round_1_date + datetime.timedelta(days=1), 1)

        assert len(self.future_games) == 0, 'future games not empty'

        # east 7 seed
        E_P_1_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['winner_name'].values[0]
        # play in round 2
        E_P_1_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['opponent'].values[0] if E_P_1_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_1']['team'].values[0]
        
        # play in round 2
        E_P_2_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['winner_name'].values[0]
        # eliminated
        E_P_2_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['opponent'].values[0] if E_P_2_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_2']['team'].values[0]

        # west 7 seed
        W_P_1_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['winner_name'].values[0]
        # play in round 2
        W_P_1_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['opponent'].values[0] if W_P_1_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_1']['team'].values[0]

        # play in round 2
        W_P_2_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['winner_name'].values[0]
        # eliminated
        W_P_2_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['opponent'].values[0] if W_P_2_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_2']['team'].values[0]

        # simulate playin round 2
        playin_round_2_date = self.get_next_date()
        self.append_future_game(self.future_games, date=playin_round_2_date, team=E_P_1_loser, opponent=E_P_2_winner, playoff_label='E_P_3')
        self.append_future_game(self.future_games, playin_round_2_date, W_P_1_loser, W_P_2_winner, 'W_P_3')
        self.update_data(games_on_date=self.future_games[:-2])
        self.simulate_day(playin_round_2_date, playin_round_2_date + datetime.timedelta(days=1), 1)

        # east 8 seed
        E_P_3_winner = self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['winner_name'].values[0]
        E_P_3_loser = self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['opponent'].values[0] if E_P_3_winner == self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'E_P_3']['team'].values[0]

        # west 8 seed
        W_P_3_winner = self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['winner_name'].values[0]
        W_P_3_loser = self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['opponent'].values[0] if W_P_3_winner == self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['team'].values[0] else self.completed_games[self.completed_games['playoff_label'] == 'W_P_3']['team'].values[0]

        self.completed_games = self.completed_games.append(self.future_games, ignore_index=True)
        self.future_games = self.future_games.iloc[0:0]

        ec_seeds = {1: e_1, 2: e_2, 3: e_3, 4: e_4, 5: e_5, 6: e_6, 7: E_P_1_winner, 8: E_P_3_winner}
        wc_seeds = {1: w_1, 2: w_2, 3: w_3, 4: w_4, 5: w_5, 6: w_6, 7: W_P_1_winner, 8: W_P_3_winner}

        return ec_seeds, wc_seeds

    def append_future_game(self, future_games, date, team, opponent, playoff_label=None):
        self.future_games = future_games.append({'date': date, 'team': team, 'opponent': opponent, 'year': self.year, 'playoff_label': playoff_label}, ignore_index=True)
       # new index
       # TODO: this is a hack, fix it
        self.completed_games.index = range(len(self.completed_games))
        self.future_games.index = range(max(self.completed_games.index) + 1, max(self.completed_games.index) + len(self.future_games) + 1)
        


    def get_next_date(self):
        return self.future_games['date'].min() if len(self.future_games) > 0 else self.completed_games['date'].max() + datetime.timedelta(days=1)


    
    def get_playoff_standings(self, record_by_team):
        '''
        takes the end of season results and returns the playoff seeding

        seeding is determined by the following, in order of priority:
        1. number of wins
        2. head to head record
        3. division leader
        4. conference record
        5. record against conference eligible playoff teams
        '''

        eastern_conference = ['BOS', 'TOR', 'PHI', 'BRK', 'IND', 'MIL', 'DET', 'CHI', 'ORL', 'WAS', 'CHO', 'NYK', 'ATL', 'MIA', 'CLE']
        western_conference = ['GSW', 'HOU', 'LAC', 'UTA', 'POR', 'OKC', 'DEN', 'SAS', 'MIN', 'NOP', 'MEM', 'SAC', 'LAL', 'DAL', 'PHO']

        # create dataframes for each conference
        ec_df = pd.DataFrame.from_dict(record_by_team, orient='index', columns=['wins', 'losses'])
        ec_df = ec_df[ec_df.index.isin(eastern_conference)]
        wc_df = pd.DataFrame.from_dict(record_by_team, orient='index', columns=['wins', 'losses'])
        wc_df = wc_df[wc_df.index.isin(western_conference)]
        ec_df['team'] = ec_df.index
        wc_df['team'] = wc_df.index


        # first, sort by wins
        ec_df['new_wins'] = ec_df['wins'] + np.random.normal(0, 0.01, len(ec_df))
        wc_df['new_wins'] = wc_df['wins'] + np.random.normal(0, 0.01, len(wc_df))
        ec_df.sort_values(by='new_wins', ascending=False, inplace=True)
        wc_df.sort_values(by='new_wins', ascending=False, inplace=True)


        # then, sort by head to head
        # ec_df = self.sort_by_head_to_head(ec_df)
        # wc_df = self.sort_by_head_to_head(wc_df)

        # # then, sort by division leader
        # ec_df = self.sort_by_division_leader(ec_df)
        # wc_df = self.sort_by_division_leader(wc_df)

        # # then, sort by conference record
        # ec_df = self.sort_by_conference_record(ec_df)
        # wc_df = self.sort_by_conference_record(wc_df)

        # # then, sort by conference eligible record
        # ec_df = self.sort_by_conference_eligible_record(ec_df)
        # wc_df = self.sort_by_conference_eligible_record(wc_df)

        # # resort by order of priorty
        # ec_df.sort_values(by=['wins', 'head_to_head', 'division_leader', 'conference_record', 'conference_eligible_record'], ascending=False, inplace=True)
        # wc_df.sort_values(by=['wins', 'head_to_head', 'division_leader', 'conference_record', 'conference_eligible_record'], ascending=False, inplace=True)

        ec_df['seed'] = [i + 1 for i in range(len(ec_df))]
        wc_df['seed'] = [i + 1 for i in range(len(wc_df))]
        ec_df.drop('new_wins', axis=1, inplace=True)
        wc_df.drop('new_wins', axis=1, inplace=True)

        return ec_df, wc_df

    def sort_by_head_to_head(self, df):
        '''
        sorts the dataframe by head to head record
        '''
        df['head_to_head'] = [0 for _ in range(len(df))]
        # get the head to head records
        prev_num_wins = None
        prev_teams = None
        for team, row in df.iterrows():
            num_wins = row['wins']
            if num_wins == prev_num_wins:
                prev_teams.append(team)
            else:
                if prev_teams:
                    # sort the teams by head to head record
                    df = self.sort_teams_by_head_to_head(df)
                prev_num_wins = num_wins
                prev_teams = [team]

        return df

    def sort_teams_by_head_to_head(self, df):
        '''
        sorts the teams by head to head record
        '''
        teams = df['team'].tolist()
        # dictionary with the head to head record for each team
        prev_head_to_head = {}
        for team in teams:
            prev_head_to_head[team] = df.loc[team, 'head_to_head']

        # get the head to head records
        head_to_head = {}
        for team in teams:
            head_to_head[team] = 0

        sim_completed_games = utils.duplicate_games(self.completed_games)

        # can do this quicker with apply then sum
        for idx, game in sim_completed_games.iterrows():
            if game['team'] in teams and game['opponent'] in teams:
                if game['team_win']:
                    head_to_head[game['team']] += 1
                else:
                    head_to_head[game['opponent']] += 1
            
        for team in teams:
            df.loc[team, 'head_to_head'] = max(head_to_head[team], prev_head_to_head[team])

        # sort the teams by head to head record
        df.sort_values(by='head_to_head', ascending=False, inplace=True)

        return df

    def sort_by_division_leader(self, df):
        '''
        sorts the dataframe by division leader
        '''
        df['division_leader'] = [0 for _ in range(len(df))]
        # get the division leaders
        divisions = {'Atlantic': ['BOS', 'TOR', 'BRK', 'PHI', 'NYK'], 'Central': ['MIL', 'IND', 'CHI', 'DET', 'CLE'], 'Southeast': ['MIA', 'ORL', 'CHO', 'WAS', 'ATL'], 'Northwest': ['DEN', 'UTA', 'POR', 'OKC', 'MIN'], 'Pacific': ['LAL', 'LAC', 'PHO', 'SAC', 'GSW'], 'Southwest': ['HOU', 'DAL', 'MEM', 'NOP', 'SAS']}
        for division, teams in divisions.items():
            # check if teams in division are in the dataframe
            if not set(teams).issubset(set(df.index)):
                continue
            
            # get the division leader
            division_df = df.loc[teams]
            max_division_wins = division_df['wins'].max()
            division_leaders = division_df[division_df['wins'] == max_division_wins].index
            for team in division_leaders:
                df.loc[team, 'division_leader'] = 1

        return df

    def sort_by_conference_record(self, conf_df):
        '''
        sorts the dataframe by conference record
        '''
        conf_df['conference_record'] = [0 for _ in range(len(conf_df))]
        # get the conference records
        for idx, game in self.completed_games.iterrows():
            if game['team'] in conf_df.index and game['opponent'] in conf_df.index:
                if game['team_win']:
                    conf_df.loc[game['team'], 'conference_record'] += 1
                else:
                    conf_df.loc[game['opponent'], 'conference_record'] += 1
        
        # sort the teams by conference record
        conf_df.sort_values(by='conference_record', ascending=False, inplace=True)

        return conf_df

    def sort_by_conference_eligible_record(self, conf_df):
        # take the top ten teams in the conference in terms of number of wins
        
        ten_best_win_counts = sorted(conf_df['wins'].unique(), reverse=True)[:10]
        ten_best_teams = [team for team in conf_df.index if conf_df.loc[team, 'wins'] in ten_best_win_counts]
        ten_best_df = conf_df.loc[ten_best_teams]

        ten_best_df['conference_eligible_record'] = [0 for _ in range(len(ten_best_df))]
        for idx, game in self.completed_games.iterrows():
            if game['team'] in ten_best_df.index and game['opponent'] in ten_best_df.index:
                if game['team_win']:
                    ten_best_df.loc[game['team'], 'conference_eligible_record'] += 1
                else:
                    ten_best_df.loc[game['opponent'], 'conference_eligible_record'] += 1
        
        # sort the teams by conference record
        ten_best_df.sort_values(by='conference_eligible_record', ascending=False, inplace=True)

        return ten_best_df

