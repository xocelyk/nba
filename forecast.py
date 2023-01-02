import datetime
import pandas as pd
import numpy as np


def predict_margin_today_games(games, win_margin_model):
    # change date to datetime object
    games['date'] = pd.to_datetime(games['date'])
    games['date'] = games['date'].dt.date
    games = games[games['completed'] == False]
    games = games[games['date'] == datetime.date.today()]
    if len(games) == 0:
        return None
    games['last_year_team_rating*num_games_into_season'] = games['last_year_team_rating'] * games['num_games_into_season']
    games['last_year_opponent_rating*num_games_into_season'] = games['last_year_opponent_rating'] * games['num_games_into_season']
    games['margin'] = win_margin_model.predict(games[['team_rating', 'opponent_rating', 'team_win_total_future','opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating']])
    for date in games['date'].unique():
        print('{} games'.format(date))
        for index, row in games[games['date'] == date].iterrows():
            print('{} vs {}: {}'.format(row['team'], row['opponent'], round(row['margin'], 1)))
        print()
    return games

def predict_margin_this_week_games(games, win_margin_model):
    to_csv_data = []
    # change date to datetime object
    games['date'] = pd.to_datetime(games['date'])
    games['date'] = games['date'].dt.date
    games = games[games['completed'] == False]
    games = games[games['date'] >= datetime.date.today()]
    games = games[games['date'] < datetime.date.today() + datetime.timedelta(days=7)]
    if len(games) == 0:
        return None
    games['last_year_team_rating*num_games_into_season'] = games['last_year_team_rating'] * games['num_games_into_season']
    games['last_year_opponent_rating*num_games_into_season'] = games['last_year_opponent_rating'] * games['num_games_into_season']
    games['margin'] = win_margin_model.predict(games[['team_rating', 'opponent_rating', 'team_win_total_future','opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating']])
    for date in games['date'].unique():
        print('{} games'.format(date))
        for index, row in games[games['date'] == date].iterrows():
            print('{} vs {}: {}'.format(row['team'], row['opponent'], round(row['margin'], 1)))
            to_csv_data.append([row['date'], row['team'], row['opponent'], round(row['margin'], 1)])
        print()
    to_csv_data = pd.DataFrame(to_csv_data, columns=['Date', 'Home', 'Away', 'Predicted Home Margin'])
    to_csv_data.to_csv('data/predicted_margins.csv', index=False)
    return games

def predict_win_prob_future_games(all_games, clf):
    future_games = all_games[all_games['completed'] == False]
    X = future_games[['team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season']]
    y = clf.predict_proba(X)
    future_games['win_prob'] = y[:, 1]
    future_games.drop([col for col in future_games.columns if col.startswith('Unnamed')], axis=1, inplace=True)
    return future_games

def get_predictive_ratings_win_margin(model, year):

    '''
    win margin model takes these features:
    ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', \
    'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])
    '''
    filename = 'data/cumulative_with_cur_year_and_last_year_ratings_2010_2023.csv'
    this_year_games = pd.read_csv(filename)[pd.read_csv(filename)['year'] == year]
    this_year_games = this_year_games[this_year_games['completed'] == True]
    this_year_games.sort_values(by='date', ascending=False)
    num_games_into_season = len(this_year_games)
    this_year_ratings = {}
    last_year_ratings = {}
    for team in this_year_games['team'].unique():
        # get games where team was either the opponent or the team
        this_year_games_for_team = this_year_games[(this_year_games['team'] == team) | (this_year_games['opponent'] == team)]
        this_year_games_for_team.sort_values(by='date', ascending=False)
        most_recent_game = this_year_games_for_team.iloc[-1]
        if most_recent_game['team'] == team:
            this_year_ratings[team] = most_recent_game['team_rating']
            last_year_ratings[team] = most_recent_game['last_year_team_rating']
        else:
            this_year_ratings[team] = most_recent_game['opponent_rating']
            last_year_ratings[team] = most_recent_game['last_year_opponent_rating']
    
    teams = list(this_year_ratings.keys())
    team_predictive_em = {}
    for team in teams:
        team_home_margins = []
        team_away_margins = []
        team_rating = this_year_ratings[team]
        team_df = this_year_games[(this_year_games['team'] == team) | (this_year_games['opponent'] == team)]
        team_last_10_rating = team_df['team_last_10_rating'].iloc[-1] if team_df['team'].iloc[-1] == team else team_df['opponent_last_10_rating'].iloc[-1]
        team_last_5_rating = team_df['team_last_5_rating'].iloc[-1] if team_df['team'].iloc[-1] == team else team_df['opponent_last_5_rating'].iloc[-1]
        team_last_3_rating = team_df['team_last_3_rating'].iloc[-1] if team_df['team'].iloc[-1] == team else team_df['opponent_last_3_rating'].iloc[-1]
        team_last_1_rating_rating = team_df['team_last_1_rating'].iloc[-1] if team_df['team'].iloc[-1] == team else team_df['opponent_last_1_rating'].iloc[-1]
        team_win_total_future = team_df['team_win_total_future'].iloc[-1] if team_df['team'].iloc[-1] == team else team_df['opponent_win_total_future'].iloc[-1]

        for opp in teams:
            opp_rating = this_year_ratings[opp]
            opp_df = this_year_games[(this_year_games['team'] == opp) | (this_year_games['opponent'] == opp)]
            opp_last_10_rating = opp_df['team_last_10_rating'].iloc[-1] if opp_df['team'].iloc[-1] == opp else opp_df['opponent_last_10_rating'].iloc[-1]
            opp_last_5_rating = opp_df['team_last_5_rating'].iloc[-1] if opp_df['team'].iloc[-1] == opp else opp_df['opponent_last_5_rating'].iloc[-1]
            opp_last_3_rating = opp_df['team_last_3_rating'].iloc[-1] if opp_df['team'].iloc[-1] == opp else opp_df['opponent_last_3_rating'].iloc[-1]
            opp_last_1_rating_rating = opp_df['team_last_1_rating'].iloc[-1] if opp_df['team'].iloc[-1] == opp else opp_df['opponent_last_1_rating'].iloc[-1]
            opp_win_total_future = opp_df['team_win_total_future'].iloc[-1] if opp_df['team'].iloc[-1] == opp else opp_df['opponent_win_total_future'].iloc[-1]

            # play a home game
            X_home = pd.DataFrame([[team_rating, opp_rating, team_win_total_future, opp_win_total_future, last_year_ratings[team], last_year_ratings[opp], num_games_into_season, last_year_ratings[team]*num_games_into_season, last_year_ratings[opp]*num_games_into_season, team_last_10_rating, opp_last_10_rating, team_last_5_rating, opp_last_5_rating, team_last_3_rating, opp_last_3_rating, team_last_1_rating_rating, opp_last_1_rating_rating]], columns=['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])
            team_home_margins.append(model.predict(X_home)[0])

            # play an away game
            X_away = pd.DataFrame([[opp_rating, team_rating, opp_win_total_future, team_win_total_future, last_year_ratings[opp], last_year_ratings[team], num_games_into_season, last_year_ratings[opp]*num_games_into_season, last_year_ratings[team]*num_games_into_season, opp_last_10_rating, team_last_10_rating, opp_last_5_rating, team_last_5_rating, opp_last_3_rating, team_last_3_rating, opp_last_1_rating_rating, team_last_1_rating_rating]], columns=['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])
            team_away_margins.append(-model.predict(X_away)[0])

        average_home_margin = np.mean(team_home_margins)
        average_away_margin = np.mean(team_away_margins)
        team_predictive_em[team] = np.mean([average_home_margin, average_away_margin])
        print(team)
        print('expected_margin_home: {}, expected_margin_away: {}'.format(round(average_home_margin, 2), round(average_away_margin, 2)))
        print()

    mean_predictive_em = np.mean(list(team_predictive_em.values()))
    for team in teams:
        team_predictive_em[team] -= mean_predictive_em
    
    team_predictive_em_df = pd.DataFrame.from_dict(team_predictive_em, orient='index', columns=['expected_margin'])
    team_predictive_em_df = team_predictive_em_df.sort_values(by='expected_margin', ascending=False)
    team_predictive_em_df.to_csv('data/predictive_expected_margin.csv')
    return team_predictive_em_df

def get_expected_wins_losses(all_data, future_games_with_win_probs):
    all_data.set_index('team', inplace=True)
    all_data['team'] = all_data.index
    wins = all_data['wins'].to_dict()
    losses = all_data['losses'].to_dict()

    expected_wins = wins.copy()
    expected_losses = losses.copy()
    for idx, row in future_games_with_win_probs.iterrows():
        team = row['team']
        opponent = row['opponent']
        expected_wins[team] += row['win_prob']
        expected_losses[team] += 1 - row['win_prob']
        expected_wins[opponent] += 1 - row['win_prob']
        expected_losses[opponent] += row['win_prob']
    return expected_wins, expected_losses

