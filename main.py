import pandas as pd
import numpy as np
import warnings
import data_loader
import utils
import datetime
import eval
import forecast
import stats
import time

'''
TODO list
 - make em ratings independent of eigenrankings - be done with eigenrankings (X)
 - make sims faster? (X)
 - try to improve win margin model
 - use sim season to predict playoff position
    - sim playoffs on top of that
 - add an "injury" feature - random chance of injury or other effect that hurts one team and bumps the others
 - add a recency bias (X)
 - fix HCA calculation
'''

# ignore warnings
warnings.filterwarnings('ignore')

def sim_season_experimental(historic_games_with_ratings, win_margin_model, margin_model_resid_mean, margin_model_resid_std, mean_pace, std_pace, year):
    from sim_season import Season, MarginModel

    margin_model = MarginModel(win_margin_model, margin_model_resid_mean, margin_model_resid_std)
    year_games = historic_games_with_ratings[historic_games_with_ratings['year'] == year]
    completed_year_games = year_games[year_games['completed'] == True]
    future_year_games = year_games[year_games['completed'] == False]
    season = Season(2023, completed_year_games, future_year_games, margin_model)
    season.simulate_season()
    playoff_results = season.playoffs()
    print(playoff_results)

def sim_season(historic_games_with_ratings, win_margin_model, margin_model_resid_mean, margin_model_resid_std, mean_pace, std_pace, year):
    historic_games_with_ratings['pace'] = np.random.normal(mean_pace, std_pace, len(historic_games_with_ratings))

    # sim_season_experimental(historic_games_with_ratings, win_margin_model, margin_model_resid_mean, margin_model_resid_std, mean_pace, std_pace, year)

    def sim_game(row):
        team = row['team']
        opponent = row['opponent']
        team_rating = row['team_rating']
        opp_rating = row['opponent_rating']
        last_year_team_rating = row['last_year_team_rating']
        last_year_opp_rating = row['last_year_opponent_rating']
        num_games_into_season = row['num_games_into_season']
        last_year_team_rating_x_num_games_into_season = row['last_year_team_rating*num_games_into_season']
        last_year_opp_rating_x_num_games_into_season = row['last_year_opponent_rating*num_games_into_season']
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

        train_data = pd.DataFrame([[team_rating, opp_rating, team_win_total_future, opponent_win_total_future, last_year_team_rating, last_year_opp_rating, num_games_into_season, last_year_team_rating_x_num_games_into_season, last_year_opp_rating_x_num_games_into_season, team_last_10_rating, opponent_last_10_rating, team_last_5_rating, opponent_last_5_rating, team_last_3_rating, opponent_last_3_rating, team_last_1_rating, opponent_last_1_rating]], columns=['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating'])

        expected_margin = win_margin_model.predict(train_data)[0]
        # normal random variable with mean 0 + margin_model_resid_mean and std margin_model_resid_std
        margin = np.random.normal(0, margin_model_resid_std) + margin_model_resid_mean + expected_margin

        if margin > 0:
            team_win = True
        else:
            team_win = False

        return pd.Series({'completed': True, 'team_win': team_win, 'margin': margin})

    win_results = {}
    loss_results = {}

    for team in historic_games_with_ratings['team'].unique():
        win_results[team] = []
        loss_results[team] = []

    games = historic_games_with_ratings[historic_games_with_ratings['year'] == year]
    games['last_year_team_rating*num_games_into_season'] = games['last_year_team_rating'] * games['num_games_into_season']
    games['last_year_opponent_rating*num_games_into_season'] = games['last_year_opponent_rating'] * games['num_games_into_season']
    completed_games = games[games['completed'] == True]
    completed_games['team_win'] = completed_games['margin'] > 0

    future_games = games[games['completed'] == False]
    future_games.sort_values(by='date', inplace=True)
    num_sims = 1
    print('min game', future_games['date'].min())
    print('max game', future_games['date'].max())

    date_increment = 1
    for sim in range(num_sims):
        sim_season_games_completed = completed_games.copy()
        sim_season_games_future = future_games.copy()
        print()
        print('sim {}/{}'.format(sim + 1, num_sims))

        # iterate over every <date_increment> days
        daterange = sorted(sim_season_games_future['date'].unique())
        daterange.append(daterange[-1] + datetime.timedelta(days=date_increment))
        for date in daterange[::date_increment]:
            start_date = date
            end_date = date + datetime.timedelta(days=date_increment)
            games_on_date = sim_season_games_future[(sim_season_games_future['date'] < end_date + datetime.timedelta(days=10)) & (sim_season_games_future['date'] >= start_date)]

            if games_on_date.empty:
                continue
            start_time = time.time()
            sim_season_games_completed = utils.last_n_games(sim_season_games_completed, 10)
            sim_season_games_completed = utils.last_n_games(sim_season_games_completed, 5)
            sim_season_games_completed = utils.last_n_games(sim_season_games_completed, 3)
            sim_season_games_completed = utils.last_n_games(sim_season_games_completed, 1)
            print('Process 0 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()

            # add the results of the simulated game to the games_on_date dataframe
            # games_on_date[['completed', 'team_win', 'margin']] = games_on_date.apply(sim_game, axis=1)[['completed', 'team_win', 'margin']]
            # rewrite the above using .loc
            games_on_date.loc[:, ['completed', 'team_win', 'margin']] = games_on_date.apply(sim_game, axis=1)[['completed', 'team_win', 'margin']]   

            sim_season_games_completed = sim_season_games_completed.append(games_on_date)
            sim_season_games_future = sim_season_games_future[~sim_season_games_future.index.isin(games_on_date.index)]

            sim_season_games_completed.loc[games_on_date.index, 'completed'] = True
            sim_season_games_completed.loc[games_on_date.index, 'team_win'] = games_on_date['team_win']
            sim_season_games_completed.loc[games_on_date.index, 'margin'] = games_on_date['margin']

            print('Process 1 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()
            last_10_games_dict = utils.get_last_n_games_dict(sim_season_games_completed, 10)
            last_5_games_dict = utils.get_last_n_games_dict(sim_season_games_completed, 5)
            last_3_games_dict = utils.get_last_n_games_dict(sim_season_games_completed, 3)
            last_1_games_dict = utils.get_last_n_games_dict(sim_season_games_completed, 1)
            print('Process 2 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()

            sim_season_games_future['team_last_10_rating'] = sim_season_games_future['team'].map(last_10_games_dict)
            sim_season_games_future['opponent_last_10_rating'] = sim_season_games_future['opponent'].map(last_10_games_dict)

            sim_season_games_future['team_last_5_rating'] = sim_season_games_future['team'].map(last_5_games_dict)
            sim_season_games_future['opponent_last_5_rating'] = sim_season_games_future['opponent'].map(last_5_games_dict)

            sim_season_games_future['team_last_3_rating'] = sim_season_games_future['team'].map(last_3_games_dict)
            sim_season_games_future['opponent_last_3_rating'] = sim_season_games_future['opponent'].map(last_3_games_dict)

            sim_season_games_future['team_last_1_rating'] = sim_season_games_future['team'].map(last_1_games_dict)
            sim_season_games_future['opponent_last_1_rating'] = sim_season_games_future['opponent'].map(last_1_games_dict)
            print('Process 3 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()

            df_for_ratings = sim_season_games_completed.copy()
            df_for_ratings['team'] = df_for_ratings['team']
            df_for_ratings['opponent'] = df_for_ratings['opponent']

            print('Process 4 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()
            em_ratings = utils.get_em_ratings(df_for_ratings)
            print('Process 5 Time:', time.time() - start_time, 'seconds')
            start_time = time.time()

            sim_season_games_future['team_rating'] = sim_season_games_future['team'].map(em_ratings)
            sim_season_games_future['opponent_rating'] = sim_season_games_future['opponent'].map(em_ratings)

        wins_by_team = {team: 0 for team in sim_season_games_completed['team'].unique()}
        losses_by_team = {team: 0 for team in sim_season_games_completed['team'].unique()}

        for idx, game in sim_season_games_completed.iterrows():
            if game['team_win']:
                wins_by_team[game['team']] += 1
                losses_by_team[game['opponent']] += 1
            else:
                wins_by_team[game['opponent']] += 1
                losses_by_team[game['team']] += 1

        assert np.sum(list(wins_by_team.values())) == 82 * 15
        for team in wins_by_team:
            win_results[team].append(wins_by_team[team])
            loss_results[team].append(losses_by_team[team])
        
        sim_res_df = pd.DataFrame({'team': list(wins_by_team.keys()), 'wins': list(wins_by_team.values()), 'losses': list(losses_by_team.values())})
        # drop index
        sim_res_df = sim_res_df.reset_index(drop=True)
        sim_res_df.sort_values(by='wins', ascending=False, inplace=True)
        sim_res_df.index = list(range(1, len(sim_res_df) + 1))
        print(sim_res_df)
       
    return win_results, loss_results

def main(update=True):
    YEAR = 2023
    if update:
        names_to_abbr = data_loader.get_team_names(year=YEAR)
        abbrs = list(names_to_abbr.values())
        games = data_loader.update_data(names_to_abbr)
    else:
        games = pd.read_csv('data/games/year_data_{}.csv'.format(YEAR))
        games.rename(columns={'team_abbr': 'team', 'opponent_abbr': 'opponent'}, inplace=True)
        names_to_abbr = {games['team_name'].iloc[i]: games['team'].iloc[i] for i in range(len(games))}
        abbrs = list(names_to_abbr.values())

    abbr_to_name = {v: k for k, v in names_to_abbr.items()}
    completed_games = games[games['completed']]
    future_games = games[~games['completed']]
    mean_pace = completed_games['pace'].mean()
    std_pace = completed_games['pace'].std()

    # EM RATINGS
    em_ratings = utils.get_em_ratings(completed_games, depth=10000)
    em_ratings = {k: v for k, v in sorted(em_ratings.items(), key=lambda item: item[1], reverse=True)}
    ratings_lst = []
    for i, (team, rating) in enumerate(em_ratings.items()):
        ratings_lst.append([i + 1, team, round(rating, 2)])
    em_ratings_df = pd.DataFrame(ratings_lst, columns=['rank', 'team', 'rating'])
    em_ratings_df.to_csv('data/ratings_' + str(YEAR) + '.csv')
    
    # INITIALIZE DATAFRAME
    df_final = pd.DataFrame(index=abbrs)
    df_final['team'] = df_final.index
    df_final['team_name'] = [abbr_to_name[abbr] for abbr in abbrs]
    df_final['em_rating'] = [em_ratings[abbr] for abbr in abbrs]
    df_final.sort_values(by='em_rating', ascending=False, inplace=True)
    df_final['rank'] = [i + 1 for i in range(len(abbrs))]

    # ADD DATA
    off_eff = stats.get_offensive_efficiency(completed_games)
    def_eff = stats.get_defensive_efficiency(completed_games)
    adj_off_eff, adj_def_eff = stats.get_adjusted_efficiencies(completed_games, off_eff, def_eff)
    paces = stats.get_pace(completed_games)
    wins, losses = stats.get_wins_losses(completed_games)

    # PUT IT ALL TOGETHER
    df_final['wins'] = df_final.apply(lambda x: wins.get(x['team'], 0), axis=1)
    df_final['losses'] = df_final.apply(lambda x: losses.get(x['team'], 0), axis=1)
    df_final['win_pct'] = df_final['wins'] / (df_final['wins'] + df_final['losses'])
    df_final['pace'] = df_final.apply(lambda x: paces.get(x['team'], 0), axis=1)
    df_final['off_eff'] = df_final.apply(lambda x: off_eff.get(x['team'], 0), axis=1)
    df_final['def_eff'] = df_final.apply(lambda x: def_eff.get(x['team'], 0), axis=1)
    df_final['off_eff'] = 100 * df_final['off_eff']
    df_final['def_eff'] = 100 * df_final['def_eff']
    df_final['adj_off_eff'] = df_final.apply(lambda x: adj_off_eff.get(x['team'], 0), axis=1)
    df_final['adj_def_eff'] = df_final.apply(lambda x: adj_def_eff.get(x['team'], 0), axis=1)
    df_final['adj_def_eff'] = -df_final['adj_def_eff']
    df_final = df_final[['rank', 'team', 'team_name', 'em_rating', 'wins', 'losses', 'win_pct', 'off_eff', 'def_eff', 'adj_off_eff', 'adj_def_eff', 'pace']]
    print(df_final.head(30))
            
    # GET MODELS
    training_data = data_loader.load_training_data(update=False)
    win_prob_model = eval.get_win_probability_model(training_data)
    win_margin_model, mean_margin_model_resid, std_margin_model_resid = eval.get_win_margin_model(training_data)
    this_week_pred_margin = forecast.predict_margin_this_week_games(training_data, win_margin_model)

    # SIMULATE SEASON
    sim_wins_dict, sim_losses_dict = sim_season(training_data, win_margin_model, mean_margin_model_resid, std_margin_model_resid, mean_pace, std_pace, year=YEAR)

    # track simulation wins
    sim_wins_lst = []
    for i, (team, wins) in enumerate(sim_wins_dict.items()):
        sim_wins_lst.append([i + 1, team, wins])
    sim_wins_df = pd.DataFrame(sim_wins_lst, columns=['rank', 'team', 'wins'])

    # track simulation losses
    sim_losses_lst = []
    for i, (team, losses) in enumerate(sim_losses_dict.items()):
        sim_losses_lst.append([i + 1, team, losses])
    sim_losses_df = pd.DataFrame(sim_losses_lst, columns=['rank', 'team', 'losses'])

    # merge wins and losses and save simulation results
    sim_df = sim_wins_df.merge(sim_losses_df, on='team')
    sim_df.to_csv('data/sim_' + str(YEAR) + '.csv', index=False)

    # get expected wins and losses
    sim_losses_avg_dict = {team: np.mean(losses) for team, losses in sim_losses_dict.items()}
    sim_wins_avg_dict = {team: np.mean(wins) for team, wins in sim_wins_dict.items()}

    # PREDICTIVE RATINGS
    predictive_ratings = forecast.get_predictive_ratings_win_margin(win_margin_model, year=YEAR)
    predictive_ratings = predictive_ratings['expected_margin'].to_dict()

    future_games_with_win_probs = forecast.predict_win_prob_future_games(training_data, win_prob_model)
    future_games_with_win_probs.to_csv('data/future_games_with_win_probs.csv', index=False)

    # ADD PREDICTIVE RATINGS and SIM RESULTS TO FINAL DF
    sim_losses_df, sim_wins_df = sim_losses_df.set_index('team'), sim_wins_df.set_index('team')
    df_final['predictive_rating'] = df_final['team'].apply(lambda x: predictive_ratings[x])
    df_final['expected_wins'] = df_final.apply(lambda x: sim_wins_avg_dict[x['team']], axis=1)
    df_final['expected_losses'] = df_final.apply(lambda x: sim_losses_avg_dict[x['team']], axis=1)
    df_final['expected_record'] = df_final.apply(lambda x: str(round(x['expected_wins'], 1)) + '-' + str(round(x['expected_losses'], 1)), axis=1)
    remaining_sos = stats.get_remaining_sos(df_final, future_games)
    df_final['remaining_sos'] = df_final['team'].apply(lambda x: remaining_sos[x])

    # FORMAT FOR CSV
    df_final['current_record'] = df_final.apply(lambda x: str(x['wins']) + '-' + str(x['losses']), axis=1)
    df_final.rename(columns={'current_record': 'Record', 'rank': 'Rank', 'team_name': 'Team', 'em_rating': 'EM Rating', 'win_pct': 'Win %', 'off_eff': 'Offensive Efficiency', 'def_eff': 'Defensive Efficiency', 'adj_off_eff': 'AdjOffEff', 'adj_def_eff': 'AdjDefEff', 'pace': 'Pace', 'predictive_rating': 'Predictive Rating', 'expected_record': 'Projected Record', 'remaining_sos': 'Remaining SOS'}, inplace=True)
    df_final = df_final[['Rank', 'Team', 'Record', 'EM Rating', 'Predictive Rating', 'Projected Record', 'AdjOffEff', 'AdjDefEff', 'Pace', 'Remaining SOS']]
    df_final['EM Rating'] = df_final['EM Rating'].apply(lambda x: round(x, 2))
    df_final['Predictive Rating'] = df_final['Predictive Rating'].apply(lambda x: round(x, 2))
    df_final['AdjOffEff'] = df_final['AdjOffEff'].apply(lambda x: round(x, 2))
    df_final['AdjDefEff'] = df_final['AdjDefEff'].apply(lambda x: round(x, 2))
    df_final['Pace'] = df_final['Pace'].apply(lambda x: round(x, 2))
    df_final['Remaining SOS'] = df_final['Remaining SOS'].apply(lambda x: round(x, 2))
    df_final.to_csv('data/all_data_' + str(YEAR) + '.csv', index=False)
    print(df_final.head(30))


if __name__ == '__main__':
    main(update=False)

