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
from sim_season import sim_season

# ignore warnings
warnings.filterwarnings('ignore')

def main(update=True, save_names=False):
    import pickle
    YEAR = 2024
    if update:
        # names_to_abbr = data_loader.get_team_names(year=YEAR)
        if save_names:
            # save to pickle
            import pickle
            with open('data/names_to_abbr_{}.pkl'.format(YEAR), 'wb') as f:
                pickle.dump(names_to_abbr, f)
        with open('data/names_to_abbr_{}.pkl'.format(YEAR), 'rb') as f:
            names_to_abbr = pickle.load(f)
        abbrs = list(names_to_abbr.values())
        games = data_loader.update_data(names_to_abbr, preload=True)
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
    em_ratings = utils.get_em_ratings(completed_games, names=abbrs, cap=20)
    em_ratings = {k: v for k, v in sorted(em_ratings.items(), key=lambda item: item[1], reverse=True)}
    ratings_lst = []
    for i, (team, rating) in enumerate(em_ratings.items()):
        ratings_lst.append([i + 1, team, round(rating, 2)])
    em_ratings_df = pd.DataFrame(ratings_lst, columns=['rank', 'team', 'rating'])
    em_ratings_df.to_csv('data/em_ratings_' + str(YEAR) + '.csv')
    
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

    # TODO: this should be done before
    completed_games['date'] = pd.to_datetime(completed_games['date'], format='mixed')
    last_thirty_day_games = completed_games[completed_games['date'] > datetime.datetime.today() - datetime.timedelta(days=30)]
    last_thirty_day_off_eff = stats.get_offensive_efficiency(last_thirty_day_games)
    last_thirty_day_def_eff = stats.get_defensive_efficiency(last_thirty_day_games)
    last_thirty_day_adj_off_eff, last_thirty_day_adj_def_eff = stats.get_adjusted_efficiencies(last_thirty_day_games, last_thirty_day_off_eff, last_thirty_day_def_eff)

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
    df_final = df_final[['rank', 'team', 'team_name', 'em_rating', 'wins', 'losses', 'win_pct', 'off_eff', 'def_eff', 'adj_off_eff', 'adj_def_eff', 'pace']]
    print(df_final.head(30))
    
    # GET MODELS
    training_data = data_loader.load_training_data(abbrs, update=True, reset=False, this_year_games=games)
    win_margin_model, mean_margin_model_resid, std_margin_model_resid, num_games_to_std_margin_model_resid = eval.get_win_margin_model(training_data)
    win_prob_model = eval.get_win_probability_model(training_data, win_margin_model)
    forecast.predict_margin_and_win_prob_future_games(training_data, win_margin_model, win_prob_model)
    forecast.predict_margin_this_week_games(training_data, win_margin_model)

    # SIMULATE SEASON
    sim_report = sim_season(training_data, win_margin_model, mean_margin_model_resid, std_margin_model_resid, num_games_to_std_margin_model_resid, mean_pace, std_pace, year=YEAR, num_sims=1000, parallel=False)
    date_string = datetime.datetime.today().strftime('%Y-%m-%d')
    sim_report.to_csv('data/sim_results/sim_report' + date_string + '.csv')

    # PREDICTIVE RATINGS
    predictive_ratings = forecast.get_predictive_ratings_win_margin(abbrs, win_margin_model, year=YEAR)
    predictive_ratings = predictive_ratings['expected_margin'].to_dict()

    # ADD PREDICTIVE RATINGS and SIM RESULTS TO FINAL DF
    df_final['predictive_rating'] = df_final['team'].apply(lambda x: predictive_ratings[x])
    df_final.sort_values(by='predictive_rating', ascending=False, inplace=True)
    df_final['rank'] = [i + 1 for i in range(len(abbrs))]
    df_final['expected_wins'] = df_final['team'].apply(lambda x: sim_report.loc[x, 'wins'])
    df_final['expected_losses'] = df_final['team'].apply(lambda x: sim_report.loc[x, 'losses'])
    # 2024 correction for midseason tournament
    df_final['expected_wins_temp'] = df_final.apply(lambda row: 82 / (row['expected_wins'] + row['expected_losses']) * row['expected_wins'], axis=1)
    df_final['expected_losses_temp'] = df_final.apply(lambda row: 82 / (row['expected_wins'] + row['expected_losses']) * row['expected_losses'], axis=1)
    df_final['expected_wins'] = df_final['expected_wins_temp']
    df_final['expected_losses'] = df_final['expected_losses_temp']
    df_final.drop(columns=['expected_wins_temp', 'expected_losses_temp'], inplace=True)
    
    df_final['expected_record'] = df_final.apply(lambda x: str(round(x['expected_wins'], 1)) + '-' + str(round(x['expected_losses'], 1)), axis=1)
    df_final['Playoffs'] = df_final['team'].apply(lambda x: round(sim_report.loc[x, 'playoffs'], 3))
    df_final['Conference Finals'] = df_final['team'].apply(lambda x: round(sim_report.loc[x, 'conference_finals'], 3))
    df_final['Finals'] = df_final['team'].apply(lambda x: round(sim_report.loc[x, 'finals'], 3))
    df_final['Champion'] = df_final['team'].apply(lambda x: round(sim_report.loc[x, 'champion'], 3))
    remaining_sos = stats.get_remaining_sos(df_final, future_games)
    df_final['remaining_sos'] = df_final['team'].apply(lambda x: remaining_sos[x])

    # FORMAT FOR CSV
    df_final['current_record'] = df_final.apply(lambda x: str(x['wins']) + '-' + str(x['losses']), axis=1)
    df_final.rename(columns={'current_record': 'Record', 'rank': 'Rank', 'team_name': 'Team', 'em_rating': 'EM Rating', 'win_pct': 'Win %', 'off_eff': 'Offensive Efficiency', 'def_eff': 'Defensive Efficiency', 'adj_off_eff': 'AdjO', 'adj_def_eff': 'AdjD', 'pace': 'Pace', 'predictive_rating': 'Predictive Rating', 'expected_record': 'Projected Record', 'remaining_sos': 'RSOS'}, inplace=True)
    df_final = df_final[['Rank', 'Team', 'Record', 'EM Rating', 'Predictive Rating', 'Projected Record', 'AdjO', 'AdjD', 'Pace', 'RSOS', 'Playoffs', 'Conference Finals', 'Finals', 'Champion']]
    df_final['EM Rating'] = df_final['EM Rating'].apply(lambda x: round(x, 2))
    df_final['Predictive Rating'] = df_final['Predictive Rating'].apply(lambda x: round(x, 2))
    df_final['AdjO'] = df_final['AdjO'].apply(lambda x: round(x, 2))
    df_final['AdjD'] = df_final['AdjD'].apply(lambda x: round(x, 2))
    df_final['Pace'] = df_final['Pace'].apply(lambda x: round(x, 2))
    df_final['RSOS'] = df_final['RSOS'].apply(lambda x: round(x, 2))
    small_df = False
    if small_df:
        df_final = df_final[['Rank', 'Team', 'Record', 'EM Rating', 'Predictive Rating', 'AdjO', 'AdjD', 'Pace']]
    df_final.to_csv('data/main_' + str(YEAR) + '.csv', index=False)
    print(df_final.head(30))


if __name__ == '__main__':
    main(update=True)

