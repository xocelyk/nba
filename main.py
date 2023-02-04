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
from sim_season import Season, MarginModel

'''
TODO list
 - use historic year data to impute pace data into training data composite
 - fix HCA calculation
 - impute with boxscore data
 - mark playoff games in training data
'''

# ignore warnings
warnings.filterwarnings('ignore')

def sim_season(data, win_margin_model, margin_model_resid_mean, margin_model_resid_std, mean_pace, std_pace, year):
    teams = data[data['year'] == year]['team'].unique()
    playoff_results_over_sims = {team: {} for team in teams}
    season_results_over_sims = {team: {'wins': [], 'losses': []} for team in teams}
    num_sims = 1000
    for sim in range(num_sims):
        start_time = time.time()
        print('Sim: ', sim + 1, '/', num_sims)
        data['date'] = pd.to_datetime(data['date']).dt.date
        margin_model = MarginModel(win_margin_model, margin_model_resid_mean, margin_model_resid_std)
        year_games = data[data['year'] == year]
        completed_year_games = year_games[year_games['completed'] == True]
        future_year_games = year_games[year_games['completed'] == False]
        season = Season(2023, completed_year_games, future_year_games, margin_model, mean_pace, std_pace)
        season.simulate_season()
        wins_losses_dict = season.get_win_loss_report()
        wins_dict = {team: wins_losses_dict[team][0] for team in wins_losses_dict}
        losses_dict = {team: wins_losses_dict[team][1] for team in wins_losses_dict}
        for team, wins in wins_dict.items():
            season_results_over_sims[team]['wins'].append(wins)
        for team, losses in losses_dict.items():
            season_results_over_sims[team]['losses'].append(losses)
        playoff_results = season.playoffs()
        for round, team_list in playoff_results.items():
            for team in team_list:
                if team not in playoff_results_over_sims:
                    playoff_results_over_sims[team] = {}
                if round not in playoff_results_over_sims[team]:
                    playoff_results_over_sims[team][round] = 0
                playoff_results_over_sims[team][round] += 1

        playoff_results_over_sims_df = pd.DataFrame(playoff_results_over_sims)
        playoff_results_over_sims_df = playoff_results_over_sims_df.transpose()
        playoff_results_over_sims_df = playoff_results_over_sims_df.reset_index()
        playoff_results_over_sims_df = playoff_results_over_sims_df.rename(columns={'index': 'team'})
        playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
        playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
        
        expected_record_dict = {}
        for team, season_results in season_results_over_sims.items():
            expected_wins = np.mean(season_results['wins'])
            expected_losses = np.mean(season_results['losses'])
            expected_record_dict[team] = {'wins': expected_wins, 'losses': expected_losses}
        
        sim_report_df = pd.DataFrame(expected_record_dict)
        sim_report_df = sim_report_df.transpose()
        sim_report_df = sim_report_df.reset_index()
        sim_report_df = sim_report_df.rename(columns={'index': 'team'})
        sim_report_df = sim_report_df.sort_values(by=['wins'], ascending=False)
        
        # merge with playoff results
        sim_report_df = sim_report_df.merge(playoff_results_over_sims_df, on='team')
        sim_report_df = sim_report_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
        sim_report_df[['champion', 'finals', 'conference_finals', 'second_round', 'playoffs']] = sim_report_df[['champion', 'finals', 'conference_finals', 'second_round', 'playoffs']] / (sim + 1)
        sim_report_df[['champion', 'finals', 'conference_finals', 'second_round', 'playoffs']] = sim_report_df[['champion', 'finals', 'conference_finals', 'second_round', 'playoffs']].round(2)
        sim_report_df = sim_report_df[['team', 'wins', 'losses', 'champion', 'finals', 'conference_finals', 'second_round', 'playoffs']]
        sim_report_df.set_index('team', inplace=True)

        print(sim_report_df)
        sim_time = np.round(time.time() - start_time, 2)
        print('Sim Time: ', sim_time, 's')
        
    for team, playoff_results in playoff_results_over_sims.items():
        # convert to percentage
        for round, num_times in playoff_results.items():
            playoff_results_over_sims[team][round] = num_times / num_sims
    
    # convert to dataframe
    playoff_results_over_sims_df = pd.DataFrame(playoff_results_over_sims)
    playoff_results_over_sims_df = playoff_results_over_sims_df.transpose()
    playoff_results_over_sims_df = playoff_results_over_sims_df.reset_index()
    playoff_results_over_sims_df = playoff_results_over_sims_df.rename(columns={'index': 'team'})
    playoff_results_over_sims_df = playoff_results_over_sims_df.fillna(0)
    playoff_results_over_sims_df = playoff_results_over_sims_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
    
    expected_record_dict = {}
    for team, season_results in season_results_over_sims.items():
        expected_wins = np.mean(season_results['wins'])
        expected_losses = np.mean(season_results['losses'])
        expected_record_dict[team] = {'wins': expected_wins, 'losses': expected_losses}
    
    sim_report_df = pd.DataFrame(expected_record_dict)
    sim_report_df = sim_report_df.transpose()
    sim_report_df = sim_report_df.reset_index()
    sim_report_df = sim_report_df.rename(columns={'index': 'team'})
    sim_report_df = sim_report_df.sort_values(by=['wins'], ascending=False)
    
    # merge with playoff results
    sim_report_df = sim_report_df.merge(playoff_results_over_sims_df, on='team')
    sim_report_df = sim_report_df.sort_values(by=['champion', 'finals', 'conference_finals', 'second_round', 'playoffs'], ascending=False)
    sim_report_df = sim_report_df[['team', 'wins', 'losses', 'champion', 'finals', 'conference_finals', 'second_round', 'playoffs']]
    sim_report_df.set_index('team', inplace=True)

    return sim_report_df

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
    em_ratings = utils.get_em_ratings(completed_games, max_iter=100)
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
    # TODO: maybe just calculate that for most recent n (30?) days
    off_eff = stats.get_offensive_efficiency(completed_games)
    def_eff = stats.get_defensive_efficiency(completed_games)
    adj_off_eff, adj_def_eff = stats.get_adjusted_efficiencies(completed_games, off_eff, def_eff)

    # TODO: this should be done before
    completed_games['date'] = pd.to_datetime(completed_games['date'])
    last_thirty_day_games = completed_games[completed_games['date'] > datetime.datetime.today() - datetime.timedelta(days=30)]
    last_thirty_day_off_eff = stats.get_offensive_efficiency(last_thirty_day_games)
    last_thirty_day_def_eff = stats.get_defensive_efficiency(last_thirty_day_games)
    last_thirty_day_adj_off_eff, last_thirty_day_adj_def_eff = stats.get_adjusted_efficiencies(last_thirty_day_games, last_thirty_day_off_eff, last_thirty_day_def_eff)

    adj_off_eff, adj_def_eff = last_thirty_day_adj_off_eff.copy(), last_thirty_day_adj_def_eff.copy()


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
    training_data = data_loader.load_training_data(update=update)
    win_margin_model, mean_margin_model_resid, std_margin_model_resid = eval.get_win_margin_model(training_data)
    win_prob_model = eval.get_win_probability_model(training_data, win_margin_model)
    forecast.predict_margin_and_win_prob_this_week_games(training_data, win_margin_model, win_prob_model)

    # SIMULATE SEASON
    sim_report = sim_season(training_data, win_margin_model, mean_margin_model_resid, std_margin_model_resid, mean_pace, std_pace, year=YEAR)

    # PREDICTIVE RATINGS
    predictive_ratings = forecast.get_predictive_ratings_win_margin(win_margin_model, year=YEAR)
    predictive_ratings = predictive_ratings['expected_margin'].to_dict()

    # ADD PREDICTIVE RATINGS and SIM RESULTS TO FINAL DF
    df_final['predictive_rating'] = df_final['team'].apply(lambda x: predictive_ratings[x])
    df_final.sort_values(by='predictive_rating', ascending=False, inplace=True)
    df_final['rank'] = [i + 1 for i in range(len(abbrs))]
    df_final['expected_wins'] = df_final['team'].apply(lambda x: sim_report.loc[x, 'wins'])
    df_final['expected_losses'] = df_final['team'].apply(lambda x: sim_report.loc[x, 'losses'])
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
    df_final.to_csv('data/all_data_' + str(YEAR) + '.csv', index=False)
    print(df_final.head(30))


if __name__ == '__main__':
    main(update=False)

