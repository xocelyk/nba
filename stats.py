import numpy as np


def get_wins_losses(game_df):
    wins = {}
    losses = {}
    for idx, row in game_df.iterrows():
        if row['margin'] > 0:
            wins[row['team']] = wins.get(row['team'], 0) + 1
            losses[row['opponent']] = losses.get(row['opponent'], 0) + 1
        else:
            losses[row['team']] = losses.get(row['team'], 0) + 1
            wins[row['opponent']] = wins.get(row['opponent'], 0) + 1
    return wins, losses

def get_offensive_efficiency(data):
    off_eff = {}
    for idx, row in data.iterrows():
        home_points = row['team_score']
        away_points = row['opponent_score']
        pace = row['pace']
        home_off_eff = home_points / pace
        away_off_eff = away_points / pace
        home = row['team']
        away = row['opponent']
        if home not in off_eff:
            off_eff[home] = []
        if away not in off_eff:
            off_eff[away] = []
        off_eff[row['team']].append(home_off_eff)
        off_eff[row['opponent']].append(away_off_eff)
    for team, off_effs in off_eff.items():
        off_eff[team] = np.mean(off_effs)
    return off_eff

def get_defensive_efficiency(data):
    def_eff = {}
    for idx, row in data.iterrows():
        home_points = row['team_score']
        away_points = row['opponent_score']
        pace = row['pace']
        home_def_eff = away_points / pace
        away_def_eff = home_points / pace
        home = row['team']
        away = row['opponent']
        if home not in def_eff:
            def_eff[home] = []
        if away not in def_eff:
            def_eff[away] = []
        def_eff[row['team']].append(home_def_eff)
        def_eff[row['opponent']].append(away_def_eff)
    for team, def_effs in def_eff.items():
        def_eff[team] = np.mean(def_effs)
    return def_eff

def get_adjusted_efficiencies(data, def_eff, off_eff):
    '''
    gets both adjusted defensive efficiency and adjusted offensive efficiency
    '''
    adj_off_eff = off_eff.copy()
    adj_def_eff = def_eff.copy()

    for loop in range(100):
        off_eff_diffs = {team: [] for team in off_eff.keys()}
        def_eff_diffs = {team: [] for team in off_eff.keys()}
        for idx, row in data.iterrows():
            home_points = row['team_score']
            away_points = row['opponent_score']
            pace = row['pace']
            home_ppp = home_points / pace
            away_ppp = away_points / pace
            offensive_effect = 0.65
            defensive_effect = 0.35
            # TODO: should build out my old model for this
            # using kenpom numbers from https://kenpom.com/blog/offense-vs-defense-the-summary/ (see ppp)
            proj_home_ppp = offensive_effect * adj_off_eff[row['team']] + defensive_effect * adj_def_eff[row['opponent']]
            proj_away_ppp = offensive_effect * adj_off_eff[row['opponent']] + defensive_effect * adj_def_eff[row['team']]
            off_eff_diffs[row['team']].append((home_ppp - proj_home_ppp) / 2)
            def_eff_diffs[row['opponent']].append((home_ppp - proj_home_ppp) / 2)
            off_eff_diffs[row['opponent']].append((away_ppp - proj_away_ppp) / 2)
            def_eff_diffs[row['team']].append((away_ppp - proj_away_ppp) / 2)

        prev_off_eff = adj_off_eff.copy()
        prev_def_eff = adj_def_eff.copy()

        for team, off_eff_diffs in off_eff_diffs.items():
            adj_off_eff[team] = adj_off_eff[team] + np.mean(off_eff_diffs)
        for team, def_eff_diffs in def_eff_diffs.items():
            adj_def_eff[team] = adj_def_eff[team] + np.mean(def_eff_diffs)

        # calc l2 norm
        off_eff_diff = 0
        def_eff_diff = 0
        for team, off_eff in adj_off_eff.items():
            off_eff_diff += (off_eff - prev_off_eff[team]) ** 2
        for team, def_eff in adj_def_eff.items():
            def_eff_diff += (def_eff - prev_def_eff[team]) ** 2
        off_eff_diff = np.sqrt(off_eff_diff)
        def_eff_diff = np.sqrt(def_eff_diff)
        if off_eff_diff < 1e-4 and def_eff_diff < 1e-4:
            break
    
    mean_off_eff = np.mean(list(adj_off_eff.values()))
    mean_def_eff = np.mean(list(adj_def_eff.values()))

    for team, off_eff in adj_off_eff.items():
        adj_off_eff[team] = 100 * (off_eff - mean_off_eff) / 2
    
    for team, def_eff in adj_def_eff.items():
        adj_def_eff[team] = 100 * (def_eff - mean_def_eff) / 2
    
    return adj_off_eff, adj_def_eff
            
def get_adjusted_offensive_efficiency(data, def_eff):
    adj_off_eff = {team: [] for team in def_eff.keys()}
    for idx, row in data.iterrows():
        home_points = row['team_score']
        away_points = row['opponent_score']
        pace = row['pace']
        away_def_eff = def_eff[row['team']]
        home_def_eff = def_eff[row['opponent']]
        home_adj_off_eff = home_points / pace - away_def_eff
        away_adj_off_eff = away_points / pace - home_def_eff
        adj_off_eff[row['team']].append(home_adj_off_eff)
        adj_off_eff[row['opponent']].append(away_adj_off_eff)
    for team, adj_off_effs in adj_off_eff.items():
        adj_off_eff[team] = np.mean(adj_off_effs)
    return adj_off_eff

def get_adjusted_defensive_efficiency(data, off_eff):
    adj_def_eff = {team: [] for team in off_eff.keys()}
    for idx, row in data.iterrows():
        home_points = row['team_score']
        away_points = row['opponent_score']
        pace = row['pace']
        away_off_eff = off_eff[row['team']]
        home_off_eff = off_eff[row['opponent']]
        home_adj_def_eff = away_points / pace - away_off_eff
        away_adj_def_eff = home_points / pace - home_off_eff
        adj_def_eff[row['team']].append(home_adj_def_eff)
        adj_def_eff[row['opponent']].append(away_adj_def_eff)
    for team, adj_def_effs in adj_def_eff.items():
        adj_def_eff[team] = np.mean(adj_def_effs)
    return adj_def_eff

def get_adjusted_efficiencies(data, off_eff, def_eff):
    off_eff_copy = off_eff.copy()
    def_eff_copy = def_eff.copy()
    for loop in range(100):
        adj_off_eff = get_adjusted_offensive_efficiency(data, def_eff_copy)
        adj_def_eff = get_adjusted_defensive_efficiency(data, off_eff_copy)
        off_eff_copy = adj_off_eff
        def_eff_copy = adj_def_eff
    return off_eff_copy, def_eff_copy

def get_pace(data):
    paces = {}
    for idx, row in data.iterrows():
        pace = row['pace']
        home = row['team']
        away = row['opponent']
        if home not in paces:
            paces[home] = []
        if away not in paces:
            paces[away] = []
        paces[home].append(pace)
        paces[away].append(pace)
    for team, pace_lst in paces.items():
        paces[team] = np.mean(pace_lst)
    return paces

def get_remaining_sos(ratings_df, future_games):
    # returns a dictionary of remaining strength of schedule for each team
    remaining_sos = {team: [] for team in ratings_df['team'].unique()}
    for idx, row in future_games.iterrows():
        team = row['team']
        opponent = row['opponent']
        remaining_sos[team].append(ratings_df[ratings_df['team'] == opponent]['predictive_rating'].values[0])
        remaining_sos[opponent].append(ratings_df[ratings_df['team'] == team]['predictive_rating'].values[0])
    for team in remaining_sos:
        remaining_sos[team] = np.mean(remaining_sos[team])
    return remaining_sos

