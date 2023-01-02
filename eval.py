import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


def compare_margin_models(all_data):
    seed = 41
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('RFR', RandomForestRegressor(n_estimators=100, random_state=seed)))
    models.append(('XGBR', XGBRegressor(random_state=seed)))

    # evaluate each model in turn
    results = []
    names = []
    scoring = 'neg_mean_squared_error'
    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, all_data[['team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season']], all_data['margin'], cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def get_win_margin_model(games):
    # TODO: fine tune this in the ipython notebook
    # just use linear regression for now
    games = games[games['completed'] == True]
    games.loc[:, 'last_year_team_rating*num_games_into_season'] = games.apply(lambda row: row['last_year_team_rating'] * row['num_games_into_season'], axis=1)
    games.loc[:, 'last_year_opponent_rating*num_games_into_season'] = games.apply(lambda row: row['last_year_opponent_rating'] * row['num_games_into_season'], axis=1)
    x_features = ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'last_year_team_rating*num_games_into_season', 'last_year_opponent_rating*num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating']
    train_df, test_df = train_test_split(games, test_size=0.2, random_state=41)
   
    def duplicate_games(df):
        '''
        duplicates the games in the dataframe so that the team and opponent are switched
        '''
        duplicated_games = []
        for idx, game in df.iterrows():
            duplicated_games.append([game['team_rating'], game['opponent_rating'], game['team_win_total_future'], game['opponent_win_total_future'], game['last_year_team_rating'], game['last_year_opponent_rating'], game['num_games_into_season'], game['last_year_team_rating']*game['num_games_into_season'], game['last_year_opponent_rating']*game['num_games_into_season'], game['team_last_10_rating'], game['opponent_last_10_rating'], game['team_last_5_rating'], game['opponent_last_5_rating'], game['team_last_3_rating'], game['opponent_last_3_rating'], game['team_last_1_rating'], game['opponent_last_1_rating'], game['margin']])
            duplicated_games.append([game['opponent_rating'], game['team_rating'], game['opponent_win_total_future'], game['team_win_total_future'], game['last_year_opponent_rating'], game['last_year_team_rating'], game['num_games_into_season'], game['last_year_opponent_rating']*game['num_games_into_season'], game['last_year_team_rating']*game['num_games_into_season'], game['opponent_last_10_rating'], game['team_last_10_rating'], game['opponent_last_5_rating'], game['team_last_5_rating'], game['opponent_last_3_rating'], game['team_last_3_rating'], game['opponent_last_1_rating'], game['team_last_1_rating'], -game['margin']])
        # create the dataframe
        duplicated_games_df = pd.DataFrame(duplicated_games, columns=x_features + ['margin'])
        return duplicated_games_df
    
    # train_df = duplicate_games(train_df)
    # test_df = duplicate_games(test_df)
    X_train, y_train, X_test, y_test = train_df[x_features], train_df['margin'], test_df[x_features], test_df['margin']
    
    model = XGBRegressor(random_state=41, max_depth=3, colsample_bytree=0.6, subsample=0.6, n_estimators=1000, learning_rate=0.01, gamma=.3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    errors = preds - y_test
    m = np.mean(errors)
    std = np.std(errors)
    print(np.sqrt(np.mean((preds - y_test)**2)))

    def prediction_interval_stdev(model, x_test, y_test):
        preds = model.predict(x_test)
        errors = preds - y_test
        m = np.mean(errors)
        std = np.std(errors)
        return m, std

    m, std = prediction_interval_stdev(model, X_test, y_test)
    return model, m, std

def get_win_probability_model(games):
    games = games[games['completed'] == True]
    games['home_win'] = games.apply(lambda x: 1 if x['margin'] > 0 else 0, axis=1)
    clf = XGBClassifier(random_state=41, max_depth=5, sampling_method='gradient_based')
    clf.fit(games[['team_rating', 'opponent_rating', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season']], games['home_win'])
    return clf

