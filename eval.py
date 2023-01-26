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
# import gridsearchcv
from sklearn.model_selection import GridSearchCV


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

def tune_xgb_margin_model(games):
    # use grid search to tune the parameters
    games = games[games['completed'] == True]

    x_features = ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating']
    train_df, test_df = train_test_split(games, test_size=0.2, random_state=41)
    x_train = train_df[x_features]
    y_train = train_df['margin']
    x_test = test_df[x_features]
    y_test = test_df['margin']

    # define the grid search parameters
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.001, 0.01, 0.1, 0.3],
        # 'n_estimators': [100, 500, 1000],
        # 'min_child_weight': [1, 3, 5],
        # 'gamma': [0, 0.1, 0.2],
        # 'subsample': [0.4, 0.5, 0.6],
        # 'colsample_bytree': [0.4, 0.5, 0.6]
    }

    xgb = XGBRegressor(random_state=41)
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_





def get_win_margin_model(games):
    # TODO: fine tune this in the ipython notebook
    # just use linear regression for now
    games = games[games['completed'] == True]
    x_features = ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game']
    train_df, test_df = train_test_split(games, test_size=0.2, random_state=41)
   
    def duplicate_games(df):
        '''
        duplicates the games in the dataframe so that the team and opponent are switched
        '''
        duplicated_games = []
        for idx, game in df.iterrows():
            duplicated_games.append([game['team_rating'], game['opponent_rating'], game['team_win_total_future'], game['opponent_win_total_future'], game['last_year_team_rating'], game['last_year_opponent_rating'], game['num_games_into_season'], game['team_last_10_rating'], game['opponent_last_10_rating'], game['team_last_5_rating'], game['opponent_last_5_rating'], game['team_last_3_rating'], game['opponent_last_3_rating'], game['team_last_1_rating'], game['opponent_last_1_rating'], game['team_days_since_most_recent_game'], game['opponent_days_since_most_recent_game'], game['margin']])
            duplicated_games.append([game['opponent_rating'], game['team_rating'], game['opponent_win_total_future'], game['team_win_total_future'], game['last_year_opponent_rating'], game['last_year_team_rating'], game['num_games_into_season'], game['opponent_last_10_rating'], game['team_last_10_rating'], game['opponent_last_5_rating'], game['team_last_5_rating'], game['opponent_last_3_rating'], game['team_last_3_rating'], game['opponent_last_1_rating'], game['team_last_1_rating'], game['opponent_days_since_most_recent_game'], game['team_days_since_most_recent_game'], -game['margin']])
        # create the dataframe
        duplicated_games_df = pd.DataFrame(duplicated_games, columns=x_features + ['margin'])
        return duplicated_games_df
    
    # train_df = duplicate_games(train_df)
    # test_df = duplicate_games(test_df)
    X_train, y_train, X_test, y_test = train_df[x_features], train_df['margin'], test_df[x_features], test_df['margin']
    
    params = {'max_depth': 5, 'learning_rate': 0.017663416491675688, 'n_estimators': 483, 'min_child_weight': 8, 'gamma': 0.2098788215570124, 'subsample': 0.8114787013965323, 'colsample_bytree': 0.776777067348792, 'reg_alpha': 0.409398999830202, 'reg_lambda': 0.5816468426359469, 'random_state': 674}
    model = XGBRegressor(**params)
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

def get_win_probability_model(games, win_margin_model):
    '''
    just predicts from predicted margin
    '''
    from sklearn.linear_model import LogisticRegression
    games = games[games['completed'] == True]
    games['expected_margin'] = win_margin_model.predict(games[['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game']])
    games['team_win'] = games['margin'] > 0
    model = LogisticRegression()
    model.fit(games[['expected_margin']], games['team_win'])
    return model

