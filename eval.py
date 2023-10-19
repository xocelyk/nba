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
import env

def get_win_margin_model_heavy(games):
    games = games[games['completed'] == True]
    x_features = env.x_features_heavy
    X = games[x_features]
    y = games['margin']
    params = {'max_depth': 5, 'learning_rate': 0.014567791445364069, 'n_estimators': 655, 'min_child_weight': 2, 'gamma': 0.11052774751544212, 'subsample': 0.9899289938848144, 'colsample_bytree': 0.9249071617042357, 'reg_alpha': 0.4468005337539522, 'reg_lambda': 0.24513091931966713, 'random_state': 996}
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model

def get_win_probability_model_heavy(games):
    games = games[games['completed'] == True]
    games['win'] = games['margin'] > 0
    games['win'] = games['win'].astype(int)
    x_features = env.x_features_heavy
    X = games[x_features]
    y = games['win']

    params = env.win_prob_model_params
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model

def get_win_margin_model(games):
    games = games[games['completed'] == True]
    x_features = env.x_features
    params = env.win_margin_model_params
    model = XGBRegressor(**params)
    train_df, test_df = train_test_split(games, test_size=0.2, random_state=41)    

    # first split train/test to get error vals (needed for simulations)
    X_train, y_train, X_test, y_test = train_df[x_features], train_df['margin'], test_df[x_features], test_df['margin']
    X = games[x_features]
    y = games['margin']
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    errors = preds - y_test
    m = np.mean(errors)
    std = np.std(errors)
    # print('Win Margin Model RMSE:')
    # print(np.sqrt(np.mean((preds - y_test)**2)))
    # print()
    # print('Win Margin Model MAE:')
    # print(np.mean(np.abs(preds - y_test)))
    # print()

    def prediction_interval_stdev(model, x_test, y_test):
        preds = model.predict(x_test)
        errors = preds - y_test
        m = np.mean(errors)
        std = np.std(errors)
        return m, std

    m, std = prediction_interval_stdev(model, X_test, y_test)

    # now fit on all the data for final model
    model = XGBRegressor(**params)
    model.fit(X, y)
    return model, m, std

def get_win_probability_model(games, win_margin_model):
    '''
    just predicts from predicted margin
    '''
    # from sklearn.linear_model import LogisticRegression
    # games = games[games['completed'] == True]
    # games['expected_margin'] = win_margin_model.predict(games[['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game']])
    # games['team_win'] = games['margin'] > 0
    # model = LogisticRegression()
    # model.fit(games[['expected_margin']], games['team_win'])
    # # print log loss
    # from sklearn.metrics import log_loss
    # print('Win Prob Model Log Loss')
    # print(log_loss(games['team_win'], model.predict_proba(games[['expected_margin']])[:,1]))
    # print()
    # return model

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    games = games[games['completed'] == True]
    x_features = env.x_features
    X = games[x_features]
    X['pred_margin'] = win_margin_model.predict(X)
    games['pred_margin'] = X['pred_margin']
    X = X[['pred_margin']]
    logit_inv = lambda x: np.log(x / (1 - x))
    intercept = -(logit_inv(0.5) / X.mean())
    print(intercept)
    games['win'] = games['margin'] > 0
    #TODO: fit intercept?
    model = LogisticRegression(fit_intercept=False)
    model.fit(X, games['win'])
    # plot win probability
    # import matplotlib.pyplot as plt
    # plt.scatter(X['pred_margin'], model.predict_proba(X[['pred_margin']])[:,1])
    # plt.xlabel('Predicted Margin')
    # plt.ylabel('Win Probability')
    # add true win proportion binned by margin of 1
    # win_probs = []
    # margins = []
    # for i in range(-30, 31):
    #     win_probs.append(games[games['pred_margin'].round(0) == i]['win'].mean())
    #     margins.append(i)
    # for i in range(-30, 31):
    #     print(i, games[games['pred_margin'] == i]['win'].mean())
    # plt.scatter(margins, win_probs, color='red')


    # plt.show()
    # print log loss
    from sklearn.metrics import log_loss
    # print('Win Prob Model Log Loss')
    # print(log_loss(games['win'], model.predict_proba(X[['pred_margin']])[:,1]))
    # print()
    return model

    games['team_win'] = games['margin'] > 0

    params = {'max_depth': 5, 'learning_rate': 0.01337501236333186, 'n_estimators': 615, 'min_child_weight': 6, 'gamma': 0.22171810700204012, 'subsample': 0.23183800840898533, 'colsample_bytree': 0.29826505641378537, 'reg_alpha': 0.5869931848470185, 'reg_lambda': 0.01392437600344064, 'random_state': 931}
    model = XGBClassifier(**params)
    model.fit(X, games['team_win'])
    # print log loss
    from sklearn.metrics import log_loss
    print('Win Prob Model Log Loss')
    print(log_loss(games['team_win'], model.predict_proba(X)[:,1]))
    print()
    return model
