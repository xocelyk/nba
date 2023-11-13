x_features = ['team_rating', 'opponent_rating', 'team_win_total_future', 'opponent_win_total_future', 'last_year_team_rating', 'last_year_opponent_rating', 'num_games_into_season', 'team_last_10_rating', 'opponent_last_10_rating', 'team_last_5_rating', 'opponent_last_5_rating', 'team_last_3_rating', 'opponent_last_3_rating', 'team_last_1_rating', 'opponent_last_1_rating', 'team_days_since_most_recent_game', 'opponent_days_since_most_recent_game']
x_features_heavy = x_features.copy()
margin_label = 'margin'
win_prob_label = 'win'
win_margin_model_params = {'n_estimators': 127, 'max_depth': 3, 'learning_rate': 0.05277779665608899, 'subsample': 0.9828987652763124, 'colsample_bytree': 0.6867888479797172, 'gamma': 0.845516600015586, 'reg_alpha': 0.14855521764812535, 'reg_lambda': 0.5708849537577776, 'min_child_weight': 1.1864436159788416}
win_prob_model_params = {'max_depth': 5, 'learning_rate': 0.01337501236333186, 'n_estimators': 615, 'min_child_weight': 6, 'gamma': 0.22171810700204012, 'subsample': 0.23183800840898533, 'colsample_bytree': 0.29826505641378537, 'reg_alpha': 0.5869931848470185, 'reg_lambda': 0.01392437600344064, 'random_state': 931}
