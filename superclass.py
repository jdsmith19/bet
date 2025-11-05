from DataAggregate.DataAggregate import DataAggregate
from prediction_models.XGBoost import XGBoostSlim
from prediction_models.LinearRegression import LinearRegression
from prediction_models.RandomForest import RandomForest
from prediction_models.LogisticRegression import LogisticRegression
from prediction_models.KNearest import KNearest

#da = DataAggregate("57d81d16fc6c36b35f5d8bb5893e1d3d", save_api_calls = False)
da = DataAggregate("57d81d16fc6c36b35f5d8bb5893e1d3d")

predictions = []
regressor_target = 'point_differential'
regressor_feature_cols = ['avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_point_differential_l5', 'days_rest', 'elo_rating']

xgb = XGBoostSlim(da, regressor_target, regressor_feature_cols)
results = xgb.predict_spread(da.prediction_set)
predictions.append(xgb.model_output)

lr = LinearRegression(da, regressor_target, regressor_feature_cols)
results = lr.predict_spread(da.prediction_set)
predictions.append(lr.model_output)

rf = RandomForest(da, regressor_target, regressor_feature_cols)
results = rf.predict_spread(da.prediction_set)
predictions.append(rf.model_output)

classifer_target = 'win'
classifier_feature_cols = ['avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5','days_rest','elo_rating']

lg = LogisticRegression(da, classifer_target, classifier_feature_cols)
results = lg.predict_winner(da.prediction_set)
predictions.append(lg.model_output)

kn = KNearest(da, classifer_target, classifier_feature_cols)
results = kn.predict_winner(da.prediction_set)
predictions.append(kn.model_output)

print(predictions)