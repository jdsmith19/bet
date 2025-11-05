#from data_sources.ProFootballReference import ProFootballReference
from prediction_models.KNearest import KNearest
from prediction_models.XGBoost import XGBoost
from prediction_models.LinearRegression import LinearRegression
from prediction_models.RandomForest import RandomForest
from prediction_models.LogisticRegression import LogisticRegression
from prediction_models.GradientBoost import GradientBoost
from DataAggregate.DataAggregate import DataAggregate
from data_sources.OddsAPI import OddsAPI
#import pandas as pd
import sqlite3

#pfr = ProFootballReference()
da = DataAggregate("57d81d16fc6c36b35f5d8bb5893e1d3d")

#conn=sqlite3.connect('db/historical_data.db')

# k = KNearest(da.aggregates, da.prediction_set)
# k.predict_winner(da.prediction_set)
# 
# xg = XGBoost(da.aggregates, da.prediction_set)
# xg.predict_spread(da.prediction_set)
# 
# lr = LinearRegression(da.aggregates, da.prediction_set)
# lr.predict_spread(da.prediction_set)
# 
# rf = RandomForest(da.aggregates, da.prediction_set)
# rf.predict_spread(da.prediction_set)
# 
# lc = LogisticRegression(da.aggregates, da.prediction_set)
# lc.predict_winner(da.prediction_set)

gbfc = ['avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5','days_rest','elo_rating']
gb = GradientBoost(da.aggregates, da.prediction_set, gbfc, 'win')
gb.predict_winner(da.prediction_set)