#from data_sources.ProFootballReference import ProFootballReference
from prediction_models.KNearest import KNearest
from prediction_models.XGBoost import XGBoost
from prediction_models.LinearRegression import LinearRegression
from prediction_models.RandomForest import RandomForest
from prediction_models.LogisticRegression import LogisticRegression
from DataAggregate.DataAggregate import DataAggregate
from data_sources.OddsAPI import OddsAPI
#import pandas as pd
import sqlite3

#pfr = ProFootballReference()
da = DataAggregate("57d81d16fc6c36b35f5d8bb5893e1d3d")

conn=sqlite3.connect('db/historical_data.db')

#game_data = pfr.load_game_data_from_db()
#team_performance = pfr.load_team_performance_from_db()

#oa = OddsAPI("57d81d16fc6c36b35f5d8bb5893e1d3d")
#upcoming = oa.get_upcoming_for_pfr_prediction()

k = KNearest(da.aggregates, da.prediction_set)
k.predict_winner(da.prediction_set)

xg = XGBoost(da.aggregates, da.prediction_set)
xg.predict_spread(da.prediction_set)

lr = LinearRegression(da.aggregates, da.prediction_set)
lr.predict_spread(da.prediction_set)

rf = RandomForest(da.aggregates, da.prediction_set)
rf.predict_spread(da.prediction_set)

** SUCKS **
lc = LogisticRegression(da.aggregates, da.prediction_set)
lc.predict_winner(da.prediction_set)