from .PredictionModel import PredictionModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
from helpers.Lookup import Lookup

class XGBoost(PredictionModel):
	def __init__(self, data_aggregate, target, feature_columns):
		super().__init__(data_aggregate, target, feature_columns)
		self.model_output = { 'model_name': 'XGBoost', 'target': target }
		self.lookup = Lookup()
		self.xgb_regressor = self.__train_model(self.training_features, test = True)
		self.xgb_regressor = self.__train_model(self.training_features)
			
	def __train_model(self, features, test = False):
		# Prep data
		X = features.drop(['team_a_' + self.target], axis=1)
		y = features['team_a_' + self.target]
		
		if(test):
			X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Train the model
		xgb = XGBRegressor(
			n_estimators = 100,
			max_depth = 5,
			learning_rate = 0.1,
			random_state = 42
		)
		xgb.fit(X, y)
		
		if(test):
			predictions = xgb.predict(X_test)
			self.model_output['mean_absolute_error'] = round(mean_absolute_error(y_test, predictions), 4)
			self.model_output['root_mean_squared_error'] = round(float(np.sqrt(mean_squared_error(y_test, predictions))), 4)
			importance = pd.DataFrame({
				'feature': self.team_specific_feature_columns,
				'importance': xgb.feature_importances_
			}).sort_values('importance', ascending=False)
			self.model_output['feature_importance']  = {
				feature: round(imp, 4)
				for feature, imp in zip(importance["feature"], importance["importance"])
			}
		
		return xgb
	
	def predict_spread(self, prediction_set):	
		X_predict = prediction_set[self.team_specific_feature_columns].copy()
		spread_predictions = self.xgb_regressor.predict(X_predict)

		# Add to dataframe for readability
		results = prediction_set[['home_team', 'away_team']].copy()
		results['home_team'] = results['home_team'].map(self.lookup.pfr_team_to_odds_api_team)
		results['away_team'] = results['away_team'].map(self.lookup.pfr_team_to_odds_api_team)
		results['predicted_spread'] = spread_predictions
		results['predicted_winner'] = results.apply(
			lambda row: f"{ row['home_team'] }"
			if row['predicted_spread'] < 0
			else f"{ row['away_team'] }",
			axis=1
		)
		results['prediction_text'] = results.apply(lambda row: f"{ row['predicted_winner'] } by { round(abs(row['predicted_spread'])) }", axis=1)
		results_obj = results.to_dict(orient="records")
		self.model_output['results'] = results_obj
		return results_obj