from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

class RandomForest:
	def __init__(self, aggregate_data, prediction_set):
		self.target = 'point_differential'
		self.feature_columns = ['avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_point_differential_l5', 'days_rest', 'elo_rating']
		X_train = self.__prepare_features(aggregate_data)
		self.prediction_features = self.__prepare_features(prediction_set)
		self.rf_regressor = self.__train_regressor(X_train)
	
	def __prepare_features(self, aggregate_data):		
		feature_columns = self.__get_team_specific_feature_columns()
		features = aggregate_data[feature_columns].copy()
		features = features.dropna()
		return features

	def __get_team_specific_feature_columns(self, prediction_columns=False):
		team_specific_feature_columns = []
		if(not prediction_columns):
			team_specific_feature_columns.append("team_a_" + self.target)
		for col in self.feature_columns:
			team_specific_feature_columns.append("team_a_" + col)
			team_specific_feature_columns.append("team_b_" + col)
		return team_specific_feature_columns
		
	def __train_regressor(self, features):
		# Prep data
		X = features.drop(['team_a_point_differential'], axis=1)
		y = features['team_a_point_differential']
		
		# Split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Train LogistRegressor
		rf = RandomForestRegressor()
		rf.fit(X_train, y_train)
		
		# Evaluate
		predictions = rf.predict(X_test)
		mae = mean_absolute_error(y_test, predictions)
		rmse = np.sqrt(mean_squared_error(y_test, predictions))
		
		print(f"***Random Forest Model***")
		print(f"Mean Absolute Error: {mae:.2f} points")
		print(f"Root Mean Squared Error: {rmse:.2f} points")
		
		# Show predictions
		results = pd.DataFrame({
			'actual_margin': y_test,
			'predicted_margin': predictions,
			'difference': np.abs(y_test - predictions)
		})
		print(results.head(10))
		
		# Feature importance
		importance = pd.DataFrame({
			'feature': self.__get_team_specific_feature_columns(self.feature_columns),
			'importance': rf.feature_importances_
		}).sort_values('importance', key=abs, ascending=False)
		print("\nFeature Importances:")
		print(importance)
		
		return rf
	
	def predict_spread(self, prediction_set):
					
		feature_columns = self.__get_team_specific_feature_columns(prediction_columns=True)
	
		X_predict = prediction_set[feature_columns].copy()
		spread_predictions = self.rf_regressor.predict(X_predict)

		# Add to dataframe for readability
		results = prediction_set[['home_team', 'away_team']].copy()
		results['predicted_spread'] = spread_predictions
		results['prediction'] = results.apply(
			lambda row: f"{row['home_team']} by {abs(row['predicted_spread']):.1f}" 
			if row['predicted_spread'] < 0 
			else f"{row['away_team']} by {abs(row['predicted_spread']):.1f}",
			axis=1
		)
		print(results)
		return results
