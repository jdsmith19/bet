from .PredictionModel import PredictionModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from helpers.Lookup import Lookup

class KNearest(PredictionModel):
	def __init__(self, data_aggregate, target, feature_columns):
		super().__init__(data_aggregate, target, feature_columns)
		self.model_output = { 'model_name': 'KNearest', 'target': target }
		self.lookup = Lookup()
		self.kn_classifier = self.__train_model(self.training_features, test = True)
		self.kn_classifier = self.__train_model(self.training_features)
			
	def __train_model(self, features, test = False):
		# Prep data
		X = features.drop(['team_a_' + self.target], axis=1)
		y = features['team_a_' + self.target]
		
		if(test):
			X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
				
		# Scale
		scaler = StandardScaler()
		X = scaler.fit_transform(X)

		# Train the model
		kn = KNeighborsClassifier()
		kn.fit(X, y)
		
		if(test):
			X_test = scaler.fit_transform(X_test)
			
			# Evaluate
			self.model_output['train_accuracy'] = kn.score(X, y)
			self.model_output['test_accuracy'] = kn.score(X_test, y_test)
			
			# Confidence Calibration
			predictions = kn.predict(X_test)
			probabilities = kn.predict_proba(X_test)		
			confidence = probabilities.max(axis=1)

			self.model_output['confidence_intervals'] = []
			for threshold in [0.6, 0.7, 0.8]:
				mask = confidence > threshold
				if mask.sum() > 0:
					acc = (predictions[mask] == y_test[mask]).mean()
					pct = 100 * mask.sum() / len(predictions)
					self.model_output['confidence_intervals'].append(
						{ f"confidence_greater_than_{ threshold }": { 
							"count_predictions": int(mask.sum()),
							"accuracy": round(float(acc), 4)
						}
					})
		return {'model': kn, 'scaler': scaler}
	
	def predict_winner(self, prediction_set):	
		X_predict = prediction_set[self.team_specific_feature_columns].copy()
		X_predict = self.kn_classifier['scaler'].transform(X_predict)
		win_predictions = self.kn_classifier['model'].predict(X_predict)
		probabilities = self.kn_classifier['model'].predict_proba(X_predict)

		# Add to dataframe for readability
		results = prediction_set[['home_team', 'away_team']].copy()
		results['home_team'] = results['home_team'].map(self.lookup.pfr_team_to_odds_api_team)
		results['away_team'] = results['away_team'].map(self.lookup.pfr_team_to_odds_api_team)
		results['predicted_winner'] = results.apply(
			lambda row: row['home_team'] if win_predictions[row.name] == 1 else row['away_team'],
			axis=1
		)
		results['confidence'] = probabilities.max(axis=1)
		results_obj = results.to_dict(orient="records")
		self.model_output['results'] = results_obj
		return results_obj