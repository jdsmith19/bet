from sklearn.linear_model import LogisticRegression as LogisticRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class LogisticRegression:
	def __init__(self, aggregate_data, prediction_set):
		self.target = 'win'
		self.feature_columns = ['avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5','days_rest','elo_rating']
		X_train = self.__prepare_features(aggregate_data)
		self.prediction_features = self.__prepare_features(prediction_set)
		self.classifier_model = self.__train_classifier(X_train)
		
	
	def __prepare_features(self, aggregate_data):		
		feature_columns = self.__get_team_specific_feature_columns()
		
		features = aggregate_data[feature_columns].copy()
		
		features = features.dropna()
		
		return features
	
	def __train_classifier(self, features):
		# Prep data
		X = features.drop(['team_a_win'], axis=1)
		y = features['team_a_win']
		
		# Split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		
		# Scale
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		
		# Train
		lr = LogisticRegressor(max_iter=1000, random_state=42)
		lr.fit(X_train_scaled, y_train)
		
		# Evaluate
		train_accuracy = lr.score(X_train_scaled, y_train)
		test_accuracy = lr.score(X_test_scaled, y_test)
		
		print("Logistic Regression Classifier trained")
		print(f"Training Accuracy: {train_accuracy:.3f}")
		print(f"Test Accuracy: {test_accuracy:.3f}")
		
		# Confidence calibration
		predictions = lr.predict(X_test_scaled)
		probabilities = lr.predict_proba(X_test_scaled)
		confidence = probabilities.max(axis=1)
		
		for threshold in [0.6, 0.7, 0.8]:
			mask = confidence > threshold
			if mask.sum() > 0:
				acc = (predictions[mask] == y_test[mask]).mean()
				pct = 100 * mask.sum() / len(predictions)
				print(f"Confidence >{threshold}: {mask.sum():4d} predictions ({pct:5.1f}%), {acc:.3f} accuracy")
		
		return {'model': lr, 'scaler': scaler}
		
	def predict_winner(self, prediction_set):
		feature_columns = self.__get_team_specific_feature_columns(prediction_columns=True)
		X_predict = prediction_set[feature_columns].copy()
		
		X_predict_scaled = self.classifier_model['scaler'].transform(X_predict)
		predictions = self.classifier_model['model'].predict(X_predict_scaled)
		probabilities = self.classifier_model['model'].predict_proba(X_predict_scaled)
		
		# Add to your dataframe
		results = prediction_set[['home_team', 'away_team']].copy()
		results['predicted_winner'] = predictions  # 0 or 1
		results['predicted_winner_team'] = results.apply(
			lambda row: row['home_team'] if predictions[row.name] == 1 else row['away_team'],
			axis=1
		)
		results['team_a_win_probability'] = probabilities[:, 1]
		results['confidence'] = probabilities.max(axis=1)
		
		# Show results
		print(results)
		
		# Filter to high confidence picks
		high_confidence = results[results['confidence'] > 0.7]
		if len(high_confidence) > 0:
			print("\nHigh confidence picks:")
			print(high_confidence)
		else:
			print("\nNo high confidence picks")
		
		return results
		
	def __get_team_specific_feature_columns(self, prediction_columns=False):	
		team_specific_feature_columns = []
		if(not prediction_columns):
			team_specific_feature_columns.append("team_a_" + self.target)
		for col in self.feature_columns:
			team_specific_feature_columns.append("team_a_" + col)
			team_specific_feature_columns.append("team_b_" + col)
		return team_specific_feature_columns