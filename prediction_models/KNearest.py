from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class KNearest:
	def __init__(self, aggregate_data, prediction_set):
		self.target = 'win'
		self.feature_columns = ['avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5']
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
		
		scaler = StandardScaler()
		X_scaled = scaler.fit_transform(X)
		
		knn = KNeighborsClassifier(n_neighbors=25)
		knn.fit(X_scaled, y)
		
		print("KNearestClassifier model trained")
		train_accuracy = knn.score(X_scaled, y)
		print(f"Training Accuracy: {train_accuracy:.3f}")
		
		return { 'model': knn, 'scaler': scaler }
		
	def predict_winner(self, prediction_set):
						
		feature_columns = self.__get_team_specific_feature_columns(prediction_columns=True)

		X_predict = prediction_set[feature_columns].copy()
		
		X_predict_scaled = self.classifier_model['scaler'].transform(X_predict)
		predictions = self.classifier_model['model'].predict(X_predict_scaled)
		probabilities = self.classifier_model['model'].predict_proba(X_predict_scaled)
		
		# Add to your dataframe
		prediction_set['predicted_winner'] = predictions  # 0 or 1
		prediction_set['team_a_win_probability'] = probabilities[:, 1]  # Probability team_a wins
		prediction_set['confidence'] = probabilities.max(axis=1)  # Max probability
		
		# Show results
		results = prediction_set[['home_team', 'away_team', 'predicted_winner', 
								   'team_a_win_probability', 'confidence']]
		print(results)
		
		# Filter to high confidence picks
		high_confidence = results[results['confidence'] > 0.7]
		print("\nHigh confidence picks:")
		print(high_confidence)

	def __get_team_specific_feature_columns(self, prediction_columns=False):
		team_specific_feature_columns = []
		if(not prediction_columns):
			team_specific_feature_columns.append("team_a_" + self.target)
		for col in self.feature_columns:
			team_specific_feature_columns.append("team_a_" + col)
			team_specific_feature_columns.append("team_b_" + col)
		return team_specific_feature_columns