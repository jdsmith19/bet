import pandas as pd

class PredictionModel:
	def __init__(self, data_aggregate, target, feature_columns):
		self.target = target
		self.feature_columns = feature_columns
		self.team_specific_feature_columns = self.__get_team_specific_feature_columns(self.feature_columns)
		self.training_features = self.__prepare_features(data_aggregate.aggregates, prediction=False)
		self.prediction_features = self.__prepare_features(data_aggregate.prediction_set, prediction=True)

	def __prepare_features(self, aggregate_data, prediction=False):
		feature_columns = self.team_specific_feature_columns
		if(prediction):
			feature_columns.remove("team_a_" + self.target)
		features = aggregate_data[feature_columns].copy()
		features = features.dropna()
		return features
	
	def __get_team_specific_feature_columns(self, prediction_columns=False):
		team_specific_feature_columns = []
		team_specific_feature_columns.append("team_a_" + self.target)
		for col in self.feature_columns:
			if "_is_away" not in col:
				team_specific_feature_columns.append("team_a_" + col)
			if "_is_home" not in col:
				team_specific_feature_columns.append("team_b_" + col)
		return team_specific_feature_columns