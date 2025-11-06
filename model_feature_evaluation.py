from DataAggregate.DataAggregate import DataAggregate
from prediction_models.XGBoost import XGBoost
from prediction_models.LinearRegression import LinearRegression
from prediction_models.RandomForest import RandomForest
from prediction_models.LogisticRegression import LogisticRegression
from prediction_models.KNearest import KNearest
import config

def evaluate_model_with_features(model_name, feature_list):
	da = DataAggregate(config.odds_api_key)
	
	if(model_name == 'XGBoost'):
		model = XGBoost(da, 'predicted_spread', feature_list)
	
	elif(model_name == 'LinearRegression'):
		model = LinearRegression(da, 'predicted_spread', feature_list)
	
	elif(model_name == 'RandomForest'):
		model = RandomForest(da, 'predicted_spread', feature_list)
	
	elif(model_name == 'LogisticRegression'):
		model = LogisticRegression(da, 'win', feature_list)
	
	elif(model_name == 'KNearest'):
		model = KNearest(da, 'win', feature_list)
	
	return model.model_output