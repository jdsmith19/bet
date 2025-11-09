from DataAggregate.DataAggregate import DataAggregate
from prediction_models.XGBoost import XGBoost
from prediction_models.LinearRegression import LinearRegression
from prediction_models.RandomForest import RandomForest
from prediction_models.LogisticRegression import LogisticRegression
from prediction_models.KNearest import KNearest
import time
import json
import config
import traceback

def get_upcoming_predictions(adjusted_aggregates = None) -> dict:
	"""
	Gets predictions for upcoming games based on the current best features identified in the current feature_optimization_results.json file.
	
	Args:
		adjusted_aggregates: An optional adjusted DataAggregates that can be passed to account for injuries or other factors
	
	Returns:
		List[dict]: A list of model prediction results, where each dict contains:
			- model_name (str): Name of the model (XGBoost, LinearRegression, RandomForest, LogisticRegression, or KNearest)
			- target (str): Prediction target ('point_differential' or 'win')
			
			For regression models (XGBoost, LinearRegression, RandomForest):
				- mean_absolute_error (float): MAE on test set
				- root_mean_squared_error (float): RMSE on test set
				- feature_importance (dict): Feature importance scores (XGBoost, RandomForest only)
				- feature_coefficients (dict): Feature coefficients (LinearRegression only)
			
			For classification models (LogisticRegression, KNearest):
				- train_accuracy (float): Accuracy on training set
				- test_accuracy (float): Accuracy on test set
				- confidence_intervals (list[dict]): Accuracy at different confidence thresholds
			
			All models:
				- results (list[dict]): Game predictions, each containing:
					- home_team (str): Home team name
					- away_team (str): Away team name
					- predicted_winner (str): Predicted winning team
					- predicted_spread (float): Predicted point differential (regression models only)
					- prediction_text (str): Human-readable prediction (regression models only)
					- confidence (float): Prediction confidence score (classification models only)
	"""
	
	with open("feature_optimization_results.json", "r") as f:
		feature_optimization_results = json.load(f)
	
	if(adjusted_aggregates):
		print(f"Running predictions with adjusted aggregates")
		da = adjusted_aggregates
	else:
		print(f"Loading data aggregates")
		da = DataAggregate(config.odds_api_key)
	
	start_time = time.time()
	
	predictions = []
	
	try:
		for best_combo in feature_optimization_results['best_results']:
			if best_combo == 'XGBoost':
				xgb = XGBoost(da, feature_optimization_results['best_results'][best_combo]['target'], feature_optimization_results['best_results'][best_combo]['features_used'])
				results = xgb.predict_spread(da.prediction_set)
				predictions.append(xgb.model_output)

			elif best_combo == 'LinearRegression':
				lr = LinearRegression(da, feature_optimization_results['best_results'][best_combo]['target'], feature_optimization_results['best_results'][best_combo]['features_used'])
				results = lr.predict_spread(da.prediction_set)
				predictions.append(lr.model_output)

			elif best_combo == 'RandomForest':
				rf = RandomForest(da, feature_optimization_results['best_results'][best_combo]['target'], feature_optimization_results['best_results'][best_combo]['features_used'])
				results = rf.predict_spread(da.prediction_set)
				predictions.append(rf.model_output)

			elif best_combo == 'LogisticRegression':
				lg = LogisticRegression(da, feature_optimization_results['best_results'][best_combo]['target'], feature_optimization_results['best_results'][best_combo]['features_used'])
				results = lg.predict_winner(da.prediction_set)
				predictions.append(lg.model_output)

			elif best_combo == 'KNearest':
				kn = KNearest(da, feature_optimization_results['best_results'][best_combo]['target'], feature_optimization_results['best_results'][best_combo]['features_used'])
				results = kn.predict_winner(da.prediction_set)
				predictions.append(kn.model_output)
				
		return predictions
	
	except Exception as e:
		return traceback.print_exc()