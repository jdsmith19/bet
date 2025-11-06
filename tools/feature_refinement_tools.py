from model_feature_evaluation import evaluate_model_with_features
import time

def train_and_evaluate_model(model_name: str, features: list[str]) -> dict:
	"""
	Train a model with specified features and return performance metrics.
	
	Args:
		model_name: One of ['XGBoost', 'LinearRegression', 'RandomForest', 'LogisticRegression', 'KNearest']
		features: List of feature names to include
	
	Returns:
		{
			'model_name': str,
			'target': str,
			'features_used': list[str],
			'mae': float (regression only),
			'rmse': float (regression only),
			'train_accuracy': float (classification only),
			'test_accuracy': float (classification only),
			'feature_importance': dict (regression only),
			'confidence_intervals': dict (classification only),
			'train_time_seconds': 
		}
	"""
	print(features)
	MODEL_CONFIG = {
		'XGBoost': { 
			'target': 'point_differential',
			'is_regression': True
		},
		'LinearRegression': {
			'target': 'point_differential',
			'is_regression': True		
		},
		'RandomForest': {
			'target': 'point_differential',
			'is_regression': True
		},
		'LogisticRegression': {
			'target': 'win',
			'is_regression': False
		},
		'KNearest': {
			'target': 'win',
			'is_regression': False
		}
	}
	
	if model_name not in MODEL_CONFIG:
		raise ValueError(f"Invalid model name. Must be one of { list(MODEL_CONFIG.keys()) }")
	
	config = MODEL_CONFIG[model_name]
	target = config['target']
	
	VALID_FEATURES = [
		'days_rest',
		'rpi_rating',
		'elo_rating'
	]
	for feature in ['points_scored', 'pass_adjusted_yards_per_attempt', 'rushing_yards_per_attempt', 'turnovers', 'penalty_yards', 'sack_yards_lost', 'points_allowed', 'pass_adjusted_yards_per_attempt_allowed', 'rushing_yards_per_attempt_allowed', 'turnovers_forced', 'sack_yards_gained', 'point_differential']:
		for interval in [3, 5, 7]:
			VALID_FEATURES.append("avg_" + feature + "_l" + str(interval))
		
	invalid_features = [f for f in features if f not in VALID_FEATURES]
	if invalid_features:
		raise ValueError(f"Invalid features: { invalid_features }")
	
	start_time = time.time()
	
	try:
		model_output = evaluate_model_with_features(model_name, features)
	
		train_time = time.time() - start_time
		
		result['train_time_in_seconds'] = train_time
		
		return result
	
	except Exception as e:
		return {
			'model_name': model_name,
			'target': target,
			'features_used': features,
			'error': str(e),
			'train_time_in_seconds': time.time() - start_time
		}