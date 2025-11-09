from DataAggregate.DataAggregate import DataAggregate
from helpers.Lookup import Lookup
import time
import json
import config
import traceback

def adjust_data_aggregates(adjustments: list, da: DataAggregate = None) -> DataAggregate:
	"""
	Makes adjustments to the prediction set of the data aggregates object
	
	Args:
		adjustments: A list of adjustments to make in the format of { 'team_name': team name, 'feature', feature name, 'adjustment_percentage': float value for the amount to adjust the aggregate value}
	
	Returns:
		DataAggregate: A DataAggregate object with updated values
	"""
	print(f"Loading data aggregates")
	if not da:
		da = DataAggregate(config.odds_api_key)
	print(f"Adjusting features")
	for adj in adjustments:
		home_cols_to_update = [col for col in da.prediction_set.columns if (adj['feature'] in col and '_away' not in col)]
		away_cols_to_update = [col for col in da.prediction_set.columns if (adj['feature'] in col and '_home' not in col)]
		#print(home_cols_to_update)
		#print(away_cols_to_update)
		#home_rows = da.prediction_set.loc[da.prediction_set['home_team'] == adj['team_name']]
		#away_rows = da.prediction_set.loc[da.prediction_set['away_team'] == adj['team_name']]
		#print(f"--- Home Rows ---")
		#print(home_rows[home_cols_to_update])
		#print(f"--- Away Rows ---")
		#print(away_rows[away_cols_to_update])
		da.prediction_set.loc[da.prediction_set['home_team'] == adj['team_name'], home_cols_to_update] *= adj['adjustment_percentage']
		da.prediction_set.loc[da.prediction_set['away_team'] == adj['team_name'], away_cols_to_update] *= adj['adjustment_percentage']
		#print(f"--- Adjusted Home Rows ---")
		#print(home_rows[home_cols_to_update])
		#print(f"--- Adjusted Away Rows ---")
		#print(away_rows[away_cols_to_update])

	
	return da