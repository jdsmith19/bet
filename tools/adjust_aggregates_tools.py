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
		da: A DataAggregate object to be modified, optional. If not passed, a fresh DataAggregate object will be instantiated.
	Returns:
		DataAggregate: A DataAggregate object with updated values
	"""
	if not da:
		print(f"Loading data aggregates")
		da = DataAggregate(config.odds_api_key)
	print(f"Adjusting features...")
	for adj in adjustments:
		home_cols_to_update = [col for col in da.prediction_set.columns if (adj['feature'] in col and '_away' not in col)]
		away_cols_to_update = [col for col in da.prediction_set.columns if (adj['feature'] in col and '_home' not in col)]
		da.prediction_set.loc[da.prediction_set['home_team'] == adj['team_name'], home_cols_to_update] *= adj['adjustment_percentage']
		da.prediction_set.loc[da.prediction_set['away_team'] == adj['team_name'], away_cols_to_update] *= adj['adjustment_percentage']
	
	return da