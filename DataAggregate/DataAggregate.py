from data_sources.ProFootballReference import ProFootballReference
from data_sources.OddsAPI import OddsAPI
import pandas as pd

class DataAggregate:
	def __init__(self, odds_api_key, save_api_calls=True):
		self.team_performance_features = ['event_id', 'team', 'avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5', 'avg_point_differential_l5', 'days_rest']

		pfr = ProFootballReference()
		oa = OddsAPI(odds_api_key)
		self.team_performance = self.__add_opponent_stats_to_team_performance(pfr.load_team_performance_from_db())
		self.game_data = pfr.load_game_data_from_db()
		self.aggregates = self.__create_aggregates(self.game_data, self.team_performance)
		if(save_api_calls):
			print("Saving your OddsAPI tokens!")
			self.upcoming_games = pd.read_csv('cache/upcoming_games.csv')
		else:
			self.upcoming_games = oa.get_upcoming_for_pfr_prediction()
		self.prediction_set = self.__get_prediction_set(self.upcoming_games, self.__get_rolling_aggregates(self.team_performance).groupby('team').tail(1))
	
	def __create_aggregates(self, game_data, team_performance):
		team_performance_with_rolling_aggregates = self.__get_rolling_aggregates(team_performance)
		print(team_performance_with_rolling_aggregates.info())
		game_data = game_data.merge(
			team_performance_with_rolling_aggregates[self.team_performance_features],
			left_on = ['event_id', 'team_a'],
			right_on = ['event_id', 'team'],
			how = 'left'
		).drop('team', axis=1).rename(columns=self.__get_dict_for_feature_rename('team_a'))
		
		game_data = game_data.merge(
			team_performance_with_rolling_aggregates[self.team_performance_features],
			left_on = ['event_id', 'team_b'],
			right_on = ['event_id', 'team'],
			how = 'left'
		).drop('team', axis=1).rename(columns=self.__get_dict_for_feature_rename('team_b'))
		
		game_data['team_a_point_differential'] = game_data['team_b_points_scored'] - game_data['team_a_points_scored']
		
		return game_data
		
	def __get_most_recent_aggregates(self, team_performance):
		return team_performance.groupby('team').tail(1)
	
	def __get_rolling_aggregates(self, team_performance):
		# Calculate days rest (do this FIRST before rolling calcs)
		team_performance['date'] = pd.to_datetime(team_performance['date'])
		team_performance = team_performance.sort_values(['team', 'date'])
		team_performance['days_rest'] = team_performance.groupby('team')['date'].diff().dt.days.fillna(7)
		team_performance['days_rest'] = team_performance['days_rest'].clip(upper=21)
		
		team_performance['avg_points_scored_l5'] = team_performance.groupby('team')['points_scored'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
		
		team_performance['avg_pass_adjusted_yards_per_attempt_l5'] = team_performance.groupby('team')['pass_adjusted_yards_per_attempt'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
		
		team_performance['avg_rushing_yards_per_attempt_l5'] = team_performance.groupby('team')['rushing_yards_per_attempt'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
		
		team_performance['avg_turnovers_l5'] = team_performance.groupby('team')['turnovers'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)		
		
		team_performance['avg_penalty_yards_l5'] = team_performance.groupby('team')['penalty_yards'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)		
		
		team_performance['avg_sack_yards_lost_l5'] = team_performance.groupby('team')['sack_yards_lost'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)

		team_performance['avg_points_allowed_l5'] = team_performance.groupby('team')['opp_points_scored'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)

		team_performance['avg_pass_adjusted_yards_per_attempt_allowed_l5'] = team_performance.groupby('team')['opp_pass_adjusted_yards_per_attempt'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)

		team_performance['avg_rushing_yards_per_attempt_allowedl5'] = team_performance.groupby('team')['opp_rushing_yards_per_attempt'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
		
		team_performance['avg_turnovers_forced_l5'] = team_performance.groupby('team')['opp_turnovers'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)		
		
		team_performance['avg_sack_yards_gained_l5'] = team_performance.groupby('team')['opp_sack_yards_lost'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
		
		team_performance['point_differential'] = team_performance['points_scored'] - team_performance['opp_points_scored']
		team_performance['avg_point_differential_l5'] = team_performance.groupby('team')['point_differential'].transform(
			lambda x: x.rolling(5, min_periods=1).mean().shift(1)
		)
	
		return team_performance
	
	def __get_prediction_set(self, upcoming_games, recent_team_performance):
		print(recent_team_performance)
		upcoming_games = upcoming_games.merge(
			recent_team_performance[self.team_performance_features],
			left_on = ['home_team'],
			right_on = ['team'],
			how = 'left'
		).rename(columns=self.__get_dict_for_feature_rename('team_a'))

		upcoming_games = upcoming_games.merge(
			recent_team_performance[self.team_performance_features],
			left_on = ['away_team'],
			right_on = ['team'],
			how = 'left'
		).rename(columns=self.__get_dict_for_feature_rename('team_b'))
		
		upcoming_games['team_a_win'] = None
		upcoming_games['team_a_point_differential'] = None
		
		return upcoming_games
	
	def __add_opponent_stats_to_team_performance(self, team_performance):
		opponent_stats = team_performance.copy()
		opponent_stats = opponent_stats.add_prefix('opp_')
		team_performance = team_performance.merge(
			opponent_stats,
			left_on=['event_id', 'opponent'],
			right_on=['opp_event_id', 'opp_team'],
			how='left'
		).drop(['opp_event_id', 'opp_team', 'opp_opponent', 'opp_is_home'], axis=1)
		return team_performance
	
	def __get_dict_for_feature_rename(self, team_prefix):
		stat_columns = [col for col in self.team_performance_features if col not in ['event_id', 'team']]
		return {col: f'{team_prefix}_{col}' for col in stat_columns}