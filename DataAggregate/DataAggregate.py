from data_sources.ProFootballReference import ProFootballReference
from data_sources.OddsAPI import OddsAPI
import pandas as pd

class DataAggregate:
	def __init__(self, odds_api_key, save_api_calls=True):
		self.team_performance_features = ['event_id', 'team', 'avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l5', 'avg_turnovers_l5', 'avg_penalty_yards_l5', 'avg_sack_yards_lost_l5', 'avg_points_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowedl5', 'avg_turnovers_forced_l5', 'avg_sack_yards_gained_l5', 'avg_point_differential_l5', 'days_rest', 'elo_rating', 'rpi_rating']

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
			self.upcoming_games.to_csv('cache/upcoming_games.csv', index=False)
		self.prediction_set = self.__get_prediction_set(self.upcoming_games, self.__get_rolling_aggregates(self.team_performance).groupby('team').tail(1))
	
	def __create_aggregates(self, game_data, team_performance):
		team_performance_with_rolling_aggregates = self.__get_rolling_aggregates(team_performance)
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
		team_performance = self.__calculate_elo(team_performance, k=20, initial_elo=1500)
		team_performance = self.__calculate_rpi(team_performance)
		
		for interval in [3, 5, 7]:
			team_performance['avg_points_scored_l' + str(interval)] = team_performance.groupby('team')['points_scored'].transform(
				lambda x: x.rolling(5, min_periods=1).mean().shift(1)
			)
			
			team_performance['avg_pass_adjusted_yards_per_attempt_l' + str(interval)] = team_performance.groupby('team')['pass_adjusted_yards_per_attempt'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
			
			team_performance['avg_rushing_yards_per_attempt_l' + str(interval)] = team_performance.groupby('team')['rushing_yards_per_attempt'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
			
			team_performance['avg_turnovers_l' + str(interval)] = team_performance.groupby('team')['turnovers'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)		
			
			team_performance['avg_penalty_yards_l' + str(interval)] = team_performance.groupby('team')['penalty_yards'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)		
			
			team_performance['avg_sack_yards_lost_l' + str(interval)] = team_performance.groupby('team')['sack_yards_lost'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
	
			team_performance['avg_points_allowed_l' + str(interval)] = team_performance.groupby('team')['opp_points_scored'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
	
			team_performance['avg_pass_adjusted_yards_per_attempt_allowed_l' + str(interval)] = team_performance.groupby('team')['opp_pass_adjusted_yards_per_attempt'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
	
			team_performance['avg_rushing_yards_per_attempt_allowed_l' + str(interval)] = team_performance.groupby('team')['opp_rushing_yards_per_attempt'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
			
			team_performance['avg_turnovers_forced_l' + str(interval)] = team_performance.groupby('team')['opp_turnovers'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)		
			
			team_performance['avg_sack_yards_gained_l' + str(interval)] = team_performance.groupby('team')['opp_sack_yards_lost'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)
			
			team_performance['point_differential'] = team_performance['points_scored'] - team_performance['opp_points_scored']
			team_performance['avg_point_differential_l' + str(interval)] = team_performance.groupby('team')['point_differential'].transform(
				lambda x: x.rolling(interval, min_periods=1).mean().shift(1)
			)

		return team_performance
	
	def __get_prediction_set(self, upcoming_games, recent_team_performance):
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
	
	def __calculate_elo(self, team_performance, k=20, initial_elo=1500):
		team_performance = team_performance.sort_values(['team', 'date'])
		
		# Initialize ELO
		elo_dict = { team: initial_elo for team in team_performance['team'].unique() }
		current_season = None
		team_performance['elo_rating'] = 0.0
		team_performance['opp_elo_rating'] = 0.0
		
		for idx, row in team_performance.iterrows():
			team = row['team']
			opp = row['opponent']
			
			team_elo = elo_dict[team]
			opp_elo = elo_dict.get(opp, initial_elo)
			
			if row['season'] != current_season:
				current_season = row['season']
				# Regress all teams toward mean
				for team in elo_dict:
					elo_dict[team] = elo_dict[team] * 0.67 + 1500 * 0.33
					elo_dict[opp] = elo_dict[opp] * 0.67 + 1500 * 0.33

			
			team_performance.at[idx, 'elo_rating'] = team_elo
			team_performance.at[idx, 'opp_elo_rating'] = opp_elo
			
			expected = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
			actual = row['win']
			elo_dict[team] = team_elo + k * (actual - expected)
			
		return team_performance
	
	def __calculate_rpi(self, team_performance):
		"""
		Calculate RPI for each team at each game.
		RPI = (0.25 × WP) + (0.50 × OWP) + (0.25 × OOWP)
		"""
		df = team_performance.sort_values(['date', 'event_id']).reset_index(drop=True)
		
		# Initialize tracking for each team
		team_stats = {}  # {team: {'wins': 0, 'games': 0, 'opponents': []}}
		
		# Store RPI values
		rpi_values = []
		
		# Process each game (both rows)
		processed_events = set()
		
		for idx, row in df.iterrows():
			team = row['team']
			opponent = row['opponent']
			event_id = row['event_id']
			
			# Initialize team if first appearance
			if team not in team_stats:
				team_stats[team] = {'wins': 0, 'games': 0, 'opponents': []}
			
			# Calculate RPI BEFORE this game
			team_rpi = self.__compute_rpi_value(team, team_stats)
			rpi_values.append(team_rpi)
			
			# Update records AFTER this game (only once per event)
			if event_id not in processed_events:
				processed_events.add(event_id)
				
				# Update both teams
				for t, opp, score, opp_score in [
					(row['team'], row['opponent'], row['points_scored'], row['opp_points_scored']),
					(row['opponent'], row['team'], row['opp_points_scored'], row['points_scored'])
				]:
					if t not in team_stats:
						team_stats[t] = {'wins': 0, 'games': 0, 'opponents': []}
					
					# Update wins
					if score > opp_score:
						team_stats[t]['wins'] += 1
					elif score == opp_score:
						team_stats[t]['wins'] += 0.5
					
					# Update games and opponents
					team_stats[t]['games'] += 1
					team_stats[t]['opponents'].append(opp)
		
		# Add RPI column
		df['rpi_rating'] = rpi_values
		
		return df
		
	def __compute_rpi_value(self, team, team_stats):
		"""
		Compute RPI for a single team at a point in time.
		Returns 0.5 (neutral) if team has no games yet.
		"""
		if team not in team_stats or team_stats[team]['games'] == 0:
			return 0.5
		
		stats = team_stats[team]
		
		# 1. WP (Winning Percentage)
		wp = stats['wins'] / stats['games']
		
		# 2. OWP (Opponent Winning Percentage)
		if len(stats['opponents']) == 0:
			owp = 0.5
		else:
			opponent_wps = []
			for opp in stats['opponents']:
				if opp not in team_stats or team_stats[opp]['games'] == 0:
					opponent_wps.append(0.5)
				else:
					opp_wp = team_stats[opp]['wins'] / team_stats[opp]['games']
					opponent_wps.append(opp_wp)
			owp = sum(opponent_wps) / len(opponent_wps)
		
		# 3. OOWP (Opponent's Opponent Winning Percentage)
		if len(stats['opponents']) == 0:
			oowp = 0.5
		else:
			oowp_values = []
			for opp in stats['opponents']:
				if opp not in team_stats or len(team_stats[opp]['opponents']) == 0:
					oowp_values.append(0.5)
				else:
					# Get opponent's opponents' winning percentages
					oo_wps = []
					for oo in team_stats[opp]['opponents']:
						if oo not in team_stats or team_stats[oo]['games'] == 0:
							oo_wps.append(0.5)
						else:
							oo_wp = team_stats[oo]['wins'] / team_stats[oo]['games']
							oo_wps.append(oo_wp)
					
					if oo_wps:
						oowp_values.append(sum(oo_wps) / len(oo_wps))
					else:
						oowp_values.append(0.5)
			
			oowp = sum(oowp_values) / len(oowp_values)
		
		# Calculate final RPI
		rpi = (0.25 * wp) + (0.50 * owp) + (0.25 * oowp)
		
		return rpi