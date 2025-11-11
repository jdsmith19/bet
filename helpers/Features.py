class Features:
	def __init__(self):
		self.base_features = features = [
			'elo_rating',
			'rpi_rating',
			'days_rest',
			'avg_points_scored',
			'avg_pass_adjusted_yards_per_attempt',
			'avg_rushing_yards_per_attempt',
			'avg_turnovers',
			'avg_penalty_yards',
			'avg_sack_yards_lost',
			'avg_points_allowed',
			'avg_pass_adjusted_yards_per_attempt_allowed',
			'avg_rushing_yards_per_attempt_allowed',
			'avg_turnovers_forced',
			'avg_sack_yards_gained',
			'avg_point_differential'
		]
		self.extended_features = self.__get_extended_features()
	
	def __get_extended_features(self):
		ef = []
		for feature in self.base_features:
			for interval in [3, 5, 7]:
				ef.append(f"avg_{ feature }_l{ interval }")
			for location in ['home', 'away']:
				ef.append(f"avg_{ feature }_{ location }")
		return ef