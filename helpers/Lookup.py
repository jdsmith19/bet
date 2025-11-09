class Lookup:
	def __init__(self):
		pass
	
	def odds_api_team_to_pfr_team(self, team):
		team_lookup_dict = {
			'Arizona Cardinals': 'crd',
			'Atlanta Falcons': 'atl',
			'Baltimore Ravens': 'rav',
			'Buffalo Bills': 'buf',
			'Carolina Panthers': 'car',
			'Chicago Bears': 'chi',
			'Cincinnati Bengals': 'cin',
			'Cleveland Browns': 'cle',
			'Dallas Cowboys': 'dal',
			'Denver Broncos': 'den',
			'Detroit Lions': 'det',
			'Green Bay Packers': 'gnb',
			'Houston Texans': 'htx',
			'Indianapolis Colts': 'clt',
			'Jacksonville Jaguars': 'jax',
			'Kansas City Chiefs': 'kan',
			'Las Vegas Raiders': 'rai',
			'Los Angeles Chargers': 'sdg',
			'Los Angeles Rams': 'ram',
			'Miami Dolphins': 'mia',
			'Minnesota Vikings': 'min',
			'New England Patriots': 'nwe',
			'New Orleans Saints': 'nor',
			'New York Giants': 'nyg',
			'New York Jets': 'nyj',
			'Philadelphia Eagles': 'phi',
			'Pittsburgh Steelers': 'pit',
			'San Francisco 49ers': 'sfo',
			'Seattle Seahawks': 'sea',
			'Tampa Bay Buccaneers': 'tam',
			'Tennessee Titans': 'oti',
			'Washington Commanders': 'was',
		}
		
		return team_lookup_dict[team]
	
	def pfr_team_to_odds_api_team(self, team):
		team_lookup_dict = {
			'crd': 'Arizona Cardinals',
			'atl': 'Atlanta Falcons',
			'rav': 'Baltimore Ravens',
			'buf': 'Buffalo Bills',
			'car': 'Carolina Panthers',
			'chi': 'Chicago Bears',
			'cin': 'Cincinnati Bengals',
			'cle': 'Cleveland Browns',
			'dal': 'Dallas Cowboys',
			'den': 'Denver Broncos',
			'det': 'Detroit Lions',
			'gnb': 'Green Bay Packers',
			'htx': 'Houston Texans',
			'clt': 'Indianapolis Colts',
			'jax': 'Jacksonville Jaguars',
			'kan': 'Kansas City Chiefs',
			'rai': 'Las Vegas Raiders',
			'sdg': 'Los Angeles Chargers',
			'ram': 'Los Angeles Rams',
			'mia': 'Miami Dolphins',
			'min': 'Minnesota Vikings',
			'nwe': 'New England Patriots',
			'nor': 'New Orleans Saints',
			'nyg': 'New York Giants',
			'nyj': 'New York Jets',
			'phi': 'Philadelphia Eagles',
			'pit': 'Pittsburgh Steelers',
			'sfo': 'San Francisco 49ers',
			'sea': 'Seattle Seahawks',
			'tam': 'Tampa Bay Buccaneers',
			'oti': 'Tennessee Titans',
			'was': 'Washington Commanders'
		}
		
		return team_lookup_dict[team]
	
	def team_name_to_espn_code(self, team_name):
		"""
		Map full team names to ESPN team codes for depth chart scraping
		
		Args:
			team_name: Full team name (e.g., 'Buffalo Bills')
		
		Returns:
			ESPN team code (e.g., 'buf')
		"""
		team_lookup_dict = {
			'Arizona Cardinals': 'ari',
			'Atlanta Falcons': 'atl',
			'Baltimore Ravens': 'bal',
			'Buffalo Bills': 'buf',
			'Carolina Panthers': 'car',
			'Chicago Bears': 'chi',
			'Cincinnati Bengals': 'cin',
			'Cleveland Browns': 'cle',
			'Dallas Cowboys': 'dal',
			'Denver Broncos': 'den',
			'Detroit Lions': 'det',
			'Green Bay Packers': 'gb',
			'Houston Texans': 'hou',
			'Indianapolis Colts': 'ind',
			'Jacksonville Jaguars': 'jax',
			'Kansas City Chiefs': 'kc',
			'Las Vegas Raiders': 'lv',
			'Los Angeles Chargers': 'lac',
			'Los Angeles Rams': 'lar',
			'Miami Dolphins': 'mia',
			'Minnesota Vikings': 'min',
			'New England Patriots': 'ne',
			'New Orleans Saints': 'no',
			'New York Giants': 'nyg',
			'New York Jets': 'nyj',
			'Philadelphia Eagles': 'phi',
			'Pittsburgh Steelers': 'pit',
			'San Francisco 49ers': 'sf',
			'Seattle Seahawks': 'sea',
			'Tampa Bay Buccaneers': 'tb',
			'Tennessee Titans': 'ten',
			'Washington Commanders': 'wsh',
		}
		
		if team_name not in team_lookup_dict:
			raise ValueError(f"Team '{team_name}' not found in lookup dictionary")
		
		return team_lookup_dict[team_name]