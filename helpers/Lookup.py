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
			'San Francisco 49s': 'sf',
			'Seattle Seahawks': 'sea',
			'Tampa Bay Buccaneers': 'tb',
			'Tennessee Titans': 'ten',
			'Washington Commanders': 'wsh',
		}
		
		if team_name not in team_lookup_dict:
			raise ValueError(f"Team '{team_name}' not found in lookup dictionary")
		
		return team_lookup_dict[team_name]
	
	def injury_report_to_pfr(self, team_name):
		team_lookup_dict = {
			'ARI': 'crd',  # Arizona Cardinals
			'ATL': 'atl',  # Atlanta Falcons
			'BAL': 'rav',  # Baltimore Ravens
			'BUF': 'buf',  # Buffalo Bills
			'CAR': 'car',  # Carolina Panthers
			'CHI': 'chi',  # Chicago Bears
			'CIN': 'cin',  # Cincinnati Bengals
			'CLE': 'cle',  # Cleveland Browns
			'DAL': 'dal',  # Dallas Cowboys
			'DEN': 'den',  # Denver Broncos
			'DET': 'det',  # Detroit Lions
			'GB': 'gnb',   # Green Bay Packers
			'HOU': 'htx',  # Houston Texans
			'IND': 'clt',  # Indianapolis Colts
			'JAX': 'jax',  # Jacksonville Jaguars
			'KC': 'kan',   # Kansas City Chiefs
			'LAC': 'sdg',  # Los Angeles Chargers
			'LAR': 'ram',  # Los Angeles Rams
			'LV': 'rai',   # Las Vegas Raiders
			'MIA': 'mia',  # Miami Dolphins
			'MIN': 'min',  # Minnesota Vikings
			'NE': 'nwe',   # New England Patriots
			'NO': 'nor',   # New Orleans Saints
			'NYG': 'nyg',  # New York Giants
			'NYJ': 'nyj',  # New York Jets
			'PHI': 'phi',  # Philadelphia Eagles
			'PIT': 'pit',  # Pittsburgh Steelers
			'SF': 'sfo',   # San Francisco 49ers
			'SEA': 'sea',  # Seattle Seahawks
			'TB': 'tam',   # Tampa Bay Buccaneers
			'TEN': 'oti',  # Tennessee Titans (not in your data but included for completeness)
			'WSH': 'was',  # Washington Commanders
		}
		
		if team_name not in team_lookup_dict:
			raise ValueError(f"Team '{team_name}' not found in lookup dictionary")
		
		return team_lookup_dict[team_name]

	def injury_report_to_team_name(self, team_code):
		"""
		Map injury report team codes to full team names
		
		Args:
			team_code: Injury report team code (e.g., 'DEN')
		
		Returns:
			Full team name (e.g., 'Denver Broncos')
		"""
		team_lookup_dict = {
			'ARI': 'Arizona Cardinals',
			'ATL': 'Atlanta Falcons',
			'BAL': 'Baltimore Ravens',
			'BUF': 'Buffalo Bills',
			'CAR': 'Carolina Panthers',
			'CHI': 'Chicago Bears',
			'CIN': 'Cincinnati Bengals',
			'CLE': 'Cleveland Browns',
			'DAL': 'Dallas Cowboys',
			'DEN': 'Denver Broncos',
			'DET': 'Detroit Lions',
			'GB': 'Green Bay Packers',
			'HOU': 'Houston Texans',
			'IND': 'Indianapolis Colts',
			'JAX': 'Jacksonville Jaguars',
			'KC': 'Kansas City Chiefs',
			'LAC': 'Los Angeles Chargers',
			'LAR': 'Los Angeles Rams',
			'LV': 'Las Vegas Raiders',
			'MIA': 'Miami Dolphins',
			'MIN': 'Minnesota Vikings',
			'NE': 'New England Patriots',
			'NO': 'New Orleans Saints',
			'NYG': 'New York Giants',
			'NYJ': 'New York Jets',
			'PHI': 'Philadelphia Eagles',
			'PIT': 'Pittsburgh Steelers',
			'SF': 'San Francisco 49ers',
			'SEA': 'Seattle Seahawks',
			'TB': 'Tampa Bay Buccaneers',
			'TEN': 'Tennessee Titans',
			'WSH': 'Washington Commanders'
		}
			
		if team_code not in team_lookup_dict:
			raise ValueError(f"Team code '{team_code}' not found in lookup dictionary")
		
		return team_lookup_dict[team_code]