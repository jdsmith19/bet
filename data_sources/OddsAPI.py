import requests
import time
import pandas as pd

class OddsAPI:
	def __init__(self, api_key):
		self.api_key = api_key
	
	def get_nfl_odds(self):
		url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={ self.api_key }&regions=us&markets=h2h,spreads&oddsFormat=american"
		r = requests.get(url)
		return r.json()
	
	def get_nfl_events(self):
		url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/?apiKey={ self.api_key }"
		r = requests.get(url)
		return r.json()
		
	def get_upcoming_for_pfr_prediction(self):
		odds_api_to_pfr = {
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
		
		odds = self.get_nfl_odds()
		upcoming_games = []
		for event in odds:
			home_team = odds_api_to_pfr[event['home_team']]
			away_team = odds_api_to_pfr[event['away_team']]
			season = event['commence_time'][:4]
			date = event['commence_time'][:10]
			week = 'next'
			day_of_week = 'next'
			event_id = season + '_' + week + '_' + home_team + '_' + away_team
			
			game_row = {
				'event_id': event_id,
				'season': season,
				'season_week_number': week,
				'date': date,
				'day_of_week': day_of_week,
				'home_team': home_team,
				'away_team': away_team
			}
			upcoming_games.append(game_row)
		
		upcoming_games_df = pd.DataFrame(upcoming_games)
		
		return upcoming_games_df
			