import requests
import time

class OddsAPI:
	def __init__(self, api_key):
		self.api_key = api_key
	
	def get_nfl_odds(self):
		url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={ self.api_key }&regions=us&markets=h2h,spreads&oddsFormat=american"
		r = requests.get(url)
		return r.json()