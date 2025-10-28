import requests
import time

class BallDontLie:
	def __init__(self, api_key):
		self.api_key = api_key
		self.headers = { 'Authorization': self.api_key }
	
	def get_all_teams(self):
		api_url = "https://api.balldontlie.io/nfl/v1/teams"
		r = requests.get(api_url, headers=self.headers)
		return r.json()
	
	def get_team_by_id(self, team_id):
		api_url = "https://api.balldontlie.io/nfl/v1/teams/" + str(team_id)
		r = requests.get(api_url, headers=self.headers)
		return r.json()

	def get_all_players(self, cursor=False, page_size=100, iteration = 0):
		max = 3
		players = []
		api_url = "https://api.balldontlie.io/nfl/v1/players"
		if(cursor):
			time.sleep(15)
			params = { 'cursor': cursor, 'per_page': page_size }
			r = requests.get(api_url, headers=self.headers, params=params)
		else:
			params = {'per_page': page_size }
			r = requests.get(api_url, headers=self.headers)
			
		# DEBUG
		print(f"Iteration {iteration}: Status Code: {r.status_code}")
		print(f"Response text (first 200 chars): {r.text[:200]}")
				
		data = r.json()
		players.extend(data['data'])
		
		if(iteration > max):
			return players
		
		if(data['meta'].get('next_cursor')):
			players.extend(
				self.get_all_players(
					cursor=data['meta']['next_cursor'],
					iteration = iteration + 1
				)
			)

		return players		