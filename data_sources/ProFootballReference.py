import numpy as np
import pandas as pd
import random
import time
import sqlite3

class ProFootballReference:
	def __init__(self):
		self.teams = [
			'crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal', 'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan',
			'sdg', 'ram', 'rai', 'mia', 'min', 'nwe', 'nor', 'nyg', 'nyj', 'phi', 'pit', 'sea', 'sfo', 'tam', 'oti', 'was'
		]		
		
	def get_historical_data(self, seasons):
		start_time = time.time()
		event_columns = ['event_id', 'season', 'season_week_number', 'date', 'day_of_week', 'home_team', 'away_team', 'overtime', 'is_playoffs', 'is_complete']
		event_df = pd.DataFrame()
		
		game_data_columns = [
			'event_id',
			'team',
			'date',
			'opponent',
			'is_home',
			'win',
			'points_scored',
			'pass_completions',
			'pass_attempts',
			'pass_completion_percentage',
			'pass_yds',
			'pass_tds',
			'pass_yards_per_attempt',
			'pass_adjusted_yards_per_attempt',
			'pass_rating',
			'sacks_allowed',
			'sack_yards_lost',
			'rushing_attempts',
			'rushing_yards',
			'rushing_tds',
			'rushing_yards_per_attempt',
			'offensive_plays',
			'total_yards',
			'yards_per_play',
			'field_goal_attempts',
			'field_goals_made',
			'extra_point_attempts',
			'extra_points_made',
			'punts',
			'punt_yards',
			'passing_first_downs',
			'rushing_first_downs',
			'penalty_first_downs',
			'first_downs',
			'third_down_conversions',
			'third_down_attempts',
			'fourth_down_conversions',
			'fourth_down_attempts',
			'penalties',
			'penalty_yards',
			'fumbles_lost',
			'interceptions_thrown',
			'turnovers',
			'time_of_possession'
		]
		game_data_df = pd.DataFrame()
		
		col_rename_dict = {
			'Gtm': 'team_game_number',
			'Week': 'season_week_number',
			'Date': 'date',
			'Day': 'day_of_week',
			'Unnamed: 5': 'is_home',
			'Opp': 'opponent',
			'Rslt': 'win',
			'Pts': 'points_scored',
			'PtsO': 'points_allowed',
			'OT': 'overtime',
			'Cmp': 'pass_completions',
			'Att': 'pass_attempts',
			'Cmp%': 'pass_completion_percentage',
			'Yds': 'pass_yds',
			'TD': 'pass_tds',
			'Y/A': 'pass_yards_per_attempt',
			'AY/A': 'pass_adjusted_yards_per_attempt',
			'Rate': 'pass_rating',
			'Sk': 'sacks_allowed',
			'Yds.1': 'sack_yards_lost',
			'Att.1': 'rushing_attempts',
			'Yds.2': 'rushing_yards',
			'TD.1': 'rushing_tds',
			'Y/A.1': 'rushing_yards_per_attempt',
			'Ply': 'offensive_plays',
			'Tot': 'total_yards',
			'Y/P': 'yards_per_play',
			'FGA': 'field_goal_attempts',
			'FGM': 'field_goals_made',
			'XPA': 'extra_point_attempts',
			'XPM': 'extra_points_made',
			'Pnt': 'punts',
			'Yds.3': 'punt_yards',
			'Pass': 'passing_first_downs',
			'Rsh': 'rushing_first_downs',
			'Pen': 'penalty_first_downs',
			'1stD': 'first_downs',
			'3DConv': 'third_down_conversions',
			'3DAtt': 'third_down_attempts',
			'4DConv': 'fourth_down_conversions',
			'4DAtt': 'fourth_down_attempts',
			'Pen.1': 'penalties',
			'Yds.4': 'penalty_yards',
			'FL': 'fumbles_lost',
			'Int': 'interceptions_thrown',
			'TO': 'turnovers',
			'ToP': 'time_of_possession'
		}
		opp_to_pfr_code = {
			'ARI': 'crd',
			'ATL': 'atl',
			'BAL': 'rav',
			'BUF': 'buf',
			'CAR': 'car',
			'CHI': 'chi',
			'CIN': 'cin',
			'CLE': 'cle',
			'DAL': 'dal',
			'DEN': 'den',
			'DET': 'det',
			'GNB': 'gnb',
			'HOU': 'htx',
			'IND': 'clt',
			'JAX': 'jax',
			'JAC': 'jax',  # Sometimes Jacksonville
			'KAN': 'kan',
			'KC': 'kan',   # Sometimes Kansas City
			'LVR': 'rai',  # Las Vegas Raiders
			'LV': 'rai',   # Also Las Vegas
			'OAK': 'rai',  # Oakland Raiders (historical)
			'LAC': 'sdg',  # LA Chargers
			'SD': 'sdg',
			'SDG': 'sdg',   # San Diego Chargers (historical)
			'LAR': 'ram',  # LA Rams
			'LA': 'ram',   # Could be Rams
			'STL': 'ram',  # St. Louis Rams (historical)
			'MIA': 'mia',
			'MIN': 'min',
			'NE': 'nwe',
			'NWE': 'nwe',
			'NO': 'nor',
			'NOR': 'nor',
			'NYG': 'nyg',
			'NYJ': 'nyj',
			'PHI': 'phi',
			'PIT': 'pit',
			'SF': 'sfo',
			'SFO': 'sfo',
			'SEA': 'sea',
			'TB': 'tam',
			'TAM': 'tam',
			'TEN': 'oti',
			'WAS': 'was',
			'WSH': 'was',
		}
		
		for season in seasons:
			for team in self.teams:
				for table_id in ['table_pfr_team-year_game-logs_team-year-regular-season-game-log','table_pfr_team-year_game-logs_team-year-playoffs-game-log']:
					try:
						game_data_url = f'https://www.pro-football-reference.com/teams/{team}/{str(season)}/gamelog/'
						print(game_data_url)
						tm_df = pd.read_html(game_data_url, header=1, attrs={'id': table_id})[0]
						tm_df = tm_df.drop(columns=['Rk'], axis=1)
						tm_df = tm_df.rename(col_rename_dict, axis=1)
						tm_df = tm_df.dropna(subset=['team_game_number', 'win'])
						tm_df['season'] = season
						if(table_id == 'table_pfr_team-year_game-logs_team-year-playoffs-game-log'):
							tm_df['is_playoffs'] = 1
						else:
							tm_df['is_playoffs'] = 0
						tm_df['team'] = team
						tm_df['team_game_number'] = tm_df['team_game_number'].astype(int)
						tm_df['season_week_number'] = tm_df['season_week_number'].astype(int)
						tm_df['is_neutral'] = np.where(tm_df['is_home'] == 'N', 1, 0)
						tm_df['is_home'] = np.where(tm_df['is_home'] == '@', 0, 1)
						tm_df['win'] = np.where(tm_df['win'] == 'W', 1, 0)
						tm_df['overtime'] = np.where(tm_df['overtime'] == 'OT', 1, 0)
						tm_df['opponent_raw'] = tm_df['opponent']
						tm_df['opponent'] = tm_df['opponent'].map(opp_to_pfr_code)
						unmapped = tm_df[tm_df['opponent'].isna()]
						if len(unmapped) > 0:
							print(f"WARNING: Found {len(unmapped)} unmapped opponents:")
							print(unmapped[['season', 'team', 'opponent_raw', 'opponent']].drop_duplicates('opponent_raw'))
							print("\nUnique unmapped values:", unmapped['Opp'].unique())
						tm_df['home_team'] = np.where(tm_df['is_home'] == 1, team, tm_df['opponent'])
						tm_df['away_team'] = np.where(tm_df['is_home'] == 1, tm_df['opponent'], team)
						# Little trick here, if it's a neutral site have to make sure the event_id is created consistently so order alphabetically
						tm_df['event_id'] = np.where(
							tm_df['is_neutral'] == 1,
							tm_df['season'].astype(str) + '_' + tm_df['season_week_number'].astype(str) + '_' + np.minimum(tm_df['team'], tm_df['opponent']) + '_' + np.maximum(tm_df['team'], tm_df['opponent']),
							(tm_df['season'].astype(str) + '_' + tm_df['season_week_number'].astype(str) + '_' + tm_df['home_team'] + '_' + tm_df['away_team'])
						)
						tm_df['is_complete'] = 1
						event_df = pd.concat([event_df, tm_df[event_columns].copy()], ignore_index=True)
						game_data_df = pd.concat([game_data_df, tm_df[game_data_columns].copy()], ignore_index=True)
						
					except(ValueError, IndexError):
						print(f"No playoff data found for {team} in {season}")
						pass
					
					time.sleep(random.randint(4,5))
		
		# Data Cleanup
		
		event_df = event_df.drop_duplicates(subset=['event_id'], keep='first')

		end_time = time.time()
		print(f'Loaded Pro Football Reference historical data in {end_time - start_time:1f} seconds')
		
		return { 'events': event_df, 'game_data': game_data_df }		
	
	def load_game_data_from_db(self):
		conn=sqlite3.connect('db/historical_data.db')
		query_game_data = """
		SELECT
		
			e.event_id,
			e.season,
			e.season_week_number,
			e.date,
			e.home_team,
			e.away_team,
			
			-- TEAM A STATS
			team_a.team as team_a,
			team_a.is_home as team_a_is_home,
			team_a.points_scored as team_a_points_scored,
			team_a.win as team_a_win,
			
			--TEAM B STATS
			team_b.team as team_b,
			team_b.is_home as team_b_is_home,
			team_b.points_scored as team_b_points_scored,
			team_b.win as team_b_win
		
		FROM
		
			event e
			join team_result team_a on e.event_id = team_a.event_id AND e.home_team = team_a.team
			join team_result team_b on e.event_id = team_b.event_id AND e.away_team = team_b.team
		
		ORDER BY
		
			e.season, 
			e.season_week_number
		"""
		game_data = pd.read_sql(query_game_data, conn)
		conn.close()
		return game_data
	
	def load_team_performance_from_db(self):
		conn=sqlite3.connect('db/historical_data.db')
		query_team_performance = """
		
		SELECT
			*
		FROM
			team_result
		
		"""

		team_performance = pd.read_sql(query_team_performance, conn)
		conn.close()
		return team_performance
