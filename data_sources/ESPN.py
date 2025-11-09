import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from datetime import datetime

class NFLDepthChartAnalyzer:
	"""
	Comprehensive NFL injury impact analysis system
	Scrapes depth charts and player stats from ESPN to quantify injury impact
	with intelligent backup detection and replacement-level baselines
	Outputs optimized for LLM agent consumption
	"""
	
	def __init__(self):
		self.headers = {
			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
		}
		
		self.injury_severity = {
			'O': 'Out',
			'D': 'Doubtful',
			'Q': 'Questionable',
			'IR': 'Injured Reserve',
			'PUP': 'Physically Unable to Perform',
			'NFI': 'Non-Football Injury',
			'SUS': 'Suspended'
		}
		
		# Position mapping for ESPN variations
		self.position_mapping = {
			'PK': 'K',      # Placekicker -> Kicker
			'H': 'P',       # Holder -> Punter
			'NB': 'CB',     # Nickel Back -> Cornerback
			'RDT': 'DT',    # Right DT -> DT
			'LDT': 'DT',    # Left DT -> DT
			'RDH': 'RB',    # Right Deep Half -> RB (rare)
			'LDH': 'RB',    # Left Deep Half -> RB (rare)
		}
		
		self.position_weights = {
			'QB': 1.0, 'LT': 0.85, 'RT': 0.75, 'EDGE': 0.80,
			'CB': 0.75, 'WR': 0.70, 'RB': 0.60, 'TE': 0.55,
			'C': 0.70, 'LG': 0.65, 'RG': 0.65, 'DT': 0.70,
			'LB': 0.65, 'S': 0.65, 'K': 0.40, 'P': 0.30,
			'LDE': 0.80, 'RDE': 0.80, 'NT': 0.70, 'WLB': 0.65,
			'SLB': 0.65, 'MLB': 0.65, 'LILB': 0.65, 'RILB': 0.65,
			'LCB': 0.75, 'RCB': 0.75, 'SS': 0.65, 'FS': 0.65,
			'FB': 0.45, 'LS': 0.20, 'PR': 0.30, 'KR': 0.30,
			'NB': 0.70, 'RDT': 0.70, 'LDT': 0.70
		}
		
		self.playing_time_thresholds = {
			'QB': 1, 'RB': 2, 'FB': 1, 'WR': 3, 'TE': 2,
			'LT': 1, 'LG': 1, 'C': 1, 'RG': 1, 'RT': 1,
			'LDE': 2, 'RDE': 2, 'EDGE': 2, 'DT': 3, 'NT': 2,
			'LB': 3, 'WLB': 2, 'SLB': 2, 'MLB': 2, 'LILB': 2, 'RILB': 2,
			'CB': 3, 'LCB': 2, 'RCB': 2, 'S': 2, 'SS': 2, 'FS': 2,
			'K': 1, 'P': 1, 'LS': 1, 'KR': 2, 'PR': 2,
			'NB': 1, 'RDT': 2, 'LDT': 2
		}
	
	def normalize_position(self, position):
		"""Normalize ESPN position codes to standard positions"""
		return self.position_mapping.get(position, position)
	
	def is_relevant_injury(self, position, depth):
		"""Determine if this injury is relevant based on playing time"""
		normalized_pos = self.normalize_position(position)
		threshold = self.playing_time_thresholds.get(normalized_pos, 2)
		return depth <= threshold
	
	def get_team_depth_chart(self, team_abbr):
		"""Parse ESPN depth charts using dual-table structure"""
		url = f"https://www.espn.com/nfl/team/depth/_/name/{team_abbr}"
		response = requests.get(url, headers=self.headers)
		soup = BeautifulSoup(response.content, 'html.parser')
		
		depth_chart = []
		responsive_tables = soup.find_all('div', class_='ResponsiveTable')
		
		for table_container in responsive_tables:
			title_div = table_container.find('div', class_='Table__Title')
			formation = title_div.get_text(strip=True) if title_div else 'Unknown'
			
			tables = table_container.find_all('table', class_='Table')
			if len(tables) < 2:
				continue
			
			position_table = tables[0]
			player_table = tables[1]
			
			position_rows = position_table.find('tbody').find_all('tr')
			positions = []
			for row in position_rows:
				cell = row.find('td')
				if cell:
					position = cell.get_text(strip=True).split()[0]
					positions.append(position)
			
			player_rows = player_table.find('tbody').find_all('tr')
			
			for idx, player_row in enumerate(player_rows):
				if idx >= len(positions):
					break
				
				position = positions[idx]
				cells = player_row.find_all('td')
				
				for depth_idx, cell in enumerate(cells, start=1):
					player_link = cell.find('a', class_='AnchorLink')
					
					if player_link:
						player_name = player_link.get_text(strip=True)
						player_url = player_link.get('href', '')
						player_id = None
						
						if '/player/_/id/' in player_url:
							try:
								player_id = player_url.split('/id/')[1].split('/')[0]
							except:
								pass
						
						injury_span = cell.find('span', class_='nfl-injuries-status')
						injury_status = injury_span.get_text(strip=True) if injury_span else None
						
						depth_chart.append({
							'formation': formation,
							'position': position,
							'normalized_position': self.normalize_position(position),
							'depth': depth_idx,
							'player_name': player_name,
							'player_id': player_id,
							'player_url': player_url,
							'injury_status': injury_status
						})
		
		return pd.DataFrame(depth_chart)
	
	def get_player_stats(self, player_id, season=2024):
		"""Get 2024 stats from ESPN using paired table structure"""
		if not player_id:
			return None
		
		url = f"https://www.espn.com/nfl/player/stats/_/id/{player_id}"
		
		try:
			response = requests.get(url, headers=self.headers)
			soup = BeautifulSoup(response.content, 'html.parser')
			
			stats = {
				'player_id': player_id,
				'season': season,
				'has_stats': False
			}
			
			tables = soup.find_all('table', class_='Table')
			
			i = 0
			while i < len(tables):
				table = tables[i]
				thead = table.find('thead')
				
				if not thead:
					i += 1
					continue
				
				header_rows = thead.find_all('tr')
				if not header_rows:
					i += 1
					continue
				
				headers = [th.get_text(strip=True) for th in header_rows[-1].find_all('th')]
				
				if headers and headers[0] in ['season', 'Team'] and len(headers) <= 2:
					tbody = table.find('tbody')
					if tbody:
						for row in tbody.find_all('tr'):
							cells = row.find_all('td')
							if not cells:
								continue
							
							first_cell = cells[0].get_text(strip=True)
							
							if str(season) in first_cell:
								stats['has_stats'] = True
								stats['Team'] = cells[1].get_text(strip=True) if len(cells) > 1 else ''
								
								if i + 1 < len(tables):
									stat_table = tables[i + 1]
									stat_thead = stat_table.find('thead')
									
									if stat_thead:
										stat_headers = [th.get_text(strip=True) for th in stat_thead.find_all('tr')[-1].find_all('th')]
										stat_tbody = stat_table.find('tbody')
										
										if stat_tbody:
											stat_rows = stat_tbody.find_all('tr')
											row_index = list(tbody.find_all('tr')).index(row)
											
											if row_index < len(stat_rows):
												stat_row = stat_rows[row_index]
												stat_cells = stat_row.find_all('td')
												
												for j, value in enumerate(stat_cells):
													if j < len(stat_headers):
														header = stat_headers[j]
														cell_text = value.get_text(strip=True)
														
														if cell_text and cell_text not in ['-', '--']:
															try:
																if '/' in cell_text:
																	stats[header] = cell_text
																elif '.' in cell_text:
																	stats[header] = float(cell_text.replace(',', ''))
																elif cell_text.replace(',', '').replace('-', '').isdigit():
																	stats[header] = int(cell_text.replace(',', ''))
																else:
																	stats[header] = cell_text
															except:
																stats[header] = cell_text
								
								return stats
				
				i += 1
			
			return stats
			
		except Exception as e:
			return None
	
	def parse_qb_stats(self, stats):
		"""Parse QB-specific stats"""
		qb_metrics = {
			'completions': 0, 'attempts': 0, 'completion_percentage': 0,
			'pass_yards': 0, 'pass_tds': 0, 'interceptions': 0,
			'passer_rating': 0, 'yards_per_attempt': 0
		}
		
		for key in ['CMP', 'Completions']:
			if key in stats:
				qb_metrics['completions'] = stats[key]
				break
		
		for key in ['ATT', 'Attempts']:
			if key in stats:
				qb_metrics['attempts'] = stats[key]
				break
		
		for key in ['CMP%', 'Completion %']:
			if key in stats:
				qb_metrics['completion_percentage'] = stats[key]
				break
		
		for key in ['YDS', 'Passing Yards']:
			if key in stats:
				qb_metrics['pass_yards'] = stats[key]
				break
		
		for key in ['TD', 'Passing TDs']:
			if key in stats:
				qb_metrics['pass_tds'] = stats[key]
				break
		
		for key in ['INT', 'Interceptions']:
			if key in stats:
				qb_metrics['interceptions'] = stats[key]
				break
		
		for key in ['RTG', 'Rating']:
			if key in stats:
				qb_metrics['passer_rating'] = stats[key]
				break
		
		for key in ['AVG', 'YPA', 'Y/A']:
			if key in stats:
				qb_metrics['yards_per_attempt'] = stats[key]
				break
		
		return qb_metrics
	
	def parse_rb_stats(self, stats):
		"""Parse RB stats"""
		rb_metrics = {
			'rush_attempts': 0, 'rush_yards': 0,
			'rush_tds': 0, 'yards_per_carry': 0
		}
		
		for key in ['CAR', 'ATT']:
			if key in stats:
				rb_metrics['rush_attempts'] = stats[key]
				break
		
		for key in ['YDS', 'Rushing Yards']:
			if key in stats:
				rb_metrics['rush_yards'] = stats[key]
				break
		
		for key in ['TD', 'Rushing TDs']:
			if key in stats:
				rb_metrics['rush_tds'] = stats[key]
				break
		
		for key in ['AVG', 'YPC']:
			if key in stats:
				rb_metrics['yards_per_carry'] = stats[key]
				break
		
		return rb_metrics
	
	def parse_wr_stats(self, stats):
		"""Parse WR/TE stats"""
		wr_metrics = {
			'receptions': 0, 'targets': 0, 'rec_yards': 0,
			'rec_tds': 0, 'yards_per_reception': 0, 'catch_rate': 0
		}
		
		for key in ['REC', 'Receptions']:
			if key in stats:
				wr_metrics['receptions'] = stats[key]
				break
		
		for key in ['TGTS', 'TAR', 'Targets']:
			if key in stats:
				wr_metrics['targets'] = stats[key]
				break
		
		for key in ['YDS', 'Receiving Yards']:
			if key in stats:
				wr_metrics['rec_yards'] = stats[key]
				break
		
		for key in ['TD', 'Receiving TDs']:
			if key in stats:
				wr_metrics['rec_tds'] = stats[key]
				break
		
		for key in ['AVG', 'YPR']:
			if key in stats:
				wr_metrics['yards_per_reception'] = stats[key]
				break
		
		if wr_metrics['targets'] > 0:
			wr_metrics['catch_rate'] = (wr_metrics['receptions'] / wr_metrics['targets']) * 100
		
		return wr_metrics
	
	def parse_ol_stats(self, stats):
		"""Parse OL stats (basic availability metrics)"""
		ol_metrics = {
			'games_played': 0,
			'games_started': 0
		}
		
		for key in ['GP', 'Games Played']:
			if key in stats:
				ol_metrics['games_played'] = stats[key]
				break
		
		ol_metrics['games_started'] = ol_metrics['games_played']
		
		return ol_metrics
	
	def parse_dl_stats(self, stats):
		"""Parse DL stats"""
		dl_metrics = {
			'tackles': 0,
			'sacks': 0,
			'tackles_for_loss': 0,
			'qb_hits': 0
		}
		
		for key in ['TOT', 'Total Tackles']:
			if key in stats:
				dl_metrics['tackles'] = stats[key]
				break
		
		for key in ['SACK', 'Sacks']:
			if key in stats:
				dl_metrics['sacks'] = stats[key]
				break
		
		return dl_metrics
	
	def parse_lb_stats(self, stats):
		"""Parse LB stats"""
		lb_metrics = {
			'tackles': 0,
			'solo_tackles': 0,
			'sacks': 0,
			'tackles_for_loss': 0
		}
		
		for key in ['TOT', 'Total Tackles']:
			if key in stats:
				lb_metrics['tackles'] = stats[key]
				break
		
		for key in ['SOLO', 'Solo Tackles']:
			if key in stats:
				lb_metrics['solo_tackles'] = stats[key]
				break
		
		for key in ['SACK', 'Sacks']:
			if key in stats:
				lb_metrics['sacks'] = stats[key]
				break
		
		return lb_metrics
	
	def parse_db_stats(self, stats):
		"""Parse DB (CB/S) stats"""
		db_metrics = {
			'tackles': 0,
			'interceptions': 0,
			'passes_defended': 0
		}
		
		for key in ['TOT', 'Total Tackles']:
			if key in stats:
				db_metrics['tackles'] = stats[key]
				break
		
		for key in ['INT', 'Interceptions']:
			if key in stats:
				db_metrics['interceptions'] = stats[key]
				break
		
		for key in ['PD', 'Passes Defended']:
			if key in stats:
				db_metrics['passes_defended'] = stats[key]
				break
		
		return db_metrics
	
	def get_replacement_level_player(self, position):
		"""Return replacement-level stats for a position"""
		# Normalize position first
		normalized_pos = self.normalize_position(position)
		
		replacement_stats = {
			'QB': {
				'passer_rating': 75.0, 'yards_per_attempt': 6.2,
				'completions': 0, 'attempts': 0, 'completion_percentage': 60.0,
				'pass_yards': 0, 'pass_tds': 0, 'interceptions': 0
			},
			'RB': {
				'yards_per_carry': 3.8, 'rush_attempts': 0,
				'rush_yards': 0, 'rush_tds': 0
			},
			'WR': {
				'yards_per_reception': 10.0, 'catch_rate': 55.0,
				'receptions': 0, 'targets': 0, 'rec_yards': 0, 'rec_tds': 0
			},
			'TE': {
				'yards_per_reception': 9.0, 'catch_rate': 60.0,
				'receptions': 0, 'targets': 0, 'rec_yards': 0, 'rec_tds': 0
			},
			'OL': {
				'games_played': 10,
				'games_started': 10
			},
			'DL': {
				'tackles': 25,
				'sacks': 2.0,
				'tackles_for_loss': 3,
				'qb_hits': 5
			},
			'LB': {
				'tackles': 50,
				'solo_tackles': 30,
				'sacks': 1.0,
				'tackles_for_loss': 3
			},
			'DB': {
				'tackles': 40,
				'interceptions': 1,
				'passes_defended': 5
			}
		}
		
		if normalized_pos in ['LT', 'LG', 'C', 'RG', 'RT']:
			return replacement_stats.get('OL', {})
		elif normalized_pos in ['LDE', 'RDE', 'EDGE', 'DT', 'NT']:
			return replacement_stats.get('DL', {})
		elif normalized_pos in ['LB', 'WLB', 'SLB', 'MLB', 'LILB', 'RILB']:
			return replacement_stats.get('LB', {})
		elif normalized_pos in ['CB', 'LCB', 'RCB', 'S', 'SS', 'FS']:
			return replacement_stats.get('DB', {})
		
		return replacement_stats.get(normalized_pos, {})
	
	def find_backup_player(self, depth_chart_df, injured_player_row):
		"""Find backup with intelligent fallback logic"""
		formation = injured_player_row['formation']
		position = injured_player_row['position']
		normalized_position = injured_player_row['normalized_position']
		depth = injured_player_row['depth']
		
		# Try 1: Direct backup
		direct_backup = depth_chart_df[
			(depth_chart_df['formation'] == formation) &
			(depth_chart_df['position'] == position) & 
			(depth_chart_df['depth'] == depth + 1)
		]
		
		if not direct_backup.empty:
			return direct_backup.iloc[0], 'direct'
		
		# Try 2: Same position, different formation
		same_position_backup = depth_chart_df[
			(depth_chart_df['position'] == position) & 
			(depth_chart_df['depth'] == depth + 1) &
			(depth_chart_df['formation'] != formation)
		]
		
		if not same_position_backup.empty:
			return same_position_backup.iloc[0], 'cross_formation'
		
		# Try 3: Positional flexibility (use normalized positions)
		position_flexibility = {
			'QB': [], 'RB': ['FB'], 'FB': ['RB', 'TE'], 'WR': ['TE'], 'TE': ['WR', 'FB'],
			'LT': ['RT', 'LG'], 'RT': ['LT', 'RG'], 'LG': ['RG', 'C', 'LT'],
			'RG': ['LG', 'C', 'RT'], 'C': ['LG', 'RG'],
			'LDE': ['RDE', 'EDGE', 'DT'], 'RDE': ['LDE', 'EDGE', 'DT'],
			'EDGE': ['LDE', 'RDE', 'LB'], 'DT': ['NT', 'LDE', 'RDE'], 'NT': ['DT'],
			'LB': ['EDGE', 'S'], 'WLB': ['SLB', 'MLB', 'LILB', 'RILB'],
			'SLB': ['WLB', 'MLB', 'LILB', 'RILB'], 'MLB': ['WLB', 'SLB', 'LILB', 'RILB'],
			'LILB': ['RILB', 'MLB', 'WLB', 'SLB'], 'RILB': ['LILB', 'MLB', 'WLB', 'SLB'],
			'CB': ['S'], 'LCB': ['RCB', 'CB', 'S'], 'RCB': ['LCB', 'CB', 'S'],
			'S': ['CB'], 'SS': ['FS', 'CB', 'LB'], 'FS': ['SS', 'CB'],
			'K': [], 'P': []
		}
		
		flex_positions = position_flexibility.get(normalized_position, [])
		
		for flex_pos in flex_positions:
			flex_backup = depth_chart_df[
				(depth_chart_df['normalized_position'] == flex_pos) &
				(depth_chart_df['depth'] <= 2)
			]
			
			if not flex_backup.empty:
				return flex_backup.iloc[0], 'positional_flex'
		
		return None, 'replacement_level'
	
	def compare_players_stats(self, starter_id, backup_id, position):
		"""Compare starter vs backup with REAL statistics for ALL positions"""
		# Normalize position
		normalized_pos = self.normalize_position(position)
		
		starter_stats = self.get_player_stats(starter_id) if starter_id else None
		backup_stats = self.get_player_stats(backup_id) if backup_id else None
		
		if not starter_stats or not backup_stats:
			return None
		
		if not starter_stats.get('has_stats') or not backup_stats.get('has_stats'):
			return None
		
		comparison = {
			'starter_id': starter_id, 'backup_id': backup_id,
			'position': normalized_pos, 'has_comparison': True,
			'differential_score': None, 'metrics': {}
		}
		
		if normalized_pos == 'QB':
			starter_metrics = self.parse_qb_stats(starter_stats)
			backup_metrics = self.parse_qb_stats(backup_stats)
			
			rating_diff = starter_metrics.get('passer_rating', 0) - backup_metrics.get('passer_rating', 0)
			ypa_diff = starter_metrics.get('yards_per_attempt', 0) - backup_metrics.get('yards_per_attempt', 0)
			
			rating_score = (rating_diff / 40) * 50
			ypa_score = (ypa_diff / 2) * 25
			
			comparison['differential_score'] = max(0, min(100, rating_score + ypa_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'rating_diff': rating_diff, 'ypa_diff': ypa_diff
			}
		
		elif normalized_pos == 'RB':
			starter_metrics = self.parse_rb_stats(starter_stats)
			backup_metrics = self.parse_rb_stats(backup_stats)
			
			ypc_diff = starter_metrics.get('yards_per_carry', 0) - backup_metrics.get('yards_per_carry', 0)
			td_rate_starter = (starter_metrics.get('rush_tds', 0) / max(1, starter_metrics.get('rush_attempts', 1))) * 100
			td_rate_backup = (backup_metrics.get('rush_tds', 0) / max(1, backup_metrics.get('rush_attempts', 1))) * 100
			td_rate_diff = td_rate_starter - td_rate_backup
			
			ypc_score = (ypc_diff / 2) * 50
			td_score = (td_rate_diff / 5) * 25
			
			comparison['differential_score'] = max(0, min(100, ypc_score + td_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'ypc_diff': ypc_diff, 'td_rate_diff': td_rate_diff
			}
		
		elif normalized_pos in ['WR', 'TE']:
			starter_metrics = self.parse_wr_stats(starter_stats)
			backup_metrics = self.parse_wr_stats(backup_stats)
			
			ypr_diff = starter_metrics.get('yards_per_reception', 0) - backup_metrics.get('yards_per_reception', 0)
			catch_rate_diff = starter_metrics.get('catch_rate', 0) - backup_metrics.get('catch_rate', 0)
			
			ypr_score = (ypr_diff / 5) * 50
			catch_score = (catch_rate_diff / 20) * 25
			
			comparison['differential_score'] = max(0, min(100, ypr_score + catch_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'ypr_diff': ypr_diff, 'catch_rate_diff': catch_rate_diff
			}
		
		elif normalized_pos in ['LT', 'LG', 'C', 'RG', 'RT']:
			starter_metrics = self.parse_ol_stats(starter_stats)
			backup_metrics = self.parse_ol_stats(backup_stats)
			
			games_diff = starter_metrics.get('games_played', 0) - backup_metrics.get('games_played', 0)
			games_score = (games_diff / 17) * 50
			
			comparison['differential_score'] = max(0, min(100, games_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'games_diff': games_diff
			}
		
		elif normalized_pos in ['LDE', 'RDE', 'EDGE', 'DT', 'NT']:
			starter_metrics = self.parse_dl_stats(starter_stats)
			backup_metrics = self.parse_dl_stats(backup_stats)
			
			sacks_diff = starter_metrics.get('sacks', 0) - backup_metrics.get('sacks', 0)
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			
			sacks_score = (sacks_diff / 10) * 50
			tackles_score = (tackles_diff / 50) * 25
			
			comparison['differential_score'] = max(0, min(100, sacks_score + tackles_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'sacks_diff': sacks_diff, 'tackles_diff': tackles_diff
			}
		
		elif normalized_pos in ['LB', 'WLB', 'SLB', 'MLB', 'LILB', 'RILB']:
			starter_metrics = self.parse_lb_stats(starter_stats)
			backup_metrics = self.parse_lb_stats(backup_stats)
			
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			sacks_diff = starter_metrics.get('sacks', 0) - backup_metrics.get('sacks', 0)
			
			tackles_score = (tackles_diff / 80) * 60
			sacks_score = (sacks_diff / 5) * 15
			
			comparison['differential_score'] = max(0, min(100, tackles_score + sacks_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'tackles_diff': tackles_diff, 'sacks_diff': sacks_diff
			}
		
		elif normalized_pos in ['CB', 'LCB', 'RCB', 'S', 'SS', 'FS']:
			starter_metrics = self.parse_db_stats(starter_stats)
			backup_metrics = self.parse_db_stats(backup_stats)
			
			int_diff = starter_metrics.get('interceptions', 0) - backup_metrics.get('interceptions', 0)
			pd_diff = starter_metrics.get('passes_defended', 0) - backup_metrics.get('passes_defended', 0)
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			
			int_score = (int_diff / 5) * 40
			pd_score = (pd_diff / 10) * 30
			tackles_score = (tackles_diff / 60) * 15
			
			comparison['differential_score'] = max(0, min(100, int_score + pd_score + tackles_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'int_diff': int_diff, 'pd_diff': pd_diff, 'tackles_diff': tackles_diff
			}
		
		else:
			comparison['has_comparison'] = False
		
		return comparison
	
	def compare_to_replacement_level(self, starter_id, position):
		"""Compare starter to replacement-level player for ALL positions"""
		# Normalize position
		normalized_pos = self.normalize_position(position)
		
		starter_stats = self.get_player_stats(starter_id) if starter_id else None
		
		if not starter_stats or not starter_stats.get('has_stats'):
			return None
		
		replacement_stats = self.get_replacement_level_player(position)
		
		if not replacement_stats:
			return None
		
		comparison = {
			'starter_id': starter_id, 'backup_id': None, 'position': normalized_pos,
			'has_comparison': True, 'backup_type': 'replacement_level',
			'differential_score': None, 'metrics': {}
		}
		
		if normalized_pos == 'QB':
			starter_metrics = self.parse_qb_stats(starter_stats)
			backup_metrics = replacement_stats
			
			rating_diff = starter_metrics.get('passer_rating', 0) - backup_metrics.get('passer_rating', 0)
			ypa_diff = starter_metrics.get('yards_per_attempt', 0) - backup_metrics.get('yards_per_attempt', 0)
			
			rating_score = (rating_diff / 40) * 50
			ypa_score = (ypa_diff / 2) * 25
			
			comparison['differential_score'] = max(0, min(100, rating_score + ypa_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'rating_diff': rating_diff, 'ypa_diff': ypa_diff
			}
		
		elif normalized_pos == 'RB':
			starter_metrics = self.parse_rb_stats(starter_stats)
			backup_metrics = replacement_stats
			
			ypc_diff = starter_metrics.get('yards_per_carry', 0) - backup_metrics.get('yards_per_carry', 0)
			
			ypc_score = (ypc_diff / 2) * 50
			comparison['differential_score'] = max(0, min(100, ypc_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'ypc_diff': ypc_diff
			}
		
		elif normalized_pos in ['WR', 'TE']:
			starter_metrics = self.parse_wr_stats(starter_stats)
			backup_metrics = replacement_stats
			
			ypr_diff = starter_metrics.get('yards_per_reception', 0) - backup_metrics.get('yards_per_reception', 0)
			
			ypr_score = (ypr_diff / 5) * 50
			comparison['differential_score'] = max(0, min(100, ypr_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'ypr_diff': ypr_diff
			}
		
		elif normalized_pos in ['LT', 'LG', 'C', 'RG', 'RT']:
			starter_metrics = self.parse_ol_stats(starter_stats)
			backup_metrics = replacement_stats
			
			games_diff = starter_metrics.get('games_played', 0) - backup_metrics.get('games_played', 0)
			games_score = (games_diff / 17) * 50
			
			comparison['differential_score'] = max(0, min(100, games_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'games_diff': games_diff
			}
		
		elif normalized_pos in ['LDE', 'RDE', 'EDGE', 'DT', 'NT']:
			starter_metrics = self.parse_dl_stats(starter_stats)
			backup_metrics = replacement_stats
			
			sacks_diff = starter_metrics.get('sacks', 0) - backup_metrics.get('sacks', 0)
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			
			sacks_score = (sacks_diff / 10) * 50
			tackles_score = (tackles_diff / 50) * 25
			
			comparison['differential_score'] = max(0, min(100, sacks_score + tackles_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'sacks_diff': sacks_diff, 'tackles_diff': tackles_diff
			}
		
		elif normalized_pos in ['LB', 'WLB', 'SLB', 'MLB', 'LILB', 'RILB']:
			starter_metrics = self.parse_lb_stats(starter_stats)
			backup_metrics = replacement_stats
			
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			sacks_diff = starter_metrics.get('sacks', 0) - backup_metrics.get('sacks', 0)
			
			tackles_score = (tackles_diff / 80) * 60
			sacks_score = (sacks_diff / 5) * 15
			
			comparison['differential_score'] = max(0, min(100, tackles_score + sacks_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'tackles_diff': tackles_diff, 'sacks_diff': sacks_diff
			}
		
		elif normalized_pos in ['CB', 'LCB', 'RCB', 'S', 'SS', 'FS']:
			starter_metrics = self.parse_db_stats(starter_stats)
			backup_metrics = replacement_stats
			
			int_diff = starter_metrics.get('interceptions', 0) - backup_metrics.get('interceptions', 0)
			pd_diff = starter_metrics.get('passes_defended', 0) - backup_metrics.get('passes_defended', 0)
			tackles_diff = starter_metrics.get('tackles', 0) - backup_metrics.get('tackles', 0)
			
			int_score = (int_diff / 5) * 40
			pd_score = (pd_diff / 10) * 30
			tackles_score = (tackles_diff / 60) * 15
			
			comparison['differential_score'] = max(0, min(100, int_score + pd_score + tackles_score))
			comparison['metrics'] = {
				'starter': starter_metrics, 'backup': backup_metrics,
				'int_diff': int_diff, 'pd_diff': pd_diff, 'tackles_diff': tackles_diff
			}
		
		return comparison
	
	def calculate_impact_score(self, injured_player_row, backup_player_row, backup_type, depth_chart_df):
		"""Calculate impact score with backup type awareness"""
		position = injured_player_row['position']
		normalized_position = injured_player_row['normalized_position']
		depth = injured_player_row['depth']
		injury_status = injured_player_row['injury_status']
		
		base_score = 25
		
		if backup_type == 'replacement_level':
			comparison = self.compare_to_replacement_level(injured_player_row['player_id'], position)
			
			if comparison and comparison.get('differential_score') is not None:
				base_score = comparison['differential_score']
			else:
				position_weight = self.position_weights.get(normalized_position, 0.5)
				base_score = position_weight * 70
		
		elif backup_type == 'positional_flex':
			position_weight = self.position_weights.get(normalized_position, 0.5)
			base_score = position_weight * 50
		
		elif backup_type == 'cross_formation':
			if backup_player_row is not None:
				try:
					stats_comparison = self.compare_players_stats(
						injured_player_row['player_id'],
						backup_player_row['player_id'],
						position
					)
					
					if stats_comparison and stats_comparison.get('has_comparison'):
						base_score = stats_comparison.get('differential_score', 25) * 1.1
					else:
						position_weight = self.position_weights.get(normalized_position, 0.5)
						depth_dropoff = {1: 0.40, 2: 0.25, 3: 0.15}
						dropoff = depth_dropoff.get(depth, 0.10)
						base_score = position_weight * dropoff * 100 * 1.1
				except:
					position_weight = self.position_weights.get(normalized_position, 0.5)
					depth_dropoff = {1: 0.40, 2: 0.25, 3: 0.15}
					dropoff = depth_dropoff.get(depth, 0.10)
					base_score = position_weight * dropoff * 100 * 1.1
		
		else:  # backup_type == 'direct'
			if backup_player_row is not None:
				try:
					stats_comparison = self.compare_players_stats(
						injured_player_row['player_id'],
						backup_player_row['player_id'],
						position
					)
					
					if stats_comparison and stats_comparison.get('has_comparison'):
						base_score = stats_comparison.get('differential_score', 25)
					else:
						position_weight = self.position_weights.get(normalized_position, 0.5)
						depth_dropoff = {1: 0.40, 2: 0.25, 3: 0.15}
						dropoff = depth_dropoff.get(depth, 0.10)
						base_score = position_weight * dropoff * 100
				except:
					position_weight = self.position_weights.get(normalized_position, 0.5)
					depth_dropoff = {1: 0.40, 2: 0.25, 3: 0.15}
					dropoff = depth_dropoff.get(depth, 0.10)
					base_score = position_weight * dropoff * 100
		
		severity_multipliers = {
			'O': 1.0, 'D': 0.8, 'Q': 0.5,
			'IR': 1.0, 'PUP': 1.0, 'NFI': 1.0, 'SUS': 1.0
		}
		severity_multiplier = severity_multipliers.get(injury_status, 0.3)
		
		impact_score = base_score * severity_multiplier
		
		# Apply minimum floor for starters to prevent 0.0 scores
		if depth == 1 and impact_score < 5.0:
			impact_score = 5.0
		
		return round(impact_score, 2)
	
	def analyze_injury_impact(self, team_abbr):
		"""Enhanced injury analysis - only include players who see playing time"""
		df = self.get_team_depth_chart(team_abbr)
		injured = df[df['injury_status'].notna() & (df['injury_status'] != '')]
		
		if injured.empty:
			return pd.DataFrame()
		
		impact_report = []
		
		for _, injured_player in injured.iterrows():
			if not self.is_relevant_injury(injured_player['position'], injured_player['depth']):
				continue
			
			backup_row, backup_type = self.find_backup_player(df, injured_player)
			
			if backup_type == 'replacement_level':
				backup_name = 'Replacement Level'
				backup_id = None
			elif backup_type == 'positional_flex':
				backup_name = f"{backup_row['player_name']} ({backup_row['position']})"
				backup_id = backup_row['player_id']
			elif backup_type == 'cross_formation':
				backup_name = f"{backup_row['player_name']} (alt formation)"
				backup_id = backup_row['player_id']
			else:
				backup_name = backup_row['player_name'] if backup_row is not None else 'Unknown'
				backup_id = backup_row['player_id'] if backup_row is not None else None
			
			impact_score = self.calculate_impact_score(injured_player, backup_row, backup_type, df)
			
			if impact_score >= 30:
				impact_level = 'CRITICAL'
			elif impact_score >= 20:
				impact_level = 'HIGH'
			elif impact_score >= 10:
				impact_level = 'MEDIUM'
			else:
				impact_level = 'LOW'
			
			position_label = f"{injured_player['position']}{injured_player['depth']}"
			is_starter = injured_player['depth'] == 1
			
			# Use normalized position for consistency
			normalized_pos = injured_player['normalized_position']
			
			impact_report.append({
				'team': team_abbr.upper(),
				'position': normalized_pos,
				'original_position': injured_player['position'],
				'depth': injured_player['depth'],
				'position_label': position_label,
				'is_starter': is_starter,
				'injured_player': injured_player['player_name'],
				'injured_player_id': injured_player['player_id'],
				'injury_status': injured_player['injury_status'],
				'injury_description': self.injury_severity.get(injured_player['injury_status'], 'Unknown'),
				'backup_player': backup_name,
				'backup_player_id': backup_id,
				'backup_type': backup_type,
				'impact_score': impact_score,
				'impact_level': impact_level,
				'position_weight': self.position_weights.get(normalized_pos, 0.5),
				'playing_time_threshold': self.playing_time_thresholds.get(normalized_pos, 2)
			})
		
		return pd.DataFrame(impact_report).sort_values('impact_score', ascending=False).reset_index(drop=True)
	
	def get_injury_summary_for_agent(self, team_abbr):
		"""Generate structured summary for agent consumption"""
		injury_df = self.analyze_injury_impact(team_abbr)
		
		if injury_df.empty:
			return {
				'team': team_abbr.upper(),
				'has_injuries': False,
				'total_impact_score': 0,
				'impact_percentage': 0,
				'injured_count': 0,
				'injuries': [],
				'summary': {
					'critical_count': 0,
					'high_count': 0,
					'starter_injuries': 0,
					'key_positions_affected': [],
					'offensive_impact': 0,
					'defensive_impact': 0
				}
			}
		
		offensive_positions = ['QB', 'RB', 'WR', 'TE', 'LT', 'LG', 'C', 'RG', 'RT']
		
		total_impact = injury_df['impact_score'].sum()
		
		# Calculate impact percentage (normalized to 0-100 scale)
		# Using 100 as baseline max, but can exceed for severe compound injuries
		impact_percentage = min(100, total_impact)
		
		summary = {
			'critical_count': len(injury_df[injury_df['impact_level'] == 'CRITICAL']),
			'high_count': len(injury_df[injury_df['impact_level'] == 'HIGH']),
			'starter_injuries': len(injury_df[injury_df['is_starter'] == True]),
			'key_positions_affected': injury_df[injury_df['is_starter'] == True]['position'].tolist(),
			'offensive_impact': float(injury_df[injury_df['position'].isin(offensive_positions)]['impact_score'].sum()),
			'defensive_impact': float(injury_df[~injury_df['position'].isin(offensive_positions + ['K', 'P', 'LS', 'KR', 'PR'])]['impact_score'].sum())
		}
		
		return {
			'team': team_abbr.upper(),
			'has_injuries': True,
			'total_impact_score': round(total_impact, 2),
			'impact_percentage': round(impact_percentage, 1),
			'injured_count': len(injury_df),
			'injuries': injury_df.to_dict('records'),
			'summary': summary
		}
	
	def to_json_for_llm(self, team_abbr):
		"""Format as JSON for LLM agent consumption"""
		data = self.get_injury_summary_for_agent(team_abbr)
		return json.dumps(data, indent=2)
	
	def get_llm_prompt_context(self, team_abbr):
		"""
		Generate natural language context for LLM agent
		This is the primary method for LLM integration - SINGLE TEAM
		"""
		data = self.get_injury_summary_for_agent(team_abbr)
		
		prompt = f"""# NFL INJURY REPORT ANALYSIS

## TEAM: {team_abbr.upper()}

### INJURY SITUATION:
- **Total Impact Score:** {data['total_impact_score']:.1f}/100
- **Critical/High Impact Injuries:** {data['summary']['critical_count'] + data['summary']['high_count']}
- **Starter Injuries:** {data['summary']['starter_injuries']}
- **Offensive Impact:** {data['summary']['offensive_impact']:.1f}
- **Defensive Impact:** {data['summary']['defensive_impact']:.1f}

### DETAILED INJURY LIST:
"""
		
		if data['injuries']:
			for inj in data['injuries']:
				status_emoji = {'O': '❌', 'D': '⚠️', 'Q': '❓', 'IR': '🏥'}.get(inj['injury_status'], '❓')
				prompt += f"""
**{inj['position_label']}** - {status_emoji} {inj['injury_status']} ({inj['injury_description']})
  - Injured: {inj['injured_player']}
  - Backup: {inj['backup_player']}
  - Impact Score: {inj['impact_score']:.0f}/100
  - Impact Level: {inj['impact_level']}
  - Backup Type: {inj['backup_type']}
"""
		else:
			prompt += "  ✅ No significant injuries\n"
		
		prompt += f"""
---

## YOUR TASK:
Based on the injury analysis above, provide adjustment recommendations for this team's prediction model metrics.

Consider the following when making adjustments:
1. **Position importance** - QB injuries matter more than RB2 injuries
2. **Backup quality** - Direct backups vs replacement level players
3. **Injury severity** - Out/IR vs Questionable
4. **Cumulative impact** - Multiple injuries compound the effect

Please provide adjustment multipliers (0.70 - 1.00) for the following categories:
- 1.00 = no adjustment needed
- 0.90 = slight negative impact (~10% worse)
- 0.80 = moderate negative impact (~20% worse)
- 0.70 = severe negative impact (~30% worse)

Provide your response in JSON format with the following structure:
{{
  "team": "{team_abbr.upper()}",
  "adjustments": {{
	"offense": {{
	  "passing": 0.XX,
	  "rushing": 0.XX,
	  "overall": 0.XX
	}},
	"defense": {{
	  "pass_defense": 0.XX,
	  "run_defense": 0.XX,
	  "overall": 0.XX
	}},
	"position_factors": {{
	  "qb": 0.XX,
	  "offensive_line": 0.XX,
	  "skill_positions": 0.XX,
	  "defensive_line": 0.XX,
	  "linebackers": 0.XX,
	  "secondary": 0.XX
	}}
  }},
  "reasoning": "Your detailed explanation here - explain which injuries drove which adjustments",
  "confidence": "HIGH/MEDIUM/LOW",
  "key_concerns": ["List of 2-3 most impactful injuries or trends"]
}}
"""
		
		return prompt