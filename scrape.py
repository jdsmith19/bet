from data_sources.ProFootballReference import ProFootballReference
import pandas as pd
import sqlite3

pfr = ProFootballReference()

def load_history_to_db():
	# Set all historical data from 2011 to 2024 -- this should be a one-time thing
	#data = pfr.get_historical_data(range(2011, 2025))
	data = pfr.get_historical_data([2011])
	
	print("Created DataFrames...")
	print(data['events'].info())
	print(data['game_data'].info())
	
	conn = sqlite3.connect('db/historical_data.db')
	data['events'].to_sql('event', conn, if_exists='replace', index=False)
	data['game_data'].to_sql('team_result', conn, if_exists='replace', index=False)
	
	conn.execute("CREATE UNIQUE INDEX idx_event ON event(event_id)")
	conn.execute("CREATE UNIQUE INDEX idx_team_result ON team_result(event_id, team)")
	conn.close()

load_history_to_db()

def load_recent_to_db():
	data = pfr.get_historical_data([2025])
	
	conn = sqlite3.connect('db/historical_data.db')
	existing_events = pd.read_sql("SELECT event_id FROM event", conn)
	existing_ids = set(existing_events['event_id'])
	
	new_events = data['events'][~data['events']['event_id'].isin(existing_ids)]
	new_events.to_sql('event', conn, if_exists='append', index=False)
	
	existing_results = pd.read_sql("SELECT event_id, team FROM team_result", conn)
	existing_keys = set(zip(existing_results['event_id'], existing_results['team']))
	
	new_results = data['game_data'][
		~data['game_data'].apply(lambda row: (row['event_id'], row['team']) in existing_keys, axis=1)
	]
	new_results.to_sql('team_result', conn, if_exists='append', index=False)
	
	conn.close()

load_recent_to_db()