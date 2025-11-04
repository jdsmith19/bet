## TO DO

# SMALL

Add season reset to ELO rating.
	1. Add season number to team_performance in initial scrape.
	2. Add code to regress towards league average at start of new season:
		
		```
		if row['season'] != current_season:
			current_season = row['season']
			# Regress all teams toward mean
			for team in elo_dict:
				elo_dict[team] = elo_dict[team] * 0.67 + 1500 * 0.33
		``` 