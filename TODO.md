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

# MEDIUM

Add RPI
Add Home / Away Splits
Add rolling aggregates for at least [3,5] days
Change how I pull upcoming games
	* Pull all upcoming games from ProFootballReference
	* Include Week #
	# Use query of current data set of complete games to find upcoming games

## DONE
Add Logistic Regression
Add Random Forest Regressor
Return data from all predictions in a JSON dictionary format
Create a parent class for all prediction engines that the individual classes inherit from