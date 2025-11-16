from data_sources.ESPN import NFLDepthChartAnalyzer
from helpers.Lookup import Lookup
import time
import json
import config
import traceback

def get_injury_report_for_teams(teams: list) -> dict:
	"""
	Gets a detailed injury analysis report for the list of teams specified.
	
	Args:
		teams: A list of team names for which to provide the detailed injury analysis report
	
	Returns:
		List[string]: A list of detailed injury reports for the teams provided
	"""
	requested_teams = []
	injury_reports = []
	dca = NFLDepthChartAnalyzer()
	lu = Lookup()
	i = 0
	for team in teams:
		if team not in requested_teams:
			if i > 32:
				break
			requested_teams.append(team)
			print(f"Generating detailed injury report for { team }")
			try:
				injury_reports.append(dca.get_injury_summary_for_agent(lu.team_name_to_espn_code(team)))
				i += 1
			except Exception as e:
				return traceback.print_exc()
	return injury_reports
	
