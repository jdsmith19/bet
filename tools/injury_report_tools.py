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
	
	injury_reports = []
	dca = NFLDepthChartAnalyzer()
	lu = Lookup()
	try:
		for team in teams:
			print(f"Generating detailed injury report for { lu.team_name_to_espn_code(team) }")
			injury_reports.append(dca.get_llm_prompt_context(lu.team_name_to_espn_code(team)))			
		return injury_reports
	
	except Exception as e:
		return traceback.print_exc()