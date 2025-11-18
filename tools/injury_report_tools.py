from data_sources.ESPN import NFLDepthChartAnalyzer
from helpers.Lookup import Lookup
import traceback
from tqdm import tqdm

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
	pbar = tqdm(teams, desc="Generating injury reports")
	pbar.write("\n🤕 Generating injury reports")
	for team in teams:
		if team not in requested_teams:
			requested_teams.append(team)
			pbar.set_description(f"{team}")
			try:
				injury_reports.append(dca.get_injury_summary_for_agent(lu.team_name_to_espn_code(team)))
			except Exception as e:
				return traceback.print_exc()
			pbar.update(1)
	pbar.set_description("DONE")
	return injury_reports
	
