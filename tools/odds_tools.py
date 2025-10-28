from data_sources.OddsAPI import OddsAPI

def create_odds_tools(api_key):
	"""Creates tools definitions for the Odds API"""
	
	oa = OddsAPI(api_key)
	
	tool_definition = {
		"name": "get_nfl_odds",
		"description": """
			Fetches current betting lines and odds for upcoming NFL games from The Odds API.
			
			Returns odds from multiple sportsbooks including:
			- Moneyline (which team is favored to win)
			- Spread (point spread betting)
			- Totals (over/under)
			
			Use this tool when you need to:
			- Check current betting lines for upcoming NFL games
			- Compare odds across different sportsbooks
			- Make betting decisions based on current market prices
			
			The data includes game start times, team names, and odds from all available bookmakers.
		""".strip(),
		"input_schema": {
			"type": "object",
			"properties": {
				# No parameters
			},
			"required": [] # No required parameters 
		}
	}
	
	# Function call
	def execute_tool(**kwargs):
		"""Defining how the API actually gets called"""
		try:
			result = oa.get_nfl_odds()
			return {
				"success": True,
				"data": result
			}
		except Exception as e:
			return {
				"success": False,
				"error": str(e)
			}
	
	return {
		"definition": tool_definition,
		"function": execute_tool
	}