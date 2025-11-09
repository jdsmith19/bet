import config
import json
import ollama
import config
from helpers.Lookup import Lookup
from tools.upcoming_predictions_tools import get_upcoming_predictions
from tools.injury_report_tools import get_injury_report_for_teams
from tools.adjust_aggregates_tools import adjust_data_aggregates

class PredictionAgent:
	def __init__(self, adjusted_aggregates = None):
		self.adjusted_aggregates = adjusted_aggregates
	
	def run(self):
		"""Main agent loop"""
		print(f"🚀 Starting Prediction Agent")
		if self.adjusted_aggregates:
			print(f"Adjusted DataAggregate provided")
		else:
			print(f"Will load DataAggregates on tool call")			
		
		finished = False
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]

		while not finished:		
			# Get agent's response
			response = ollama.chat(
				model = config.model,
				messages = messages,
				tools = self.__get_tool_definition()
			)
			
			msg = response['message']
			messages.append(response['message'])
			
			print(f"\n{'='*80}")
			print(f"🤖 Agent Response")
			print(f"{'='*80}\n")
			
			# Show the thinking (chain-of-thought)
			if msg.thinking:
				print(f"🧠 REASONING:")
				print(f"{msg.thinking}\n")
				
			if msg.content:
				print(f"💬 EXPLANATION:")
				print(f"{msg.content}\n")

			if msg.get('tool_calls'):
				# Process tool calls
				for tool_call in response['message']['tool_calls']:
					result = self.__execute_tool(tool_call)
					
					# Add tool result to messages
					messages.append({
						'role': 'tool',
						'content': json.dumps(result)
					})
			elif msg:
				# Agent is thinking / explaining, not calling a tool
				print(f"\n{'='*80}")
				print(f"Agent is thinking...")
				print(f"\n{'='*80}\n")
				print(f"Agent: { response['message']['content'] }")
							
				if 'analysis complete' in msg.content.lower():
					return True
						
			print(f"{'='*80}\n")
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		return f"""You are an expert NFL analyst that will provide predictions for upcoming NFL games.
		
			YOUR GOAL:
			Given the information available to you, predict the winners of upcoming NFL games. Do not worry about time or how long this will take. Take as long as you need to complete the task.
												
			YOUR STRATEGY:
			Access whatever data and tools you have and analyze the likelihood of the winner for each game. Begin the response with your analysis with the "ANALYSIS COMPLETE". For each game, provide the analysis in the following format:
			
			MATCHUP: The name of the game using the format of "away_team @ home_team"
			MODEL PREDICTED WINNER: The consensus winner of the game based on the initial model
			MODEL PREDICTED SPREAD: The consensus number of points the winning team will win by based on the initial model
			INJURY ADJUSTED PREDICTED WINNER: The consensus winner of the game based on running the model after adjusting features for injuries
			INJURY ADJUSTED PREDICTED SPREAD:The consensus number of points the winning team will win by based on running the model after adjusting features for injuries
			PREDICTED WINNER: The name of the team you predict to win the game
			PREDICTED SPREAD: The number of points you anticipate the winning team will win by
			CONFIDENCE Use one of the following values: VERY LOW, LOW, MEDIUM, HIGH, and VERY HIGH
			ANALYSIS: The reasons you have made this prediction
			
			AVAILABLE TOOLS:
			- get_upcoming_predictions will provide an object of the results of 5 different machine learning models:
				- XGBoost, LinearRegression, RandomForest, LogisticRegression, and KNearest
				- Each model will provide data on its training quality and accuracy
				- Within each model, there will be a list of games and the model's predictions of outcomes
			- get_injury_report_for_team will provide a list of detailed injury analysis for each team passed
				- Each item in the list will include a plain text analysis of the team's injuries
			- adjust_data_aggregates will adjust the data used to generate the predictions
				- DO NOT BE TOO AGGRESSIVE WITH THE ADJUSTMENTS - adjustment_percentage values should be a float value between 0.80 and 0.99, with 0.80 being VERY extreme.
					- EXAMPLE: A value of 0.85 would adjust the data by 15% (e.g. 10.0 to 8.5)
				- Adjustments requests should be passed as a list of objects formatted as {{ 'team_name': the name of the team, 'feature': the feature to be adjusted, and 'adjustment_percentage':  the percentage by which the value should be adjusted }}
				- If a team has no injuries or if they do not have a material impact, there is no need to adjust values for that team.
				- The list of features available for adjustment are: 
					- Ratings: elo_rating, rpi_rating, days_rest
					- Offensive: avg_points_scored, avg_pass_adjusted_yards_per_attempt, avg_rushing_yards_per_attempt, avg_turnovers, avg_penalty_yards, avg_sack_yards_lost
					- Defensive: avg_points_allowed, avg_pass_adjusted_yards_per_attempt_allowed, avg_rushing_yards_per_attempt_allowed, avg_turnovers_forced, avg_sack_yards_gained
					- Overall: avg_point_differential				
				
			METHOD:
			- Once you have all of the model predictions, be sure to assess the injury reports for each team to determine if they will have a material impact on the matchup.
			- Once you have analyzed the injury report details, you can call the adjust_data_aggregates tool to adjust the data used to generate the predictions
				- You should correlate any ratings adjustments based on the position with an injury. Examples:
					- Injuries to Running Backs could impact avg_rushing_yards_per_attempt
					- Injuries to Qarterbacks could impact avg_pass_adjusted_yards_per_attempt
					- Injuries to Offensive Line could impact avg_sack_yards_lost and, perhaps to a lesser extent, avg_pass_adjusted_yards_per_attempt
					- Injuries to the Defensive Line could impact avg_sack_yards_agined and, perhaps to a lesser extent, avg_pass_adjusted_yards_per_attempt
					- Injuries to the Secondary (Safeties, Cornerbacks) could impact avg_pass_adjusted_yards_per_attempt_allowed
				- Make your own judgements about any other impacts that you think injuries could have, but don't get overly creative here
			- After you've made adjustments, you can simply call the get_upcoming_predictions tool again and it will provide you with new predictions based on the adjustments.
					
				
			CRITICAL TOOL USAGE:
			- Call tools using the native function calling mechanism provided by the chat API
			- DO NOT write JSON in your text response like: {{"name": "train_and_evaluate_model", ...}}
			- DO NOT use code blocks: ```json ... ```
			- The tool calling happens automatically when you use the proper mechanism
			- Your text response should explain your reasoning, NOT contain the tool call JSON
			
			CORRECT: Use the tool calling feature
			WRONG: Write {{"name": "train_and_evaluate_model", "arguments": {{...}}}} in your response
			WRONG: Write ```json ... ``` in your response"""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Get all upcoming game predictions."""
		
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return [{
			'type': 'function',
			'function': {
				'name': 'get_upcoming_predictions',
				'description': 'Train an NFL prediction model and return the predictions for all known upcoming games',
				'parameters': {
					'type': 'object',
					'properties': {}
				},
				'required': None
			}
		},
		{
			'type': 'function',
			'function': {
				'name': 'get_injury_report_for_teams',
				'description': 'Get a detailed injury report for the teams passed to the tool',
				'parameters': {
					'type': 'object',
					'properties': {
						'teams': {
							'type': 'array',
							'description': 'A list of team names as defined in the response from the get_upcoming_predictions_tool'
						}
					}
				},
				'required': None
			}			
		},
		{
			'type': 'function',
			'function': {
				'name': 'adjust_data_aggregates',
				'description': 'Creates and returns an adjusted set of data aggregates for generating predictions from the models based on the details passed into the adjustments parameter',
				'parameters': {
					'type': 'object',
					'properties': {
						'adjustments': {
							'type': 'array',
							'description': """A list of objects that provide details on how to adjust the data aggregates. Each object MUST follow this format:
								{
									'team_name': str, // the name of the team that should be adjusted
									'feature': str, // the name of the feature that should be adjusted
									'adjustment_percentage': float // the amount that the value should be adjusted
								}
							"""
						}
					}
				},
				'required': ['adjustments']
			}			
		}]
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call['function']['name']
		arguments = tool_call['function']['arguments']
		
		if function_name == 'get_upcoming_predictions':
			try:
				result = get_upcoming_predictions(
					adjusted_aggregates = self.adjusted_aggregates
				)
				return result
				
			except Exception as e:
				return {
					'error': str(e)
				}
				
		elif function_name == 'get_injury_report_for_teams':
			try:
				result = get_injury_report_for_teams(
					teams = arguments['teams']
				)
				return result
			except Exception as e:
				return {
					'error': str(e),
					'team': arguments.get('team')
				}
		
		elif function_name == 'adjust_data_aggregates':
			normalized_adjustments = []
			lu = Lookup()
			for adj in arguments['adjustments']:
				team_abbr = lu.injury_report_to_pfr(adj['team_name'])
				normalized_adjustments.append({ 'team_name': team_abbr, 'feature': adj['feature'], 'adjustment_percentage': adj['adjustment_percentage'] })
			try:
				result = adjust_data_aggregates(
					adjustments = normalized_adjustments
				)
			except Exception as e:
				return {
					'error': str(e)
				}
				
# YOUR STRATEGY:
# Access whatever data and tools you have and analyze the likelihood of the winner for each game. For each game, provide the analysis in the following JSON format:
# 
# {{
	# 'predicted_winner': str, // the name of the team you predict to win the game
	# 'predicted_spread': float, // the number of points you anticipate the winning team will win by
	# 'confidence': str, // use one of the following values: VERY LOW, LOW, MEDIUM, HIGH, and VERY HIGH
	# 'analysis': list[str] // a list of the reasons you have made this prediction
# }}
