import config
import json
import ollama
import config
from helpers.Lookup import Lookup
from tools.upcoming_predictions_tools import get_upcoming_predictions
from tools.injury_report_tools import get_injury_report_for_teams
from tools.html_generation_tools import generate_html_report
from agents.InjuryAdjustmentAgent import InjuryAdjustmentAgent
from agents.GameAnalysisAgent import GameAnalysisAgent

class PredictionOrchestrationAgent:
	def __init__(self):
		self.adjusted_aggregates = None
		self.injury_report = None
		self.matchup_details = {}
		self.analysis = {}
	
	def run(self):
		"""Main agent loop"""
		print(f"🎹 Starting Prediction Orchestration Agent")
		print(f"MODEL: { config.orchestration_model }")		
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
			print(f"🎹 Prediction Orchestration Agent Response")
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
					finished = True
						
			print(f"{'='*80}\n")
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are an expert NFL analyst that will provide predictions for upcoming NFL games.
		
		=== MANDATORY WORKFLOW - NO DEVIATIONS ALLOWED ===
		
		You MUST complete these steps in exact order. Skipping or reordering steps is task failure.
		
		STEP 1: GET INITIAL PREDICTIONS
		- Call get_upcoming_predictions
		- Store the results
		- DO NOT make any predictions yet
		
		STEP 2: GET ALL INJURY REPORTS
		- Identify unique teams from the predictions
		- Call get_injury_report_for_team for all unique teams at once
		- Store all injury reports
		- DO NOT skip any teams due to "complexity" or "time"
		
		STEP 3: GET INJURY ADJUSTMENTS
		- Call get_injury_adjustments
				
		STEP 4: GET INJURY-ADJUSTED PREDICTIONS  
		- Call get_upcoming_predictions again
		- Compare with initial predictions
		- Store both sets of results
		
		STEP 5: GET ANALYSIS
		- Call the get_game_analysis tool, which will return the data you need for the next step
		
		STEP 6: GENERATE REPORT
		- You now have analysis for every game:
		- Format all game analyses in a single HTML string, do not send a list
		- Call generate_html_report with the complete HTML string
		- When it returns True, respond with "ANALYSIS COMPLETE"
		
		Format each game as:
		<div class="game">
		  <h3>[Away Team] @ [Home Team]</h3>
		  <p><strong>Base Model Prediction:</strong> [winner] by [spread] pts</p>
		  <p><strong>Injury-Adjusted Prediction:</strong> [winner] by [spread] pts</p>
		  <p><strong>Final Prediction:</strong> [your call]</p>
		  <p><strong>Confidence:</strong> [level]</p>
		  <p><strong>Analysis:</strong> [your reasoning considering both predictions]</p>
		</div>
		
		=== ANTI-SHORTCUT REQUIREMENTS ===
		
		❌ FORBIDDEN BEHAVIORS - These constitute task failure:
		- "This is too complex" - No, it's not. Do it anyway.
		- "Time constraints" - There are no time constraints. You have unlimited time.
		- "Let's simplify by..." - No. Follow the exact workflow above.
		- "Maybe we can approximate..." - No approximations. Use the actual tools.
		- "Manually adjust winner/spread..." - No manual calculations. Use the tools.
		- "Let's focus on key games only..." - No. Process ALL games.
		- "I'll use qualitative assessment instead..." - No. Use the quantitative tools.
		
		✅ CORRECT BEHAVIOR:
		- "I have 18 teams with injuries. Creating 18 adjustment objects now..."
		- "Here are all 23 adjustments I'm making: [complete list]..."
		- "Calling adjust_data_aggregates with all adjustments..."
		- "Retrieving injury-adjusted predictions now..."
		
		=== REALITY CHECK ===
		
		- Creating 20+ adjustment objects takes 30 seconds of work
		- You are a language model with no time constraints
		- The complexity of listing 20 items is trivial
		- Calling a function with a list of 20 items is not "overwhelming"
		- If a human can do this task in 5 minutes, you can do it instantly
		
		=== VERIFICATION CHECKLIST ===
		
		Before generating your report, confirm:
		□ Called get_upcoming_predictions (initial)
		□ Called get_injury_report_for_team for ALL teams
		□ Created adjustments for ALL materially injured teams  
		□ Called adjust_data_aggregates with complete adjustment list
		□ Called get_upcoming_predictions (injury-adjusted)
		□ Generated HTML report with both sets of predictions
		
		If any box is unchecked, you have failed the task.
		
		=== YOUR GOAL ===
		
		Predict winners of upcoming NFL games using BOTH model predictions AND injury-adjusted predictions. The injury-adjusted predictions require you to actually adjust the data and re-run the models. There are no shortcuts."""
			
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
					'properties': {
						
					}
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
				'name': 'get_injury_adjustments',
				'description': 'Updates data aggregates for generating predictions from the models based on the injury report and will let you know when complete',
				'parameters': {
					'type': 'object',
					'properties': {
						#'injury_report': {
						#	'type': 'str',
						#	'description': """A detailed injury report generated by the get_injury_report_for_team tool"""
						# }
					}
				},
				'required': [] #['injury_report']
			}
		},
		{
			'type': 'function',
			'function': {
				'name': 'get_game_analysis',
				'description': 'Generates analysis for games using known data. Returns analysis of all games for generating the report.',
				'parameters': {
					'type': 'object',
					'properties': {}
				},
				'required': [] #['injury_report']
			}
		},		
		{
			'type': 'function',
			'function': {
				'name': 'generate_html_report',
				'description': 'Saves a string of HTML to a file.',
				'parameters': {
					'type': 'object',
					'properties': {
						'html': {
							'type': 'array',
							'description': "Raw HTML to be saved to a file."
						}
					}
				},
				'required': ['html']
			}			
		}]
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call['function']['name']
		arguments = tool_call['function']['arguments']
		
		if function_name == 'get_upcoming_predictions':
			if self.adjusted_aggregates:
				prediction_type = 'injury_adjusted_predictions'
			else:
				prediction_type = 'base_predictions'
			try:
				result = get_upcoming_predictions(
					adjusted_aggregates = self.adjusted_aggregates
				)
				for ml_model in result:
					for prediction in ml_model['results']:
						matchup_name = f"{prediction['away_team']} @ { prediction['home_team']}"
						printf(f"Analyzing { matchup_name }")
						if matchup_name not in self.matchup_details:
							self.matchup_details[matchup_name] = {}
						if prediction_type not in self.matchup_details[matchup_name]:
							self.matchup_details[matchup_name][prediction_type] = []
						self.matchup_details[matchup_name][prediction_type].append(self.__organize_prediction_details(ml_model, prediction))
				return result
				
			except Exception as e:
				import traceback
				traceback.print_exc()
				return {
					'error': str(e)
				}
				
		elif function_name == 'get_injury_report_for_teams':
			try:
				result = get_injury_report_for_teams(
					teams = arguments['teams']
				)
				#print(result)
				self.injury_report = result
				print(self.injury_report)
				return json.dumps(result)
			except Exception as e:
				return {
					'error': str(e),
					'team': arguments.get('team')
				}
		
		elif function_name == 'get_injury_adjustments':
			# print(arguments)
			iaa = InjuryAdjustmentAgent(json.dumps(self.injury_report))
			try:
				self.adjusted_aggregates = iaa.run()
				return "Adjustments complete"
			except Exception as e:
				return {
					'error': str(e)
				}

		elif function_name == 'get_game_analysis':
				# print(arguments)
				analysis = []
				i = 0
				for matchup in self.matchup_details:
					i += 1
					if i >= 4:
						break
					gaa = GameAnalysisAgent(self.matchup_details[matchup])
					try:
						analysis.append(gaa.run())
					except Exception as e:
						print(f"Error in get_game_analysis: {e}")
						import traceback
						traceback.print_exc()
						return {"error": str(e)}
				self.analysis = analysis
				return analysis
		
		elif function_name == 'generate_html_report':
			return generate_html_report(arguments['html'])
		
		else:
			raise ValueError(f"{ function_name }is not a valid tool.")
	
	def __organize_prediction_details(self, ml_model, prediction):
		matchup_details = {
			'model_details': {
				'model_name': ml_model['model_name'],
				'mean_absolute_error': ml_model.get('mean_absolute_error'),
				'root_mean_squared_error': ml_model.get('root_mean_squared_error'),
				'feature_importance': ml_model.get('feature_importance'),
				'feature_coefficients': ml_model.get('feature_coefficients'),
				'train_accuracy': ml_model.get('train_accuracy'),
				'test_accuracy': ml_model.get('test_accuracy'),
				'confidence_intervals': ml_model.get('confidence_intervals')
			},
			'prediction': {
				'predicted_spread': prediction.get('predicted_spread'),
				'predicted_winner': prediction['predicted_winner'],
				'prediction_text': prediction.get('prediction_text'),
				'confidence': prediction.get('confidence')
			}
		}
		return matchup_details

# YOUR STRATEGY:
# Access whatever data and tools you have and analyze the likelihood of the winner for each game. For each game, provide the analysis in the following JSON format:
# 
# {{
	# 'predicted_winner': str, // the name of the team you predict to win the game
	# 'predicted_spread': float, // the number of points you anticipate the winning team will win by
	# 'confidence': str, // use one of the following values: VERY LOW, LOW, MEDIUM, HIGH, and VERY HIGH
	# 'analysis': list[str] // a list of the reasons you have made this prediction
# }}
