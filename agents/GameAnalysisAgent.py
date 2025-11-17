from dotenv import load_dotenv
import os
import time
import config
import json
import ollama
import config
from openai import OpenAI
from helpers.Lookup import Lookup
from tools.adjust_aggregates_tools import adjust_data_aggregates
from DataAggregate.DataAggregate import DataAggregate

load_dotenv()

class GameAnalysisAgent:
	def __init__(self, game_details):
		self.game_details = game_details
		self.analysis = None
		self.api_type = os.getenv('API_TYPE')
	
	def run(self):
		"""Main agent loop"""
		start_time = time.time()
		print(f"🏈 Starting Game Analysis Agent")
		
		finished = False
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
				
		while not finished:
			base_url = os.getenv('OPEN_AI_BASE_URL')
			model = os.getenv('ADJUSTMENT_MODEL')
		
			client = OpenAI(
				base_url = base_url,
				api_key = "no-key-needed"
			)
						
			response = client.chat.completions.create(
				model = model,
				messages = messages,
				tools = self.__get_tool_definition()
			)
						
			msg = response.choices[0].message
			messages.append(msg.model_dump(exclude_none=True))
			
			print(f"\n{'='*80}")
			print(f"🏈 Game Analysis Agent Response")
			print(f"{'='*80}\n")
							
			if msg.content:
				print(f"💬 EXPLANATION:")
				print(f"{msg.content}\n")

			if msg.tool_calls:
				# Process tool calls
				for tool_call in msg.tool_calls:
					result = self.__execute_tool(tool_call)
					
					# Add tool result to messages
					messages.append({
						'role': 'tool',
						'tool_call_id': tool_call.id,
						'content': json.dumps(result)
					})
					
					if tool_call.function.name == 'save_analysis':
						print(f"🏈 Exiting Game Analysis Agent")
						print(f"Completed in { round(time.time() - start_time, 3) }s")
						print(f"{'='*80}\n")
						return self.analysis
						
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are an expert NFL analyst that uses an exhaustive data set to predict upcoming NFL games.
		
		INPUT: 
		- A detailed set of data for an upcoming NFL game. These details can include:
			- Prediction Data: Data from multiple machine learning models that predict the outcome of games. Some models will also include the predicted point spread.
			- Injury Adjusted Prediction Data: Data from multiple machine learning models that predict the outcome of games where the prediction is calculated based on baseline data adjusted from injury reports.
			- Injury Report: A detailed injury report for the team
			- Expert Analysis: Summaries of expert analysis about the upcoming game. Should be compared against the Machine Learning Model results
			- Podcast Analysis: A summary of analysis from The Bill Simmons Podcast for the upcoming game. Should be compared against the Machine Learning Model results
		
		OUTPUT:
		- An object containing some of the details passed to you along with your analysis and final prediction based on all available information
		
		TASK:
		- Generate an analysis report for the upcoming game. Think deeply about the impacts of all the data available to you to make your final prediction.
		- Pass your analysis in the OUTPUT FORMAT to the save_analysis tool as a string

		OUTPUT FORMAT:		
		Return ONLY valid JSON, nothing else:
			{{
				'matchup': '[Away Team] @ [Home Team]',
				'base_model_prediction': '[winner] by [spread] pts',
				'injury_adjusted_prediction': '[winner by spread] pts',
				'final_prediction': '[your call based on all available information]',
				'confidence': '[VERY LOW, LOW, MEDIUM, HIGH, or VERY HIGH],
				'analysis': '[your reasoning for your final prediction, include at least 3 reasons which each reason as an entry of a list]
			}}
		
		CONFIDENCE:
		Confidence is defined as how sure you are of your pick for the WINNER of the game. When deciding the value for confidence, take the metrics for the model into account. For regression models consider the mean_squared_error and root_mean_squared_error values. In order for the confidence to be considered VERY HIGH, the spread should be larger than the variance. If the spread is very small and the variance is high, then the confidence should be considered VERY SMALL. For classifier models, you can use the prediction's confidence value should be combined with the test_accuracy to determine your overall confidence.
		
		IMPORTANT: Before you call the tool make SURE that you are passing valid characters. DO NOT HALLUCINATE CHARACTERS."""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Here are the full game details. Analyze.
		
		{ self.game_details }"""
		
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return [{
			'type': 'function',
			'function': {
				'name': 'save_analysis',
				'description': 'Saves game analysis',
				'parameters': {
					'type': 'object',
					'properties': {
						'analysis': {
							'type': 'string',
							'description': """Complete analysis JSON as string"""
						}
					},
					'required': ['analysis']  # ✅ Move it here!
				}
			}
		}]
		
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call.function.name
		arguments = tool_call.function.arguments

		if function_name == 'save_analysis':
			try:
				self.analysis = json.loads(arguments)['analysis']
				return "save_analysis tool has been called successfully."
			except Exception as e:
				return {
					'error': str(e)
				}		
		
		else:
			raise ValueError(f"{ function_name }is not a valid tool.")
