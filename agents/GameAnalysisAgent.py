import config
import json
import ollama
import config
from helpers.Lookup import Lookup
from tools.adjust_aggregates_tools import adjust_data_aggregates
from DataAggregate.DataAggregate import DataAggregate

class GameAnalysisAgent:
	def __init__(self, game_details):
		self.game_details = game_details
		self.analysis = None
	
	def run(self):
		"""Main agent loop"""
		print(f"🏈 Starting Game Analysis Agent")
		
		finished = False
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		empty_responses = 0
		
		while not finished:
			if empty_responses >= 3:
				raise ValueError(f"I couldn't complete my task. You must have not passed me the data that I needed. Be sure to send me all of the matchup details.")
			# Get agent's response
			response = ollama.chat(
				model = config.model,
				messages = messages,
				tools = self.__get_tool_definition()
			)
			
			msg = response['message']
			messages.append(response['message'])
			
			if not msg.thinking and not msg.content and not msg.tool_calls:
				empty_responses += 1
			
			print(f"\n{'='*80}")
			print(f"🏈 Game Analysis Agent Response")
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
					print(f"Agent is calling a tool: { tool_call['function']['name'] }")
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
							
				if 'game analysis complete' in msg.content.lower():
					print(f"🏈 Exiting Game Analysis Agent")
					finished = True
					return self.analysis
						
			print(f"{'='*80}\n")
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are an expert NFL analyst that uses an exhaustive data set to predict upcoming NFL games.
		
		INPUT: 
		- A detailed set of data for an upcoming NFL game. These details can include:
			- Prediction Data: Data from multiple machine learning models that predict the outcome of games. Some models will also include the predicted point spread.
			- Injury Adjusted Prediction Data: Data from multiple machine learning models that predict the outcome of games where the prediction is calculated based on baseline data adjusted from injury reports.
			- Injury Report: A detailed injury report for the team
			- Expert Analysis: Summaries of expert analysis about the upcoming game. Should be compared against the Machine Learning Model results
		
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
		
		Before you call the tool make SURE that you are passing valid characters. DO NOT HALLUCINATE CHARACTERS.
		
		After you have successfully called the save_analysis tool, respond with 'game analysis complete'"""
			
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
					}
				},
				'required': ['analysis']
			}
		}]
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call['function']['name']
		arguments = tool_call['function']['arguments']
		
		if function_name == 'save_analysis':
			try:
				self.analysis = json.loads(arguments['analysis'])
				return "save_analysis tool has been called successfully."
			except Exception as e:
				return {
					'error': str(e)
				}		
		
		else:
			raise ValueError(f"{ function_name }is not a valid tool.")
