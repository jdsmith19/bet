from dotenv import load_dotenv
import os
import time
import config
import json
from openai import OpenAI
import config
import requests
import xmltodict
from helpers.Lookup import Lookup
from tools.adjust_aggregates_tools import adjust_data_aggregates
from DataAggregate.DataAggregate import DataAggregate

load_dotenv()

class ExternalAnalysisAgent:
	def __init__(self, games):
		self.games = games
		self.analysis = None
		self.api_type = os.getenv('API_TYPE')
	
	def run(self):
		"""Main agent loop"""
		start_time = time.time()
		print(f"🛜 External Analysis Agent")
		
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
			
			#print(msg.content)
		
			if not msg.content and not msg.tool_calls:
				empty_responses += 1
				
			print(f"\n{'='*80}")
			print(f"🛜 External Analysis Agent Response")
			print(f"{'='*80}\n")
			
			# Show the thinking (chain-of-thought)				
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
						print(f"🛜 Exiting External Analysis Agent")
						print(f"Completed in { round(time.time() - start_time, 3) }s")
						print(f"{'='*80}\n")
						return self.analysis

					
			else:
				messages.append({
					'role': 'user',
					'content': 'You must CALL the save_analysis tool with that data. Do not just show me the JSON. Actually invoke the save_analysis function.'
				})
						
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		return f"""You are a research agent that extracts PREDICTIVE NFL analysis from expert sources.
		
		INPUT: A list of NFL matchups
		
		OUTPUT: Structured analysis via the save_analysis tool
		
		TASK:
		1. Call sports_chat_palace tool to get expert analysis
		2. Extract relevant analysis for each matchup in the list
		3. Summarize each in 3-5 key points
		4. CALL the save_analysis tool with your complete analysis
		
		CRITICAL RULES:
		- You MUST call save_analysis tool with the complete JSON
		- DO NOT output the JSON in your explanation
		- DO NOT just describe what you would pass to the tool
		- The ONLY way to complete this task is by calling save_analysis
		
		WHAT TO INCLUDE (High Value):
		✓ Current injuries to key players
		✓ Recent performance trends (last 3-4 games)
		✓ Head-to-head matchup advantages (O-line vs D-line, secondary vs receivers)
		✓ Weather impact for this specific game
		✓ Coaching/scheme advantages
		✓ Motivation factors (playoff implications, divisional rivalry)
		✓ Rest advantages/disadvantages
		✓ Home/away splits THIS season
		✓ Key statistical mismatches (e.g., "#1 rush defense vs #32 run offense")

		WHAT TO EXCLUDE (Low Value):
		✗ Historical records beyond current season (e.g., "8-2 in November over last 5 years")
		✗ Arbitrary date-range statistics (e.g., "6-1 in games after a loss since 2021")
		✗ All-time series records
		✗ Career stats for players no longer relevant
		✗ Vague statements without specifics (e.g., "strong defense")
		✗ Stadium trivia or historical facts
		✗ Cherry-picked statistical splits without context

		QUALITY TEST:
		Ask yourself: "Would this insight help predict THIS WEEK'S game?"
		- If it's about current form, injuries, or matchups → INCLUDE
		- If it's a historical fun fact or arbitrary split → EXCLUDE

		MATCHUPS TO ANALYZE:
		{self.games}
		
		OUTPUT FORMAT (pass this to save_analysis tool):
		[
		  {{
			"matchup": "[Away Team] @ [Home Team]",
			"analysis": [{{
			  "source": "Sports Chat Place",
			  "key_points": ["point 1", "point 2", "point 3"]
			}}]
		  }}
		]
		
		EXAMPLES:

		GOOD KEY POINTS:
		- "Chiefs missing LT Jawaan Taylor (ankle), backup gave up 3 sacks last week"
		- "Broncos rush defense ranks 3rd, allowing 89 ypg, while Chiefs average 145 rushing ypg"
		- "Game-time temp expected 15°F with 20mph winds, historically reduces scoring by 8 points"
		- "Eagles 5-0 at home this season, averaging 31 ppg vs 22 ppg on road"

		BAD KEY POINTS:
		- "Cowboys are 12-4 in December games over the last 5 seasons"
		- "This will be the 47th meeting between these teams"
		- "Raiders have won 6 of last 10 after a bye week since 2018"
		- "Team has strong defensive tradition dating back to 1970s"
		
		Focus on CURRENT SEASON context and SPECIFIC matchup factors that impact THIS game."""
		
		return f"""You are a research agent that is responsible for gathering expert NFL analysis for upcoming games from the web and summrizing them with key details.
		
		INPUT: 
		- A list of NFL matchups that need expert NFL analysis
		
		OUTPUT:
		- An object containing a summary of the analysis for each matchup
		
		TASK:
		- Get expert analysis from your available tools
		- Map the analysis from the tool to one of the following games:
		
		{ self.games }
		
		- You may not have external analysis for every game. That's fine, just skip that game if so.
		- Think critically about what factors you think will really affect the outcome of each game.
		- Summarize the analysis in 3 - 5 key points
		- Pass your analysis as an object by passing the OUTPUT FORMAT to the save_analysis tool

		OUTPUT FORMAT:		
		Return ONLY valid JSON, nothing else:
			{{
				'matchup': '[Away Team] @ [Home Team]' should exactly match the value in the list of games provided to you,
				'analysis': [{{
					'source': '[source of the analysis based on the information provided by a tool],
					'key_points': [a list of 3 - 5 key points as strings]
				}}]
			}}
			
		CRITICAL: Ensure your JSON is valid before calling save_analysis.
		The tool call should be exactly:
		{
		  "analysis": [
			{"matchup": "...", "analysis": [...]},
			...
		  ]
		}
		
		Do NOT add extra braces or formatting."""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Here are the list of games we need details for. Call available tools to get external analysis.
		
		{ self.games }"""
		
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return [{
				'type': 'function',
				'function': {
					'name': 'save_analysis',
					'description': 'Saves game analysis as a structured array',
					'parameters': {
						'type': 'object',
						'properties': {
							'analysis': {
								'type': 'array',  # Changed from 'string'
								'items': {
									'type': 'object',
									'properties': {
										'matchup': {'type': 'string'},
										'analysis': {
											'type': 'array',
											'items': {
												'type': 'object',
												'properties': {
													'source': {'type': 'string'},
													'key_points': {
														'type': 'array',
														'items': {'type': 'string'}
													}
												}
											}
										}
									}
								},
								'description': 'Array of matchup analysis objects'
							}
						},
						'required': ['analysis']
					}
				}
			},
			{
				'type': 'function',
				'function': {
					'name': 'sports_chat_palace',
					'description': 'Fetches expert NFL analysis for upcoming games from Sports Chat Palace',
					'parameters': {
						'type': 'object',
						'properties': {}
					},
				},
			}]
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call.function.name
		arguments = json.loads(tool_call.function.arguments)
		
		if function_name == 'save_analysis':
			
			try:
				if isinstance(arguments['analysis'], (list, dict)):
					self.analysis = arguments['analysis']
				# If it's a string, parse it
				elif isinstance(arguments['analysis'], str):
					self.analysis = json.loads(arguments['analysis'])
				#self.analysis = json.loads(arguments['analysis'])
				return "save_analysis tool has been called successfully."
			except Exception as e:
				return {
					'error': str(e)
				}		
		elif function_name == 'sports_chat_palace':
			try:
				headers = {
					'User-Agent': 'curl/7.68.0'  # Pretend to be curl
				}
				r = requests.get('https://sportschatplace.com/nfl-picks/feed/', headers=headers)
				data = xmltodict.parse(r.text)
			except Exception as e:
				print(f"Malformed XML: { r.text }")
				return {
					'error': str(e)
				}		
				
			relevant_items = []
			for item in data['rss']['channel']['item']:
				if 'nfl picks today' in item['title'].lower():
					relevant_items.append(item['content:encoded'])
			return json.dumps(relevant_items)
			
		else:
			raise ValueError(f"{ function_name }is not a valid tool.")
