from dotenv import load_dotenv
import os
import time
import json
import config
from openai import OpenAI
from tools.adjust_aggregates_tools import adjust_data_aggregates

load_dotenv()

class PodcastSummarizationAgent:
	def __init__(self, games, chunk):
		self.games = games
		self.chunk = chunk
		self.summary = None
	
	def run(self):
		"""Main agent loop"""
		start_time = time.time()
		print(f"🎤 Starting Podcast Summarization Agent")
		
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
			print(f"🎤 Podcast Summarization Agent Response")
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
					
					if tool_call.function.name == 'save_summary' and self.summary is not None:
						print(f"🎤 Exiting Podcast Summarization Agent")
						print(f"Completed in { round(time.time() - start_time, 3) }s")
						print(f"{'='*80}\n")
						return self.summary
					
			else:
				messages.append({
					'role': 'user',
					'content': 'You MUST call the tool with a summary in the required format.'
				})
						
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are extracting NFL betting insights from a podcast transcript.

		THIS WEEK'S GAMES:
		{"\n• ".join(self.games)}

		TRANSCRIPT CHUNK:
		{self.chunk}

		CRITICAL RULES:
		1. ONLY extract information explicitly stated in the transcript
		2. DO NOT infer, interpret, or add betting recommendations
		3. DO NOT make up betting lines or spreads
		4. DO NOT draw conclusions - just report what was said
		5. ONLY include games from this week's list above
		6. If hosts contradict themselves, include both statements and describe the contradiction

		EXTRACT (if mentioned):
		- Specific quotes about team performance/outlook
		- Injury mentions (player name + status)
		- Betting lines/spreads (exact numbers given)
		- Weather conditions
		- Matchup advantages/disadvantages
		- Coaching factors
		- Player form (hot/cold streaks)
		- Host predictions or picks

		QUALITY TEST:
		Ask yourself: "Would this insight help predict THIS WEEK'S game?"
		- If it's about current form, injuries, or matchups → INCLUDE
		- If it's a historical fun fact or arbitrary split → EXCLUDE

		Return JSON with SEPARATE keys for each team:

		{{
			// GAME 1
			// FOR AWAY @ HOME, YOU MUST USE THE FULL TEXT FROM THIS WEEK'S GAMES ABOVE
			"Away @ Home": ["Direct insight 1 from transcript", "Direct insight 2 from transcript"],
			// GAME 2
			"Away @ Home": ["Direct insight 1 from transcript", "Direct insight 2 from transcript"],
			// ...
		}}

		EXAMPLE - GOOD:
		{{"Carolina Panthers @ Atlanta Falcons": [
    		"Hosts expressed doubt about Falcons consistency after recent loss",
    		"One host said Panthers are 'still a game back' in AFC North and this game is important for them"
  		]}}

		EXAMPLE - BAD:
		{{
		"Ravens @ Browns": [
			"Ravens +7.5 line as value play",  // ❌ Made up recommendation
			"Market over-reacting to recent win"  // ❌ Interpretation not in transcript
		]
		}}
		
		ONLY include games where the transcript chunk actually discusses them.
		Return {{}} if no games mentioned."""

		return f"""You are analyzing an NFL podcast transcript for betting insights.

		This week's games:

		{ f"\n• ".join(self.games) }

		Transcript chunk:
		
		{ self.chunk }

		If this chunk discusses any of this week's games, extract betting-relevant insights, including:
		- injuries
		- matchups
		- weather
		- coaching factors
		- picks
		- team / matchup sentiments
		- betting lines
		- hot / cold players

		Summarize 1 - 3 key insights for each team.

		Return JSON:

		{{
			// GAME 1
			// FOR AWAY @ HOME, YOU MUST USE THE FULL TEXT FROM THIS WEEK'S GAMES ABOVE
			"Away @ Home": ["key point 1", "key point 2"],
			// GAME 2
			"Away @ Home": ["key point 1", "key point 2"]
			// ...
		}}"""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Analyze the podcast transcript."""
	
	def __get_tool_definition(self):
		"""Tool definition"""
		return [{
			'type': 'function',
			'function': {
				'name': 'save_summary',
				'description': 'Saves podcast transcript summary',
				'parameters': {
					'type': 'object',
					'properties': {
						'summary': {
							'type': 'string',
							'description': """Complete summary JSON as string"""
						}
					},
					'required': ['summary']
				}
			}
		}]
		
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call.function.name
		arguments = tool_call.function.arguments

		if function_name == 'save_summary':
			try:
				self.summary = json.loads(arguments)['summary']
				return "save_summary tool has been called successfully."
			except Exception as e:
				return {
					'error': str(e)
				}		
		
		else:
			raise ValueError(f"{ function_name }is not a valid tool.")
