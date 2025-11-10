import config
import json
import ollama
import config
from helpers.Lookup import Lookup
from tools.adjust_aggregates_tools import adjust_data_aggregates
from DataAggregate.DataAggregate import DataAggregate

class InjuryAdjustmentAgent:
	def __init__(self, injury_report, current_aggregates = None):
		self.injury_report = injury_report
		if not current_aggregates:
			self.aggregates = DataAggregate(config.odds_api_key)
		else:
			self.aggregates = current_aggregates
	
	def run(self):
		"""Main agent loop"""
		print(f"🏭 Starting Injury Adjustment Agent")
		
		finished = False
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		empty_responses = 0
		
		while not finished:
			if empty_responses >= 1:
				messages.append({
					'role': 'user',
					'content': 'You MUST CALL the adjust_data_aggregates tool with data in the OUTPUT FORMAT.'
				})
				
			# Get agent's response
			response = ollama.chat(
				model = config.adjustment_model,
				messages = messages,
				tools = self.__get_tool_definition()
			)
			msg = response['message']
			messages.append(response['message'])
			
			if not msg.get('thinking') and not msg.get('content') and not msg.get('tool_calls'):
				empty_responses += 1
			
			print(f"\n{'='*80}")
			print(f"🏭 Injury Adjustment Agent Response")
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
			elif msg.content:							
				if 'injury report complete' in msg.content.lower():
					print(f"🏭 Exiting Injury Adjustment Agent")
					finished = True
					return self.aggregates
						
			print(f"{'='*80}\n")
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are an injury adjustment specialist. You have ONE job.
		
		INPUT: 
		- An exhaustive injury report for NFL teams
		
		OUTPUT:
		- An object from a custom DataAggregate class
		
		TASK:
		For each team in the injury report:
		1. Read the injury report
		2. Create 1-5 adjustment objects per team based on injuries
		3. Add to your output list
		4. Call the adjust_data_aggregates tool with the list of objects you create. YOU MUST USE THE LIST FORMAT SHOWN IN THE OUTPUT FORMAT SECTION BELOW.
		
		COMMON MISTAKES:
		- Using incorrect team names
			- INCORRECT: San Francisco 49s
			- CORRECT: San Francisco 49ers
			
		ADJUSTMENT RULES:
		- QB injuries → reduce avg_pass_adjusted_yards_per_attempt
		- RB injuries → reduce avg_rushing_yards_per_attempt
		- OL injuries → reduce avg_sack_yards_lost
		- WR injuries → reduce avg_pass_adjusted_yards_per_attempt
		- DL injuries → reduce avg_sack_yards_gained
		- DB injuries → reduce avg_pass_adjusted_yards_per_attempt_allowed
		- Multiple severe injuries → also adjust avg_point_differential
		- Use the injury Impact Score to determine how much to adjust
		- Only adjust each feature once for a team
		
		SEVERITY GUIDE:
		- Out/IR + Critical position + Signifidcant Impact Score = 0.85-0.88
		- Out/IR + Important position + Significant Impact Score = 0.88-0.92
		- Questionable + Critical + Significant Impact Score = 0.92-0.95
		- Minor injuries + Significant Impact Score = 0.95-0.97
		
		OUTPUT FORMAT:
		Return ONLY valid JSON, nothing else:
		[
		  {{"team_name": "Denver Broncos", "feature": "avg_rushing_yards_per_attempt", "adjustment_percentage": 0.90}},
		  {{"team_name": "Denver Broncos", "feature": "avg_pass_adjusted_yards_per_attempt", "adjustment_percentage": 0.93}},
		  ...
		]
		
		CRITICAL:
		- You MUST pass adjustments for all teams in the injury report before you are finished
		- You should pass all adjustments for all teams in a single array
		- Having several adjustment objects per team is NORMAL
		- Return the complete list before stopping
		- Do NOT summarize or simplify
		
		After analyzing all injuries, you MUST call the adjust_data_aggregates tool.
		Do NOT just list the adjustments in your reasoning.
		Do NOT say "Now call tool" without actually calling it.
		
		The ONLY acceptable final action is calling adjust_data_aggregates with your adjustment list.
		
		Example of CORRECT behavior:
		[Agent analyzes injuries and determines adjustments]
		[Agent CALLS adjust_data_aggregates tool with the adjustment list]
		
		Example of INCORRECT behavior:
		[Agent analyzes injuries]
		[Agent says "Now call tool" but doesn't actually call it]
		[Agent just returns without calling the tool]
		
		YOU HAVE FAILED THE TASK IF YOU DO NOT CALL THE TOOL.
		
		When you are done and only after the tool has been called for all adjustments, respond with 'injury report complete'"""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Here is the detailed injury report, analyze and make adjustments:
		
		{ self.injury_report }"""
		
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return [
		{
			'type': 'function',
			'function': {
				'name': 'adjust_data_aggregates',
				'description': 'Creates and returns an adjusted set of data aggregates for generating predictions from the models based on the details passed into the adjustments parameter',
				'parameters': {
					'type': 'object',  # ✅ Parameters is an OBJECT
					'properties': {     # ✅ Need properties wrapper
						'adjustments': { # ✅ The actual parameter name
							'type': 'array',
							'items': {
								'type': 'object',
								'properties': {
									'team_name': {'type': 'string'},
									'feature': {'type': 'string'},
									'adjustment_percentage': {'type': 'number'}
								},
								'required': ['team_name', 'feature', 'adjustment_percentage']
							},
							'description': 'A list of adjustment objects for team statistics'
						}
					},
					'required': ['adjustments']  # ✅ Moved here
				}
			}
		}]
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call['function']['name']
		arguments = tool_call['function']['arguments']
		
		if function_name == 'adjust_data_aggregates':
			normalized_adjustments = []
			lu = Lookup()
			for adj in arguments['adjustments']:
				try:
					team_abbr = lu.injury_report_to_pfr(adj['team_name'])
				except:
					try:
						team_abbr = lu.odds_api_team_to_pfr_team(adj['team_name'])
					except Exception as e:
						return {
							'error': str(e)
						}

						
				normalized_adjustments.append({ 'team_name': team_abbr, 'feature': adj['feature'], 'adjustment_percentage': adj['adjustment_percentage'] })
			try:
				result = adjust_data_aggregates(
					adjustments = normalized_adjustments,
					da = self.aggregates
				)
				self.aggregates = result
			except Exception as e:
				return {
					'error': str(e)
				}
		
		elif function_name == 'generate_html_report':
			return generate_html_report(arguments['html'])