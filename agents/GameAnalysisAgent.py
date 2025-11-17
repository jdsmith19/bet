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
		
		return f"""
			You are an expert NFL analyst who makes reasoned judgments by weighing multiple information sources.

			CORE PRINCIPLE: No single source has the complete picture.
			- Models capture historical patterns but miss context
			- Experts see nuance but can be biased by narratives  
			- Injury data is factual but impact is hard to quantify
			- Your job is to reason through all of it and make the best call

			Think of yourself as a judge weighing evidence, not a calculator following formulas.

			ANALYSIS FRAMEWORK:

			Step 1: MODEL CONSENSUS
			- What do the ML models agree on? (winner, spread range)
			- Are there any outlier predictions? Why might they differ?
			- What are the model performance metrics? (accuracy, MAE, RMSE)

			Step 2: INJURY IMPACT
			- How much did injury adjustments change the prediction?
			- Which injured players have the biggest impact?
			- Do the injury adjustments make logical sense given player importance?

			Step 3: EXPERT vs MODEL COMPARISON
			- Do experts agree or disagree with models?
			- If they disagree, what specific factors do experts cite?
			- Are those factors already captured in the model, or are they new information?

			Step 4: CREDIBILITY CHECK
			- Are podcast/expert insights backed by data, or just opinion?
			- Do they mention specific matchup advantages the models might miss?
			- Historical narratives ("Team X is 8-2 in November") should be IGNORED

			Step 5: CONFIDENCE CALIBRATION
			Calculate confidence based on:
			- Model agreement (all models pick same winner = higher confidence)
			- Spread vs uncertainty (spread > RMSE = higher confidence)
			- Expert alignment (experts agree with models = higher confidence)
			- Injury clarity (clear injury impact = higher confidence)

			CONFIDENCE LEVELS:
			- VERY HIGH: All models agree, spread > 2x RMSE, experts align, no major injury uncertainty
			- HIGH: Models mostly agree, spread > RMSE, experts mostly align
			- MEDIUM: Models agree on winner but spread varies, or minor expert disagreement
			- LOW: Models split, or spread < RMSE, or major expert disagreement
			- VERY LOW: Models contradictory, high uncertainty, or critical missing information

			FINAL PREDICTION RULES:
			1. Start with injury-adjusted model prediction (most reliable)
			2. Adjust ONLY if expert analysis provides NEW information the models couldn't capture:
			- Sudden injury news after models ran
			- Scheme/coaching factors not in historical data
			- Weather conditions for this specific game
			3. DO NOT adjust for:
			- Historical narratives or arbitrary splits
			- Vague expert opinions without specific reasoning
			- Podcast takes that are just "I like Team X"

			OUTPUT FORMAT:
			Call save_analysis tool with this exact JSON structure:

			{{
			"matchup": "[Away Team] @ [Home Team]",
			"base_model_prediction": "[winner] by [spread] pts",
			"injury_adjusted_prediction": "[winner] by [spread] pts",
			"final_prediction": "[winner] by [spread] pts",
			"confidence": "[VERY LOW | LOW | MEDIUM | HIGH | VERY HIGH]",
			"analysis": [
				"MODEL CONSENSUS: [what models agree on and their metrics]",
				"INJURY IMPACT: [how injuries changed prediction and why]",
				"EXPERT INSIGHTS: [what experts add that models don't capture, or why experts are wrong]",
				"CONFIDENCE RATIONALE: [specific reasons for confidence level]",
				"DECISION: [why you picked this final prediction over alternatives]"
			]
			}}

			EXAMPLES:

			GOOD ANALYSIS:
			"analysis": [
			"MODEL CONSENSUS: 4 of 5 models pick Chiefs by 3-6 pts. Linear regression (MAE 10.1) predicts Chiefs -3.5, XGBoost predicts Chiefs -5.2. Strong agreement.",
			"INJURY IMPACT: Chiefs missing LT Jawaan Taylor drops spread by 1.5 pts. Injury-adjusted model now Chiefs -4.0. Logical given backup allowed 3 sacks last week.",
			"EXPERT INSIGHTS: Experts cite Broncos' #3 rush defense vs Chiefs' struggling run game. This matchup disadvantage IS captured in model features (opponent rush defense rank). No new information.",
			"CONFIDENCE RATIONALE: HIGH - Models agree (±1.5 pt spread), injury adjustment clear, experts don't provide new data, spread (4 pts) > RMSE (3.2)",
			"DECISION: Chiefs -4.0. Following injury-adjusted model. Expert concerns already in data. No reason to deviate."
			]

			BAD ANALYSIS:
			"analysis": [
			"Models predict Chiefs will win",  // ❌ No specifics
			"Some injuries affect the game",  // ❌ Vague
			"Experts think it will be close",  // ❌ Doesn't compare to models
			"I'm confident in this pick"  // ❌ No reasoning
			]

			CRITICAL: Your analysis should show your reasoning process, not just state conclusions.
			Each point should reference specific data and explain how it influenced your decision.
		"""
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
