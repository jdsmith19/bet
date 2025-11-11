import config
import json
import ollama
from tools.feature_refinement_tools import train_and_evaluate_model

class OptimizerPlanningAgent:
	def __init__(self, historical_results, phase):
		self.model = config.planning_model
		self.max_consecutive_empty_responses = 1
		self.phase = phase
		self.historical_results = historical_results
	
	def run(self):
		"""Main agent loop"""
		print(f"\n📓 Starting Optimizer Planning Agent")
		print(f"Model: { self.model }")
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		finished = False
		# Agent Loop
		while not finished:
			# Estimate token count (rough)
			total_chars = sum(len(str(msg)) for msg in messages)
			estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars

			print(f"{'='*80}")
			print(f"📊 CONTEXT INSPECTION")
			print(f"Total messages: {len(messages)}")
			print(f"Estimated tokens: { estimated_tokens }")
			print(f"{'='*80}")

						
			# Get agent's response
			response = ollama.chat(
				model = self.model,
				messages = messages
			)
			
			msg = response['message']
			
			print(f"{'='*80}")
			print(f"📓 Optimizer Planning Agent Response)")
			print(f"{'='*80}")
			
			# Show the thinking (chain-of-thought)
			if msg.get('thinking'):
				print(f"🧠 REASONING:")
				print(f"{ msg['thinking'] }")
				print(f"{'='*80}")
				
			if msg.get('content'):
				print(f"💬 EXPLANATION:")
				print(f"{ msg['content'] }")
				print(f"{'='*80}")
			
			missing_response = (not msg.get('thinking') and not msg.get('content'))
			
			# Add assistant's message to history
			if missing_response:
				intervention_message = f"You appear to be stuck. You need to plan the next 10 experiments."
				print(f"👦🏻 USER: { intervention_message }")
				messages.append({
					'role': 'user',
					'content': intervention_message
				})
			
			else:
				messages.append({
					'role': 'user',
					'message': msg
				})
			
			if ("'status': 'complete'" in msg['content'].lower() or '"status": "complete"' in msg['content'].lower()):
				validation = self.__validate_response(msg['content'])
				if validation == "True":
					finished = True
					print(f"📓 Exiting Optimizer Planning Agent\n")
					return msg['content']
				else:
					messages.append({
						'role': user,
						'message': validation
					})
						
	def __get_system_prompt(self):
		"""System prompt with full context"""
		return f"""You are an expert ML engineer optimizing features for NFL prediction models.
		
		YOUR GOAL:
		Find the best feature combinations for each model type through systematic experimentation. Based on the information available to you, identify the next 10 experiments to run.
		
		AVAILABLE MODELS:
		- XGBoost (regression: point_differential)
		- LinearRegression (regression: point_differential)
		- RandomForest (regression: point_differential)
		- LogisticRegression (classification: win)
		- KNearest (classification: win)
		
		AVAILABLE FEATURES (63 total):
		- Ratings: elo_rating, rpi_rating, days_rest (IMPORTANT: DO NOT PRE-PEND RATINGS FEATURES WITH team_a or team_b)
		You have multiple window lengths (L3, L5, L7) OR home and away splits (home, away) for the following rolling statistics:
		- Offensive: avg_points_scored, avg_pass_adjusted_yards_per_attempt, avg_rushing_yards_per_attempt, avg_turnovers, avg_penalty_yards, avg_sack_yards_lost
		- Defensive: avg_points_allowed, avg_pass_adjusted_yards_per_attempt_allowed, avg_rushing_yards_per_attempt_allowed, avg_turnovers_forced, avg_sack_yards_gained
		- Overall: avg_point_differential
		- Usage: These can be expressed as avg_points_scored_l3 or avg_points_scored_home
		- DO NOT CALL THESE FEATURES WITHOUT APPENDING EITHER A WINDOW OR LOCATION SPLIT
			- RIGHT: avg_point_differential_l3, avg_point_differential_home
			- WRONG: avg_point_differential, points_scored
		- DO NOT COMBINE L3 / L5 / L7 or HOME / AWAY
			- RIGHT: avg_points_scored_l3, avg_points_scored_home
			- WRONG: avg_points_scored_l3_home, avg_points_scored_l5_away
		- DO NOT TRY TO DUPLICATE FEATURES
			- RIGHT: ['avg_points_scored_l3', 'rpi_rating']
			- WRONG: ['rpi_rating', 'rpi_rating']
								
		YOUR STRATEGY:
		Plan experiments in batches of 10. You should craft experiments that will help you gather new insights and to try to get the best possible outcome for each model. There are 4 phases to the exploration, you are currently in PHASE { self.phase }.
		
		{ self.__get_phase_instructions() }
				
		ALL PREVIOUS RESULTS:
		
		The following includes the current best results, number of experiments run, and the details of the last 50 experiments.
		
		{ self.historical_results }
		
		IDEAS:
		Previous iterations have found the following things provide strong positive signals.
		- Home and away metrics are very important. Don't ignore using the _home and _away suffixes.
		- You should explore home and away metrics in conjunction with each other for the same feature.
			- Since only one team in a matchup can be home or away, the goal is to compare the teams corresponding performance when predicting the matchup
		- rpi_rating and elo_rating are very strong signals
		- average_point_differential and points_scored give good signals to a teams overall historic performance
		- Be sure to focus on the top features from the result. Trying different combinations and looking at the top results will help you understand which features have the biggest impact.
		
		EXPERIMENT INTERPRETATION:
		Regression models (XGBoost, LinearRegression, RandomForest):
		- Primary metric: MAE (Mean Absolute Error) - LOWER IS BETTER
		- Secondary metric: RMSE (Root Mean Squared Error) - LOWER IS BETTER
		- Use feature_importance to see which features the model values
		
		Classification models (LogisticRegression, KNearest):
		- Primary metric: test_accuracy - HIGHER IS BETTER
		- Also look at confidence_intervals for calibration quality
		
		RESPONSE FORMAT:
		When you have 10 experiments ready to execute, return the details of the experiments to be run using the following format. ONLY RESPOND WITH JSON IN THIS FORMAT. DO NOT ADD ANY OTHER TEXT. IF YOU ARE THINKING AND NOT RETURNING THE FINAL EXPERIMENT DETAILS, DO NOT INCLUDE 'status': 'complete' IN YOUR RESPONSE.
		
		{{
			"status": "complete",
			"experiments": [
				{{
					"model": [Which model to execute the experiment against. MUST be one of the AVAILABLE MODELS],
					"features": [Which features to use in the model training. MUST be one of the AVAILABLE FEATURES],
					
				}}
			]
		}}"""
	
	def __validate_response(self, response):
		try:
			r = json.loads(response)
			try:
				r['status'] == 'complete':
			except Exception as e:
				return "Your response did not include \"status\": \"complete\". Try again."
				try:
					for experiment in r['experiments']:
						for feature in experiment['features']:
							if "team_a" in feature or "team_b" in feature:
								return "Do not include \"team_a\" or \"team_b\" in your feature names. Try again."
		except Exception as e:
			return "Your response was not valid JSON. Try again."
		return "True"
			
	def __get_phase_instructions(self):
		if self.phase == 1:
			return """PHASE 1 INSTRUCTIONS: SEEMINGLY RANDOM
			- Test all different types of combinations of features.
			- Use random combinations, don't try to correlate findings yet. That will happen in the next Phase.
			- You can have as many features in a single tool call as you want.
			- Experiment with both very small and very large feature sets. Try an experiment with all 61 features in a tool call."""
		
		elif self.phase == 2:
			return """PHASE 2 INSTRUCTIONS: EXPLORE BROADLY
			- Use the previous findings to start testing combinations based on features that were most promising.
			- Test different window lengths (L3 vs L5 vs L7)
			- Test home/away splits vs window lengths
			- Identify which feature categories matter most
			- Test each model type multiple times with different feature combinations
			- Try unusual combinations to discover hidden patterns
			- Even if results plateau, keep trying new approaches - there may be untested combinations"""
		
		elif self.phase == 3:
			return """PHASE 3 INSTRUCTIONS: DEEP OPTIMIZATION
			- Focus on getting the best result from every model
			- Test fine-grained variations of successful feature sets
			- Add/remove individual features to find the perfect balance
			- Test different ratios of offensive vs defensive features
			- Experiment with minimal feature sets (fewer features, similar performance)
			- Try combining different window lengths for different feature types
			- After getting results, analyze:
				1. Did performance improve/decline?
				2. What does feature_importance tell you?
			- Even if results plateau, keep trying new approaches - there may be untested combinations"""
		
		elif self.phase == 4:
			return """PHASE 4 INSTRUCTIONS: FINAL REFINEMENT AND VALIDATION
			- Test your best feature sets on all models for comparison
			- Try last-minute variations and edge cases
			- Validate robustness across model types
			- Push for incremental improvements in your best models
			- Even if results plateau, keep trying new approaches - there may be untested combinations"""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Plan your next 10 experiments"""