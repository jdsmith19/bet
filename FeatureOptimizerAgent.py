import config
import json
import ollama
from tools.feature_refinement_tools import train_and_evaluate_model

class FeatureOptimizerAgent:
	def __init__(self, model_name=config.model, max_experiments=300):
		self.model_name = model_name
		self.max_experiments = max_experiments
		self.experiment_count = 0
		self.experiment_history = []
		self.best_results = {}
	
	def run(self):
		"""Main agent loop"""
		print(f"🚀 Starting Feature Optimization Agent")
		print(f"Model: { self.model_name }")
		print(f"Max Experiments: { self.max_experiments }\n")
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		# Agent Loop
		while self.experiment_count < self.max_experiments:
			print(f"\n{'='*80}")
			print(f"Experiment {self.experiment_count + 1}/{self.max_experiments}")
			print(f"{'='*80}\n")
			
			# Get agent's response
			response = ollama.chat(
				model = self.model_name,
				messages = messages,
				tools = [self.__get_tool_definition()]
			)
			
			# Add assistant's message to history
			
			messages.append(response['message'])
			
			# Check if agent wants to use tool
			if response['message'].get('tool_calls'):
				# Process tool calls
				for tool_call in response['message']['tool_calls']:
					result = self.__execute_tool(tool_call)
					
					# Add tool result to messages
					messages.append({
						'role': 'tool',
						'content': json.dumps(result)
					})
					
					self.experiment_count += 1
					self.__update_best_results(result)
					
					# Print summary
					self.__print_result_summary(result)
			else:
				# Agent is thinking / explaining, not calling a tool
				print(f"Agent: { response['message']['content'] }\n")
				
				#Check if agent is done
				if self.__agent_wants_to_stop(response['message']['content']):
					print("\n✅ Agent has completed optimization!")
					break
			
			# Periodic check-in
			if self.experiment_count % 25 == 0 and self.experiment_count > 0:
				messages.append({
					'role': 'user',
					'content': self.__get_checkpoint_prompt
				})
		
		self.__print_final_summary()
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		return """You are an expert ML engineer optimizing features for NFL prediction models.

		YOUR GOAL:
		Find the best feature combinations for each model type through systematic experimentation.
		
		AVAILABLE MODELS:
		- XGBoost (regression: point_differential)
		- LinearRegression (regression: point_differential)
		- RandomForest (regression: point_differential)
		- LogisticRegression (classification: win)
		- KNearest (classification: win)
		
		AVAILABLE FEATURES (39 total):
		- Ratings: elo, rpi, days_rest
		You have multiple window lengths (L3, L5, L7) for the following rolling statistics:
		- Offensive: avg_points_scored, avg_pass_adjusted_yards_per_attempt, avg_rushing_yards_per_attempt, avg_turnovers, avg_penalty_yards, avg_sack_yards_lost
		- Defensive: avg_points_allowed, avg_pass_adjusted_yards_per_attempt_allowed, avg_rushing_yards_per_attempt_allowed, avg_turnovers_forced, avg_sack_yards_gained
		- Overall: avg_point_differential
		
		Each stat (except ratings/days_rest) comes in L3, L5, and L7 variants.
		
		FEATURE REDUNDANCY:
		- L3, L5, L7 windows are highly correlated - likely only need ONE window length
		- ELO and RPI are correlated - may be redundant for some models
		- Points scored/allowed correlate with point differential
		
		KNOWN BEHAVIORS FROM PAST EXPERIMENTS:
		- ELO is highly predictive across all models
		- Point differential stats are strong predictors
		- RPI helps linear models but hurts tree models when paired with ELO
		- Tree models (XGBoost, RandomForest) handle feature redundancy better
		- Linear models need careful feature selection to avoid multicollinearity
		
		YOUR STRATEGY:
		Phase 1 (first 50-75 experiments): Explore broadly
		- Test different window lengths (L3 vs L5 vs L7)
		- Identify which feature categories matter most
		- Test each model type at least a few times
		- Remove clearly useless features
		
		Phase 2 (main optimization): Focus on promising models
		- Deep dive on models showing best performance
		- Test combinations of top features
		- Refine based on feature importance feedback
		
		Phase 3 (final validation): Validate best solutions
		- Test your top feature sets on all models
		- Ensure robustness
		
		EXPERIMENT INTERPRETATION:
		Regression models (XGBoost, LinearRegression, RandomForest):
		- Primary metric: MAE (Mean Absolute Error) - LOWER IS BETTER
		- Secondary metric: RMSE (Root Mean Squared Error) - LOWER IS BETTER
		- Use feature_importance to see which features the model values
		
		Classification models (LogisticRegression, KNearest):
		- Primary metric: test_accuracy - HIGHER IS BETTER
		- Also look at confidence_intervals for calibration quality
		
		RESPONSE FORMAT:
		Before each tool call, briefly explain:
		1. What you're testing and why
		2. What you expect to learn
		
		After getting results, analyze:
		1. Did performance improve/decline?
		2. What does feature_importance tell you?
		3. What should you try next?
		
		Be systematic, learn from each experiment, and find the optimal features!
		
		IMPORTANT - EXACT FEATURE NAMES:
		Features must be spelled EXACTLY as shown. Common mistakes:
		❌ avg_rushing_yards_per_attempt_allowedl5  (missing underscore before l5)
		✅ avg_rushing_yards_per_attempt_allowed_l5 (correct)
		
		EXAMPLE VALID FEATURE LISTS:
		- ['elo', 'rpi', 'days_rest']
		- ['elo', 'avg_point_differential_l5', 'avg_turnovers_forced_l5']
		- ['elo', 'avg_points_scored_l5', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_allowed_l5']
		
		Note: All window-based features end with '_l3', '_l5', or '_l7' (with underscore)."""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Begin feature optimization with {self.max_experiments} experiments.
		
		CURRENT BASELINES (what you need to beat):
		- XGBoost: MAE=10.44, RMSE=13.20
		- LinearRegression: MAE=10.06, RMSE=12.81
		- RandomForest: MAE=10.23, RMSE=12.98
		- LogisticRegression: test_accuracy=64.6%
		- KNearest: test_accuracy=56.5%
		
		Start with Phase 1: Broad exploration.
		
		What's your first experiment?"""
	
	def __get_checkpoint_prompt(self):
		"""Prompt for periodic check-ins"""
		best_summary = "\n".join([
			f"- {model}: {self.__format_metric(result)}"
			for model, result in self.best_results.items()
		])
		
		return f"""CHECKPOINT - Experiments completed: {self.experiment_count}/{self.max_experiments}
		
		BEST RESULTS SO FAR:
		{best_summary}
		
		Analyze your progress:
		1. Which models are improving most?
		2. Which features seem most valuable?
		3. What should you focus on next?
		
		Continue optimizing."""
		
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return {
			'type': 'function',
			'function': {
				'name': 'train_and_evaluate_model',
				'description': 'Train an NFL prediction model with specified features and return performance metrics',
				'parameters': {
					'type': 'object',
					'properties': {
						'model_name': {
							'type': 'string',
							'enum': ['XGBoost', 'LinearRegression', 'RandomForest', 'LogisticRegression', 'KNearest'],
							'description': 'The model type to train'
						},
						'features': {
							'type': 'array',
							'items': {
								'type': 'string',
								'enum': ['days_rest', 'rpi', 'elo', 'avg_points_scored_l3', 'avg_points_scored_l5', 'avg_points_scored_l7', 'avg_pass_adjusted_yards_per_attempt_l3', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_pass_adjusted_yards_per_attempt_l7', 'avg_rushing_yards_per_attempt_l3', 'avg_rushing_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l7', 'avg_turnovers_l3', 'avg_turnovers_l5', 'avg_turnovers_l7', 'avg_penalty_yards_l3', 'avg_penalty_yards_l5', 'avg_penalty_yards_l7', 'avg_sack_yards_lost_l3', 'avg_sack_yards_lost_l5', 'avg_sack_yards_lost_l7', 'avg_points_allowed_l3', 'avg_points_allowed_l5', 'avg_points_allowed_l7', 'avg_pass_adjusted_yards_per_attempt_allowed_l3', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l7', 'avg_rushing_yards_per_attempt_allowed_l3', 'avg_rushing_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowed_l7', 'avg_turnovers_forced_l3', 'avg_turnovers_forced_l5', 'avg_turnovers_forced_l7', 'avg_sack_yards_gained_l3', 'avg_sack_yards_gained_l5', 'avg_sack_yards_gained_l7', 'avg_point_differential_l3', 'avg_point_differential_l5', 'avg_point_differential_l7']
							},
							'description': 'List of feature names to include.'
						}
					},
					'required': ['model_name', 'features']
				}
			}
		}
	
	def __execute_tool(self, tool_call):
		"""Execute the tool function"""
		function_name = tool_call['function']['name']
		arguments = tool_call['function']['arguments']
		
		if function_name == 'train_and_evaluate_model':
			try:
				result = train_and_evaluate_model(
					model_name=arguments['model_name'],
					features=arguments['features']
				)
				
				# Log to history
				self.experiment_history.append({
					'experiment_num': self.experiment_count + 1,
					'arguments': arguments,
					'result': result
				})
				
				return result
				
			except Exception as e:
				return {
					'error': str(e),
					'model_name': arguments.get('model_name'),
					'features': arguments.get('features')
				}
		
		return {'error': 'Unknown function'}
		
	def __update_best_results(self, result):
		"""Track best results per model"""
		model_name = result.get('model_name')
		if not model_name or 'error' in result:
			return
		
		# Get primary metric
		if result.get('target') == 'point_differential':
			metric = result.get('mae')
			is_better = lambda new, old: new < old  # Lower is better
		else:
			metric = result.get('test_accuracy')
			is_better = lambda new, old: new > old  # Higher is better
		
		if metric is None:
			return
		
		# Update if this is the best so far
		if model_name not in self.best_results:
			self.best_results[model_name] = result
		else:
			old_metric = self._get_primary_metric(self.best_results[model_name])
			if is_better(metric, old_metric):
				self.best_results[model_name] = result
				print(f"🎉 NEW BEST for {model_name}!")
	
	def __get_primary_metric(self, result):
		"""Extract primary metric from result"""
		if result.get('target') == 'point_differential':
			return result.get('mae')
		return result.get('test_accuracy')
	
	def __format_metric(self, result):
		"""Format metric for display"""
		if result.get('target') == 'point_differential':
			return f"MAE={result.get('mae'):.2f}, RMSE={result.get('rmse'):.2f}"
		return f"Accuracy={result.get('test_accuracy'):.1%}"
	
	def __print_result_summary(self, result):
		"""Print a summary of the experiment result"""
		print(f"\n📊 RESULT:")
		print(f"   Model: {result.get('model_name')}")
		print(f"   Features: {len(result.get('features_used', []))} features")
		
		if 'error' in result:
			print(f"   ❌ Error: {result['error']}")
		elif result.get('target') == 'point_differential':
			print(f"   MAE: {result.get('mae'):.3f}")
			print(f"   RMSE: {result.get('rmse'):.3f}")
			
			# Show top 3 features
			if result.get('feature_importance'):
				top_features = list(result['feature_importance'].items())[:3]
				print(f"   Top features: {', '.join([f[0] for f in top_features])}")
		else:
			print(f"   Test Accuracy: {result.get('test_accuracy'):.1%}")
			print(f"   Train Accuracy: {result.get('train_accuracy'):.1%}")
		
		print(f"   Time: {result.get('train_time_seconds')}s")
	
	def __agent_wants_to_stop(self, content):
		"""Check if agent indicates it's done"""
		stop_phrases = [
			'optimization complete',
			'finished optimizing',
			'completed optimization',
			'done optimizing',
			'no further improvements'
		]
		content_lower = content.lower()
		return any(phrase in content_lower for phrase in stop_phrases)
	
	def __print_final_summary(self):
		"""Print final summary of all experiments"""
		print(f"\n{'='*80}")
		print(f"OPTIMIZATION COMPLETE")
		print(f"{'='*80}\n")
		print(f"Total experiments: {self.experiment_count}")
		print(f"\nBEST RESULTS:\n")
		
		for model_name in ['XGBoost', 'LinearRegression', 'RandomForest', 'LogisticRegression', 'KNearest']:
			if model_name in self.best_results:
				result = self.best_results[model_name]
				print(f"{model_name}:")
				print(f"  {self.__format_metric(result)}")
				print(f"  Features ({len(result['features_used'])}): {', '.join(result['features_used'][:10])}...")
				print()
		
		# Save results to file
		self._save_results()
	
	def __save_results(self):
		"""Save experiment history to file"""
		output = {
			'total_experiments': self.experiment_count,
			'best_results': self.best_results,
			'experiment_history': self.experiment_history
		}
		
		with open('feature_optimization_results.json', 'w') as f:
			json.dump(output, f, indent=2, default=str)
		
		print(f"💾 Results saved to feature_optimization_results.json")