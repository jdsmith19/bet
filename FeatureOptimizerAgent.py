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
		self.empty_response_count = 0
		self.max_consecutive_empty_responses = 2
	
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
			print(f"📊 CONTEXT INSPECTION - Experiment {self.experiment_count}")
			print(f"{'='*80}")
			
			print(f"Total messages: {len(messages)}")

			# Estimate token count (rough)
			total_chars = sum(len(str(msg)) for msg in messages)
			estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
			print(f"Estimated tokens: {estimated_tokens:,}")
			
			if estimated_tokens > 100000:
				print(f"⚠️  WARNING: Very large context ({estimated_tokens:,} tokens)")
				print(f"   This may cause slowness or model confusion!")
			
			print(f"{'='*80}\n")
			MAX_MESSAGES = 60
			if len(messages) > MAX_MESSAGES:
				print(f"Trimming messages...")
				best_summary = f"""BEST RESULTS SO FAR (Experiment {self.experiment_count}:
				
				{ self.__format_all_best_results() }
				
				Continue optimization using your available insights."""
				
				print(f"Reminding the agent of the { best_summary }")

				messages = [
					messages[0],  # System prompt
					{'role': 'user', 'content': best_summary},
					*messages[-(MAX_MESSAGES-2):]  # Recent experiments
				]
			
			# Get agent's response
			response = ollama.chat(
				model = self.model_name,
				messages = messages,
				tools = [self.__get_tool_definition()]
			)
			# Add assistant's message to history
			if (not response['message']['content'] or len(response['message']['content']) < 10) and not response['message'].get('tool_calls'):
				intervention_message = f"You appear to be stuck. You gave me an empty response. Immediately plan and execute your next experiment."
				print(intervention_message)
				messages.append({
					'role': 'user',
					'content': intervention_message
				})
			else:
				messages.append(response['message'])
			
			# Check if agent wants to use tool
			if response['message'].get('tool_calls'):
				print(f"\n{'='*80}")
				print(f"Experiment {self.experiment_count + 1} / {self.max_experiments}")
				print(f"{'='*80}\n")
				# Process tool calls
				print(f"Agent: { response['message']['content']}\n")
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
			elif response['message']['content'] or len(response['message']['content']) > 10:
				# Agent is thinking / explaining, not calling a tool
				print(f"\n{'='*80}")
				print(f"Agent is thinking...")
				print(f"\n{'='*80}\n")
				print(f"Agent: { response['message']['content'] }")
				
				#Check if agent is done
				if self.__agent_wants_to_stop(response['message']['content']):
					print("\n✅ Agent has completed optimization!")
					break
			
			# Periodic check-in
			if self.experiment_count % 25 == 0 and self.experiment_count > 0:
				checkpoint_prompt = self.__get_checkpoint_prompt()
				print(f"\n{'='*80}")
				print(f"CHECKPOINT")
				print(f"\n{'='*80}\n")
				print(checkpoint_prompt)
				messages.append({
					'role': 'user',
					'content': self.__get_checkpoint_prompt()
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
		- Ratings: elo, rpi, days_rest (IMPORTANT: DO NOT PRE-PEND RATINGS FEATURES WITH team_a or team_b)
		You have multiple window lengths (L3, L5, L7) OR home and away splits (home, away) for the following rolling statistics:
		- Offensive: avg_points_scored, avg_pass_adjusted_yards_per_attempt, avg_rushing_yards_per_attempt, avg_turnovers, avg_penalty_yards, avg_sack_yards_lost
		- Defensive: avg_points_allowed, avg_pass_adjusted_yards_per_attempt_allowed, avg_rushing_yards_per_attempt_allowed, avg_turnovers_forced, avg_sack_yards_gained
		- Overall: avg_point_differential
		- Usage: These can be expressed as avg_points_scored_l3 or avg_points_scored_home
								
		YOUR STRATEGY:
		Plan 5 experiments at a time and execute them sequentially by calling the tool. After executing those 5 experiments, analyze the results and plan the next 5 experiments.
		
		Phase 1 (first 100-150 experiments): Explore broadly
		- Test different window lengths (L3 vs L5 vs L7)
		- Identify which feature categories matter most
		- Test each model type at least a few times
		- Remove clearly useless features
		
		Phase 2 (main optimization): Focus on promising model and feature set combinations
		- Deep dive on those models showing the best performance
		- Test combinations of top features
		- Refine based on feature importance feedback
		
		Phase 3 (final validation): Validate best solutions
		- Test your feature sets on all models
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
		
		CRITICAL INSTRUCTIONS:
		- You MUST call the train_and_evaluate_model tool to run experiments
		- NEVER generate fake JSON results
		- NEVER pretend you ran an experiment
		- WAIT for real tool results before analyzing
		- If you see JSON in your response, you are doing it wrong
		
		After analyzing results, immediately call the tool for your next experiment.
		Do not write fake results. Do not role-play. Use the tool.
		
		Be exhaustive, try lots of different possible combination types. If you are very confident that the results cannot improve and have found the best possible combinations, start your message with "optimization complete"
		"""
			
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
								'enum': ['days_rest', 'rpi_rating', 'elo_rating', 'avg_points_scored_l3', 'avg_points_scored_l5', 'avg_points_scored_l7', 'avg_pass_adjusted_yards_per_attempt_l3', 'avg_pass_adjusted_yards_per_attempt_l5', 'avg_pass_adjusted_yards_per_attempt_l7', 'avg_rushing_yards_per_attempt_l3', 'avg_rushing_yards_per_attempt_l5', 'avg_rushing_yards_per_attempt_l7', 'avg_turnovers_l3', 'avg_turnovers_l5', 'avg_turnovers_l7', 'avg_penalty_yards_l3', 'avg_penalty_yards_l5', 'avg_penalty_yards_l7', 'avg_sack_yards_lost_l3', 'avg_sack_yards_lost_l5', 'avg_sack_yards_lost_l7', 'avg_points_allowed_l3', 'avg_points_allowed_l5', 'avg_points_allowed_l7', 'avg_pass_adjusted_yards_per_attempt_allowed_l3', 'avg_pass_adjusted_yards_per_attempt_allowed_l5', 'avg_pass_adjusted_yards_per_attempt_allowed_l7', 'avg_rushing_yards_per_attempt_allowed_l3', 'avg_rushing_yards_per_attempt_allowed_l5', 'avg_rushing_yards_per_attempt_allowed_l7', 'avg_turnovers_forced_l3', 'avg_turnovers_forced_l5', 'avg_turnovers_forced_l7', 'avg_sack_yards_gained_l3', 'avg_sack_yards_gained_l5', 'avg_sack_yards_gained_l7', 'avg_point_differential_l3', 'avg_point_differential_l5', 'avg_point_differential_l7']
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
			metric = result.get('mean_absolute_error')
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
			# FOR DEBUGGING THE NASTY BUG I HAD BEFORE
			old_result = self.best_results[model_name]
			# print(f"\n🔍 DEBUG: Comparing to previous best:")
			# print(f"   Previous best keys: {list(old_result.keys())}")
			# print(f"   Previous best: {old_result}")
			#  
			old_metric = self.__get_primary_metric(old_result)
			# print(f"   old_metric returned: {old_metric}")
			# print(f"   current metric: {metric}")
			if old_metric is None:
				print("   ⚠️  old_metric is None! Replacing.")
				self.best_results[model_name] = result
			elif is_better(metric, old_metric):
				self.best_results[model_name] = result
				print(f"🎉 NEW BEST for {model_name}!")
	
	def __get_primary_metric(self, result):
		"""Extract primary metric from result"""
		if result.get('target') == 'point_differential':
			return result.get('mean_absolute_error')
		return result.get('test_accuracy')
	
	def __format_metric(self, result):
		"""Format metric for display"""
		if result.get('target') == 'point_differential':
			return f"MAE={result.get('mean_absolute_error'):.2f}, RMSE={result.get('root_mean_squared_error'):.2f}"
		return f"Accuracy={result.get('test_accuracy'):.1%}"
	
	def __print_result_summary(self, result):
		"""Print a summary of the experiment result"""
		print(f"\n📊 RESULT:")
		print(f"   Model: {result.get('model_name')}")
		print(f"   Features: {len(result.get('features_used', []))} features")
		
		if 'error' in result:
			print(f"   ❌ Error: {result['error']}")
		elif result.get('target') == 'point_differential':
			print(f"   MAE: {result.get('mean_absolute_error'):.3f}")
			print(f"   RMSE: {result.get('root_mean_squared_error'):.3f}")
			
			# Show top 3 features
			if result.get('feature_importance'):
				top_features = list(result['feature_importance'].items())[:3]
				print(f"   Top features: {', '.join([f[0] for f in top_features])}")
		else:
			print(f"   Test Accuracy: {result.get('test_accuracy'):.1%}")
			print(f"   Train Accuracy: {result.get('train_accuracy'):.1%}")
		
		print(f"   Time: {result.get('train_time_in_seconds')}s")
	
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
	
	def __format_all_best_results(self):
		"""Format all best results for context summary"""
		if not self.best_results:
			return "No results yet."
		
		lines = []
		
		# Group by model type
		regression_models = ['XGBoost', 'LinearRegression', 'RandomForest']
		classification_models = ['LogisticRegression', 'KNearest']
		
		# Format regression models
		if any(model in self.best_results for model in regression_models):
			lines.append("REGRESSION MODELS (predicting point_differential):")
			for model_name in regression_models:
				if model_name in self.best_results:
					result = self.best_results[model_name]
					mae = result.get('mean_absolute_error')
					rmse = result.get('root_mean_squared_error')
					features = result.get('features_used', [])
					
					lines.append(f"\n{model_name}:")
					lines.append(f"  MAE: {mae:.3f}, RMSE: {rmse:.3f}")
					lines.append(f"  Features ({len(features)}): {', '.join(features[:10])}")
					
					# Show top 3 features if available
					if result.get('feature_importance'):
						top_3 = list(result['feature_importance'].items())[:3]
						top_features = ', '.join([f[0] for f in top_3])
						lines.append(f"  Top features: {top_features}")
		
		# Format classification models
		if any(model in self.best_results for model in classification_models):
			lines.append("\n\nCLASSIFICATION MODELS (predicting win):")
			for model_name in classification_models:
				if model_name in self.best_results:
					result = self.best_results[model_name]
					test_acc = result.get('test_accuracy')
					train_acc = result.get('train_accuracy')
					features = result.get('features_used', [])
					
					lines.append(f"\n{model_name}:")
					lines.append(f"  Test Accuracy: {test_acc:.1%}, Train Accuracy: {train_acc:.1%}")
					lines.append(f"  Features ({len(features)}): {', '.join(features[:10])}")
					
					# Show confidence calibration if available
					if result.get('confidence_intervals'):
						best_interval = result['confidence_intervals'][-1]  # Highest confidence
						for key, value in best_interval.items():
							threshold = key.split('_')[-1]
							acc = value.get('accuracy', 0)
							count = value.get('count_predictions', 0)
							lines.append(f"  At >{threshold} confidence: {acc:.1%} accuracy ({count} predictions)")
							break  # Just show one
		
		# Add key insights
		lines.append("\n\nKEY INSIGHTS:")
		
		# Find best overall regression model
		best_regression = None
		best_mae = float('inf')
		for model in regression_models:
			if model in self.best_results:
				mae = self.best_results[model].get('mean_absolute_error', float('inf'))
				if mae < best_mae:
					best_mae = mae
					best_regression = model
		
		if best_regression:
			lines.append(f"- Best regression: {best_regression} (MAE={best_mae:.3f})")
		
		# Find best classification model
		best_classification = None
		best_acc = 0
		for model in classification_models:
			if model in self.best_results:
				acc = self.best_results[model].get('test_accuracy', 0)
				if acc > best_acc:
					best_acc = acc
					best_classification = model
		
		if best_classification:
			lines.append(f"- Best classification: {best_classification} (Accuracy={best_acc:.1%})")
		
		# Common features across best models
		all_features = set()
		for result in self.best_results.values():
			all_features.update(result.get('features_used', []))
		
		if all_features:
			# Find most common features
			feature_counts = {}
			for result in self.best_results.values():
				for feature in result.get('features_used', []):
					feature_counts[feature] = feature_counts.get(feature, 0) + 1
			
			common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]
			common_list = ', '.join([f[0] for f in common_features])
			lines.append(f"- Most used features: {common_list}")
		
		return '\n'.join(lines)
	
	def _save_results(self):
		"""Save experiment history to file"""
		output = {
			'total_experiments': self.experiment_count,
			'best_results': self.best_results,
			'experiment_history': self.experiment_history
		}
		
		with open('feature_optimization_results.json', 'w') as f:
			json.dump(output, f, indent=2, default=str)
		
		print(f"💾 Results saved to feature_optimization_results.json")