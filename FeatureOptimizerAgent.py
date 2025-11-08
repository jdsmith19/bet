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
			MAX_MESSAGES = 200
			if len(messages) > MAX_MESSAGES:
				print(f"Trimming messages...")
				best_summary = f"""BEST RESULTS SO FAR (Experiment {self.experiment_count}:
				
				{ self.__format_all_best_results() }
				
				Continue optimization using your available insights."""
				
				filtered_messages = [messages[0]]
				for msg in messages[1:]:
					if msg.get('role') == 'user' and 'BEST RESULTS SO FAR' in msg.get('content', ''):
						continue  # Skip old summaries
					filtered_messages.append(msg)
				
				print(f"Reminding the agent of the { best_summary }")
				
				messages = [
					filtered_messages[0],  # System
					{'role': 'user', 'content': best_summary},  # NEW summary only
					*filtered_messages[-(MAX_MESSAGES-50):]  # Last 38 messages (no old summaries)
				]
			
			# Get agent's response
			response = ollama.chat(
				model = self.model_name,
				messages = messages,
				tools = [self.__get_tool_definition()]
			)
			msg = response['message']
			
			print(f"\n{'='*80}")
			print(f"🤖 Agent Response (Experiment {self.experiment_count + 1})")
			print(f"{'='*80}\n")
			
			# Show the thinking (chain-of-thought)
			if msg.thinking:
				print(f"🧠 REASONING:")
				print(f"{msg.thinking}\n")
				
			if msg.content:
				print(f"💬 EXPLANATION:")
				print(f"{msg.content}\n")
			
			# Show tool calls
			if msg.tool_calls:
				for tc in msg.tool_calls:
					args = tc.function.arguments
					print(f"🔧 ACTION:")
					print(f"   Model: {args.get('model_name')}")
					print(f"   Features ({len(args.get('features', []))}): {args.get('features')}\n")
			
			print(f"{'='*80}\n")
			
			missing_response = (not msg.thinking or len(msg.thinking) < 10) and (not msg.content or len(msg.content) < 10) and (not msg.tool_calls or len(msg.tool_calls) < 10)				
			# Add assistant's message to history
			if (missing_response):
				intervention_message = f"You appear to be stuck. You gave me an empty response. Immediately plan and execute your next experiment."
				print(intervention_message)
				messages.append({
					'role': 'user',
					'content': intervention_message
				})
			else:
				messages.append(response['message'])
				has_message = True
			
			# Check if agent wants to use tool
			if msg.get('tool_calls'):
				print(f"\n{'='*80}")
				print(f"Experiment {self.experiment_count + 1} / {self.max_experiments}")
				print(f"{'='*80}\n")
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
			elif has_message:
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
		return f"""You are an expert ML engineer optimizing features for NFL prediction models.
		
			YOUR GOAL:
			Find the best feature combinations for each model type through systematic experimentation.
			You have {self.max_experiments} experiments available - USE THEM ALL to find the absolute best combinations.
			
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
			Plan experiments in batches of 5 to 10 and execute them sequentially by calling the tool. Continue experimenting until you reach the maximum experiment count.
			
			Phase 1 (experiments 1 - 100): Seemingly random
			- Test all different types of combinations of features
			- Don't test one feature at a time, that will take too long. Batch features into at least 5 at a time.
			- Use random combinations, don't try to correlate findings yet. That will happen in the next Phase
			- You can have as many features in a single tool call as you want, but the model will likely begin to overfit with too many features
			- Experiment with very large feature sets. Try an experiment with all 61 features in a tool call.
			- You are not complete with Phase 1 until you have tested every possible feature across every model
				- This can be done in as few as 5 or as many as 315 experiments
				- Don't optimize for as few experiments as possible or you will not have a lot of information when you proceed to Phase 2
			
			Phase 2 (experiments 100-200): Explore broadly
			- Use your findings to start testing combinations based on features that were most promising
			- Test different window lengths (L3 vs L5 vs L7)
			- Test home/away splits vs window lengths
			- Identify which feature categories matter most
			- Test each model type multiple times with different feature combinations
			- Try unusual combinations to discover hidden patterns
			
			Phase 3 (experiments 201-400): Deep optimization
			- Focus on getting the best result from every model
			- Test fine-grained variations of successful feature sets
			- Add/remove individual features to find the perfect balance
			- Test different ratios of offensive vs defensive features
			- Experiment with minimal feature sets (fewer features, similar performance)
			- Try combining different window lengths for different feature types
			
			Phase 4 (experiments 401-{self.max_experiments}): Final refinement and validation
			- Test your best feature sets on all models for comparison
			- Try last-minute variations and edge cases
			- Validate robustness across model types
			- Push for incremental improvements in your best models
			
			IMPORTANT: Do NOT stop early! Even small improvements (0.01 MAE reduction, 0.1% accuracy gain) are valuable.
			Keep experimenting with new combinations until you hit the experiment limit.
			
			CHECKPOINTS: Every 25 experiments, you will hit a checkpoint where your current best findings will be summarized. When analyzing the data from the checkpoint, do not forget which Phase you are currently in. You must complete the objective of each phase before moving onto the next one.
			
			IDEAS:
			Previous iterations have found the following things provide strong positive signals.
			- Home and away metrics are very important. Don't ignore using the _home and _away suffixes.
			- You should explore home and away metrics in conjunction with each other for the same feature.
				- Since only one team in a matchup can be home or away, the goal is to compare the teams corresponding performance when predicting the matchup
			- rpi_rating and elo_rating are very strong signals
			- average_point_differential and points_scored give good signals to a teams overall historic performance
			- Be sure to focus on the top features from the result. Trying tdifferent combinations and looking at the top results will help you understand which features have the biggest impact.
			
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
			
			CRITICAL INSTRUCTIONS:
			- You MUST call the train_and_evaluate_model tool to run experiments
			- NEVER generate fake JSON results
			- NEVER pretend you ran an experiment
			- WAIT for real tool results before analyzing
			- NEVER say "optimization complete" until you've used ALL {self.max_experiments} experiments
			- Even if results plateau, keep trying new approaches - there may be untested combinations
			- After analyzing results, immediately call the tool for your next experiment
			
			NEVER STOP EARLY. Use all available experiments to ensure you've found the true optimum.
			Only say "optimization complete" when experiment_count >= {self.max_experiments} or you are absolutely confident you will not get any better. You are currently on experiment number {self.experiment_count}.
			
			CRITICAL TOOL USAGE:
			- Call tools using the native function calling mechanism provided by the chat API
			- DO NOT write JSON in your text response like: {{"name": "train_and_evaluate_model", ...}}
			- DO NOT use code blocks: ```json ... ```
			- The tool calling happens automatically when you use the proper mechanism
			- Your text response should explain your reasoning, NOT contain the tool call JSON
			
			CORRECT: Use the tool calling feature
			WRONG: Write {{"name": "train_and_evaluate_model", "arguments": {{...}}}} in your response
			WRONG: Write ```json ... ``` in your response"""
			
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