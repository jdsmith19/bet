import config
import json
import ollama
from tools.feature_refinement_tools import train_and_evaluate_model
from agents.OptimizerPlanningAgent import OptimizerPlanningAgent

class OptimizerOrchestrationAgent:
	def __init__(self, model_name=config.orchestration_model, max_experiments=300):
		self.model = model_name
		self.max_experiments = max_experiments
		self.experiment_count = 0
		self.experiment_history = []
		with open('feature_optimization_results.json', 'r') as f:
			self.best_results = json.load(f)['best_results']
		self.empty_response_count = 0
		self.max_consecutive_empty_responses = 1
		self.phase = 1
	
	def run(self):
		"""Main agent loop"""
		print(f"🚀 Starting Optimizer Optimization Agent")
		print(f"Model: { self.model }")
		print(f"Max Experiments: { self.max_experiments }")
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		# Agent Loop
		while self.experiment_count < self.max_experiments:	
			# Set the phase based on the experiment count
			if self.experiment_count > 100 and self.experiment_count <= 200:
				self.phase = 2
			elif self.experiment_count > 200 and self.experiment_count <= 400:
				self.phase = 3
			elif self.experiment_count > 400:
				self.phase = 4
			
			# Estimate token count (rough)
			total_chars = sum(len(str(msg)) for msg in messages)
			estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 chars
	
			print(f"\n{'='*80}")
			print(f"📊 CONTEXT INSPECTION - Experiment { self.experiment_count } / { self.max_experiments }")
			print(f"Total messages: { len(messages) }")
			print(f"Estimated tokens: { estimated_tokens }")
			print(f"{'='*80}")
									
			# Get agent's response
			response = ollama.chat(
				model = self.model,
				messages = messages,
				tools = self.__get_tool_definition()
			)
			msg = response['message']
			
			print(f"{'='*80}")
			print(f"🤖 Agent Response")
			print(f"{'='*80}")
			
			# Show the thinking (chain-of-thought)
			if msg.get('thinking'):
				print(f"🧠 REASONING:")
				print(f"{msg.thinking}")
				
			if msg.get('content'):
				print(f"💬 EXPLANATION:")
				print(f"{msg.content}")
						
			print(f"{'='*80}\n")
			
			missing_response = (not msg.get('thinking') and not msg.get('content') and not msg.get('tool_calls'))				
			
			# Add assistant's message to history
			if missing_response:
				intervention_message = f"You appear to be stuck. You need to execute the next experiments. If you don't have any experiments to run, call the plan_next_experiments tool to get more experiments."
				print(f"👦🏻 USER: { intervention_message }")
				messages.append({
					'role': 'user',
					'content': intervention_message
				})
			
			else:
				messages.append(msg)
			
			# Check if agent wants to use tool
			if msg.get('tool_calls'):
				# Process tool calls
				for tool_call in msg['tool_calls']:
					
					print(f"Agent is calling tool { tool_call['function']['name'] }\n")
					result = self.__execute_tool(tool_call)
	
					if tool_call['function']['name'] == 'train_and_evaluate_model':
						print(f"{'='*80}")
						print(f"Experiment {self.experiment_count + 1} / {self.max_experiments}")
						print(f"{'='*80}")
						self.__update_best_results(result)					
						self.__print_result_summary(result)
						self.experiment_count += 1
						filtered_messages = []
						for message in messages:
							if '"status": "complete"' in message.get('content'):
								filtered_messages.append(message)
						messages = filtered_messages					
								
					# Add tool result to messages
					messages.append({
						'role': 'tool',
						'content': result
					})
		
		self.__save_results()
		self.__print_final_summary()
		
	def __get_system_prompt(self):
		"""System prompt with full context"""
		return f"""You are an Orchestration Agent tasked with executing experiments and keeping track of the results of those experiments to find the best possible set of feature combinations for a set of Machine Learning models through systematic experimentation.
		
		YOUR GOAL:
		Complete experiments to identify the best posssible outcome. Utilize the available tools to plan experiments and execute the experiments made available to you through those tools.
								
		YOUR STRATEGY:
		- Call the plan_next_experiments tool to get a set of experiments to execute.
		- Do not provide any analysis on the results of the models. That is the job of the plan_next_experiments tool.
		- After you have executed all the experiments from plan_next_experiments, call plan_next_experiments again.
							
		EXPERIMENT INTERPRETATION:
		Regression models (XGBoost, LinearRegression, RandomForest):
		- Primary metric: MAE (Mean Absolute Error) - LOWER IS BETTER
		- Secondary metric: RMSE (Root Mean Squared Error) - LOWER IS BETTER
		- Use feature_importance to see which features the model values
		
		Classification models (LogisticRegression, KNearest):
		- Primary metric: test_accuracy - HIGHER IS BETTER
		- Also look at confidence_intervals for calibration quality
		
		IDENTIFYING EXPERIMENTS:
		- You will receive JSON from the plan_next_experiments tool in the following format:
		
		{{
			'status': 'complete',
			'experiments': [
				{{
					'model': [Which model to execute the experiment against. MUST be one of the AVAILABLE MODELS],
					'features': [Which features to use in the model training. MUST be one of the AVAILABLE FEATURES],
					
				}}
			]
		}}
		
		- For each object in the 'experiments' array, use the data to call the train_and_evaluate_model tool.
		- HINT: you can make multiple tool calls in one turn.
					
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
		return f"""Begin feature optimization by calling the plan_next_experiments tool to get your initial experiments."""
			
	def __get_tool_definition(self):
		"""Tool definition for Ollama"""
		return [{
			'type': 'function',
			'function': {
				'name': 'plan_next_experiments',
				'description': 'Plans the next feature optimization experiment based on historical results',
				'parameters': {},
					'required': None
				}
			},
			{
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
		]
	
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
				
				return json.dumps(result)
				
			except Exception as e:
				import traceback
				traceback.print_exc()
				return {
					'error': str(e),
					'model_name': arguments.get('model_name'),
					'features': arguments.get('features')
				}
		
		elif function_name == 'plan_next_experiments':
			try:
				opa = OptimizerPlanningAgent(self.__get_current_results(), self.phase)
				response = opa.run()
				return response
			except Exception as e:
				import traceback
				traceback.print_exc()
				return {
					'error': str(e),
				}
				
		return {'error': 'Unknown function'}
	
	def __get_current_results(self):
		output = {
			'total_experiments': self.experiment_count,
			'best_results': self.best_results,
			'experiment_history': self.experiment_history
		}
		return output
		
	def __update_best_results(self, result):
		"""Track best results per model"""
		result = json.loads(result)
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
			old_metric = self.__get_primary_metric(old_result)
			if old_metric is None:
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
		result = json.loads(result)
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