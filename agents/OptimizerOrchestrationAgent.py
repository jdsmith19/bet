import config
import json
import ollama
from datetime import datetime
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
		print(f"🚀 Starting Optimizer Orchestration")
		print(f"Max Experiments: { self.max_experiments }")
		
		# Initialize conversation with system prompt
		messages = [
			{ 'role': 'system', 'content': self.__get_system_prompt() },
			{ 'role': 'user', 'content': self.__get_initial_prompt() }
		]
		
		# Agent Loop
		while self.experiment_count < self.max_experiments:	
			# Set the phase based on the experiment count
			if self.experiment_count >= 100 and self.phase == 1:
				print(f"Phase { self.phase } complete")
				print(f"BEST RESULTS")
				print(self.__format_all_best_results())
				self.phase = 2
			elif self.experiment_count >= 200 and self.experiment_count < 400 and self.phase == 2:
				print(f"Phase { self.phase } complete")
				print(f"BEST RESULTS")
				print(self.__format_all_best_results())
				self.phase = 3
			elif self.experiment_count >= 400 and self.phase == 3:
				print(f"Phase { self.phase } complete")
				print(f"BEST RESULTS")
				self.phase = 4
			
			#print(self.__get_current_results())
			results = self.__plan_next_experiments()
			if not results.get("status") == "complete":
				print("Results not in appropriate format.")
				continue
			else:
				for e in results.get('experiments'):
					print(f"{'='*80}")
					print(f"Experiment {self.experiment_count + 1} / {self.max_experiments}")
					print(f"{'='*80}")
					try:
						result = self.__train_and_evaluate_model(e['model'], e['features'])
						self.experiment_count += 1
						self.__update_best_results(result)
						self.__print_result_summary(result)

					except Exception as ex:
						print(f"Could not execute Experiment # { self.experiment_count + 1 }")
						print(f"Skipping Experiment")
						print(f"Error details: { ex }")
						print(f"Experiment details: { e }")

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
	
	def __plan_next_experiments(self):
		opa = OptimizerPlanningAgent(self.__get_current_results(), self.phase)
		response = opa.run()
		return json.loads(response)
	
	def __train_and_evaluate_model(self, model_name, features):
			result = train_and_evaluate_model(
				model_name = model_name,
				features = features
			)
			
			self.experiment_history.append({
				'experiment_num': self.experiment_count + 1,
				'arguments': { 'model_name': model_name, 'features': features },
				'result': result
			})
			
			return json.dumps(result)
	
	def __get_current_results(self):
		output = {
			'total_experiments': self.experiment_count,
			'best_results': self.best_results,
			'experiment_history': self.experiment_history[-50:]
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
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		with open(f'feature_optimization_results_{ timestamp }.json', 'w') as f:
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