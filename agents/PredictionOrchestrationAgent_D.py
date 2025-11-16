import config
import json
#import ollama
import time
from helpers.Lookup import Lookup
from tools.upcoming_predictions_tools import get_upcoming_predictions
from tools.injury_report_tools import get_injury_report_for_teams
from tools.html_generation_tools import generate_html_report
from agents.InjuryAdjustmentAgent import InjuryAdjustmentAgent
from agents.GameAnalysisAgent import GameAnalysisAgent
from agents.ExternalAnalysisAgent import ExternalAnalysisAgent
from agents.HTMLGenerationAgent import HTMLGenerationAgent

class PredictionOrchestrationAgent:
	def __init__(self):
		self.start_time = time.time()
		self.adjusted_aggregates = None
		self.injury_report = None
		self.matchup_details = {}
		self.analysis = {}
	
	def run(self):
		"""Main agent loop"""
		print(f"🎹 Starting Prediction Orchestration Agent")
		# GET INITIAL PREDICTIONS
		print("Getting upcoming predictions...")
		predictions = self.__get_upcoming_predictions()
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		
		# # GET ALL INJURY REPORTS
		print("Generating injury reports...")
		unique_teams = self.__get_unique_teams_from_predictions(predictions)
		injury_report = self.__get_injury_reports(unique_teams)
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		# 
		# # GET INJURY ADJUSTMENTS
		print("Adjusting team statistics to account for injuries...")
		injury_adjustments = self.__get_injury_adjustments(injury_report)
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		 
		# # GET INJURY ADJUSTED PREDICTIONS
		print("Getting upcoming predictions with adjusted statistics...")
		adjusted_predictions = self.__get_upcoming_predictions()
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		 
		# GET EXTERNAL EXPERT ANALYSIS
		print("Getting expert analysis...")
		expert_analysis = self.__get_expert_analysis()
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		
		# GET GAME ANALYSIS
		print("Generating final analysis for each game...")
		game_analysis = self.__get_game_analysis()
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
		
		# GENERATE REPORT
		print("Generating the final report...")
		final_report = self.__make_final_report(game_analysis)
		
		print(f"Elapsed time { round(time.time() - self.start_time, 3) }s\n")
	
	def __get_upcoming_predictions(self):
		if self.adjusted_aggregates:
			prediction_type = 'injury_adjusted_predictions'
		else:
			prediction_type = 'base_predictions'

		result = get_upcoming_predictions()
		for ml_model in result:
			for prediction in ml_model['results']:
				matchup_name = f"{prediction['away_team']} @ { prediction['home_team']}"
				if matchup_name not in self.matchup_details:
					self.matchup_details[matchup_name] = {}
				if prediction_type not in self.matchup_details[matchup_name]:
					self.matchup_details[matchup_name][prediction_type] = []
				self.matchup_details[matchup_name][prediction_type].append(self.__organize_prediction_details(ml_model, prediction))
		return result
	
	def __get_unique_teams_from_predictions(self, predictions):
		unique_teams = []
		for prediction_set in predictions:
			for result in prediction_set['results']:
				if result['home_team'] not in unique_teams:
					unique_teams.append(result['home_team'])
				if result['away_team'] not in unique_teams:
					unique_teams.append(result['away_team'])
		return unique_teams
	
	def __get_injury_reports(self, teams):
		result = get_injury_report_for_teams(
			teams = teams
		)
		self.injury_report = result
		lu = Lookup()
		for ir in self.injury_report:
			for matchup in self.matchup_details:
				team_name = lu.injury_report_to_team_name(ir['team'])
				if team_name in matchup:
					if 'detailed_injury_report' not in self.matchup_details[matchup]:
						self.matchup_details[matchup]['detailed_injury_report'] = []
					self.matchup_details[matchup]['detailed_injury_report'].append(ir)
		return result

	def __get_injury_adjustments(self, injury_report):
		for ir in injury_report:
			iaa = InjuryAdjustmentAgent(json.dumps(ir), self.adjusted_aggregates)
			print(f"Making injury adjustments for { ir['team']}")
			adjustments = iaa.run()
			self.adjusted_aggregates = adjustments
		self.__get_upcoming_predictions()
		return self.adjusted_aggregates

	def __get_expert_analysis(self):
		games = []
		for matchup in self.matchup_details:
			games.append(matchup)
		eaa = ExternalAnalysisAgent(games)
		eaa.run()
		for a in eaa.analysis:
			matchup = a['matchup']
			if matchup in self.matchup_details:
				if 'expert_analysis' not in self.matchup_details[matchup]:
					self.matchup_details[matchup]['expert_analysis'] = []
				self.matchup_details[matchup]['expert_analysis'].append(a['analysis'])
		return eaa.analysis

	def __get_game_analysis(self):
		# TEMPORARILY LIMITING TO 15 GAMES UNTIL I FIX THE WAY UPCOMING GAMES ARE PULLED TO USE CURRENT / UPCOMING WEEK
		i = 0
		analysis = []
		for matchup in self.matchup_details:
			print(f"Analyzing { matchup }")
			i += 1
			if i <= 15:
				matchup_details = {}
				matchup_details[matchup] = self.matchup_details[matchup]
				gaa = GameAnalysisAgent(matchup_details)
				try:
					analysis.append(gaa.run())
				except Exception as e:
					print(f"Error in get_game_analysis: {e}")
					import traceback
					traceback.print_exc()
					return {"error": str(e)}
			else:
				break
		self.analysis = analysis
		return analysis
	
	def __make_final_report(self, game_analysis):
		hga = HTMLGenerationAgent(game_analysis)
		html = hga.run()
		if isinstance(html, (list)):
			generate_html_report(''.join(html))
		# If it's a string, parse it
		elif isinstance(html, str):
			generate_html_report(html)

	def __organize_prediction_details(self, ml_model, prediction):
		matchup_details = {
			'model_details': {
				'model_name': ml_model['model_name'],
				'mean_absolute_error': ml_model.get('mean_absolute_error'),
				'root_mean_squared_error': ml_model.get('root_mean_squared_error'),
				'feature_importance': ml_model.get('feature_importance'),
				'feature_coefficients': ml_model.get('feature_coefficients'),
				'train_accuracy': ml_model.get('train_accuracy'),
				'test_accuracy': ml_model.get('test_accuracy'),
				'confidence_intervals': ml_model.get('confidence_intervals')
			},
			'prediction': {
				'predicted_spread': prediction.get('predicted_spread'),
				'predicted_winner': prediction['predicted_winner'],
				'prediction_text': prediction.get('prediction_text'),
				'confidence': prediction.get('confidence')
			}
		}
		return matchup_details