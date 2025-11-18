import json
import time
from helpers.Lookup import Lookup
from tools.upcoming_predictions_tools import get_upcoming_predictions
from tools.injury_report_tools import get_injury_report_for_teams
from tools.html_generation_tools import generate_html_report
from agents.InjuryAdjustmentAgent import InjuryAdjustmentAgent
from agents.GameAnalysisAgent import GameAnalysisAgent
from agents.ExternalAnalysisAgent import ExternalAnalysisAgent
from agents.HTMLGenerationAgent import HTMLGenerationAgent
from agents.PodcastSummarizationAgent import PodcastSummarizationAgent
from data_sources.BillSimmonsPodcast import BillSimmonsPodcast
import tiktoken
from tqdm import tqdm
from dotenv import load_dotenv
import os
import requests

load_dotenv()

class PredictionOrchestrationAgent:
	def __init__(self):
		self.start_time = time.time()
		self.full_aggregates = None
		self.adjusted_aggregates = None
		self.injury_report = None
		self.matchup_details = {}
		self.analysis = {}
		self.week = 0
		self.season = 0
		self.debug = False
	
	def run(self):
		"""Main agent loop"""
		print(f"\n{'='*80}")
		print(f"🎹 Starting Prediction Orchestration Agent")
		print(f"{'='*80}")

		# GET INITIAL PREDICTIONS
		predictions = self.__get_upcoming_predictions()
		print(predictions)
		self.week = predictions['prediction_set']['season_week_number'].unique()[0]
		self.season = predictions['prediction_set']['season'].unique()[0]
		
		# GET ALL INJURY REPORTS
		unique_teams = self.__get_unique_teams_from_predictions(predictions)
		injury_report = self.__get_injury_reports(unique_teams)
		
		# GET INJURY ADJUSTMENTS
		injury_adjustments = self.__get_injury_adjustments(injury_report)
		 
		# GET INJURY ADJUSTED PREDICTIONS
		adjusted_predictions = self.__get_upcoming_predictions(adjusted=True)
		 
		# GET PODCAST SUMMARIES
		gtl_summary = self.__get_podcast_analysis('guess_the_lines')
		
		# GET EXTERNAL EXPERT ANALYSIS
		expert_analysis = self.__get_expert_analysis()
		
		# GET GAME ANALYSIS
		game_analysis = self.__get_game_analysis()
		
		# GENERATE REPORT
		final_report = self.__make_final_report(game_analysis)
		
		print(f"Total elapsed time { round(time.time() - self.start_time, 3) }s\n")
	
	def __get_upcoming_predictions(self, adjusted = False):
		if adjusted == True:
			prediction_type = 'injury_adjusted_predictions'			
			result = get_upcoming_predictions(self.adjusted_aggregates)

		else:
			prediction_type = 'base_predictions'
			result = get_upcoming_predictions()
		
		for item in result['predictions']:
			for prediction in item['results']:
				matchup_name = f"{prediction['away_team']} @ { prediction['home_team']}"
				if matchup_name not in self.matchup_details:
					self.matchup_details[matchup_name] = {}
				if prediction_type not in self.matchup_details[matchup_name]:
					self.matchup_details[matchup_name][prediction_type] = []
				self.matchup_details[matchup_name][prediction_type].append(self.__organize_prediction_details(item, prediction))
		
		if not adjusted:
			self.full_aggregates = result['full_aggregates']
		
		return result
	
	def __get_unique_teams_from_predictions(self, predictions):
		unique_teams = []
		for prediction_set in predictions['predictions']:
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
		pbar = tqdm(injury_report, "Getting injury adjustments")
		pbar.write("\n👨🏻‍🔬 Getting injury adjustments")
		lu = Lookup()
		for ir in pbar:
			if self.adjusted_aggregates is None:
				iaa = InjuryAdjustmentAgent(json.dumps(ir), self.full_aggregates)
			else:
				iaa = InjuryAdjustmentAgent(json.dumps(ir), self.adjusted_aggregates)
			pbar.set_description(f"{ lu.injury_report_to_team_name(ir['team']) }")
			adjustments = iaa.run()

			self.adjusted_aggregates = adjustments
		return self.adjusted_aggregates

	def __get_expert_analysis(self):
		games = []
		for matchup in self.matchup_details:
			games.append(matchup)
		
		with tqdm(total=100, desc="Getting expert analysis") as pbar:
			pbar.write("\n🌍 Getting expert analysis")
			pbar.set_description("Summarizing expert analysis")
			eaa = ExternalAnalysisAgent(games)
			eaa.run()
			pbar.update(90)
			pbar.set_description("Adding expert analysis to dataset")
			for a in eaa.analysis:
				matchup = a['matchup']
				if matchup in self.matchup_details:
					if 'expert_analysis' not in self.matchup_details[matchup]:
						self.matchup_details[matchup]['expert_analysis'] = []
					self.matchup_details[matchup]['expert_analysis'].append(a['analysis'])
			pbar.update(10)
			return eaa.analysis

	def __get_podcast_analysis(self, podcast):
		games = []
		for matchup in self.matchup_details:
			games.append(matchup)
		with tqdm(total=100, desc="Summarizing Podcasts") as pbar:
			pbar.write("\n🎙️  Summarizing Podcasts")
			pbar.set_description("Downloading")
			bsp = BillSimmonsPodcast(week = self.week, season = self.season, podcast="guess_the_lines")
			job_id = bsp.transcribe_episode(episode_type=podcast)
			episode_name = bsp.current_episode_name
			duration = int(bsp.current_episode_duration)
			# Update progress while waiting
			job_completed = False
			while not job_completed:
				data = bsp.check_job_status(job_id)
				status = bsp.job_status
				if status == "started":
					pbar.set_description(f"Starting { episode_name }")
				elif status == "downloading":
					pbar.set_description(f"Downloading { episode_name }")
				elif status == "transcribing":
					pbar.set_description(f"Transcribing { episode_name }")
					pbar.n = 10
				elif status == "completed":
					pbar.n = 75
					job_completed = True
				elif status == "failed":
					pbar.set_description("❌ FAILED")
					print(data['error'])
					print(data['traceback'])
				time.sleep(1)
			transcription = bsp.check_job_status(job_id)
			bsp.job_id = None
			bsp.current_job_id = None
			bsp.current_episode_name = "Guess the Lines with Cousin Sal"

			chunked_transcription = bsp.chunk_transcription(transcription['result'], chunk_size = 2500, overlap = 250)
			chunks = []
			summaries = []
			
			for chunk in chunked_transcription:
				if self.debug:
					encoding = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, close enough for most models
					tokens = len(encoding.encode(chunk))
					print(f"Estimating { tokens } tokens. Sending to LLM.\n")
				chunks.append(chunk)
				pbar.set_description(f"Summarizing { episode_name }")
				psa = PodcastSummarizationAgent(games, chunk)
				summary = psa.run()
				pbar.n = 100
				if not json.loads(summary):
					continue
				for s in json.loads(summary):
					summaries.append(summary)
					self.matchup_details.setdefault(s, {}).setdefault('podcast_analysis', []).append(s)
		pbar.set_description("DONE")
		return summaries
		

	def __get_game_analysis(self):
		analysis = []
		pbar = tqdm(self.matchup_details, "Generating final game analysis")
		pbar.write("\n🧐 Generating final game analysis")
		for matchup in pbar:
			pbar.set_description(f"{ matchup }")
			
			matchup_details = {}
			matchup_details[matchup] = self.matchup_details[matchup]
			if self.debug:
				encoding = tiktoken.get_encoding("cl100k_base")  # Used by GPT-4, close enough for most models
				tokens = len(encoding.encode(json.dumps(matchup_details)))
				print(f"Estimating { tokens } tokens. Sending to LLM.\n")
			gaa = GameAnalysisAgent(matchup_details)
			try:
				analysis.append(gaa.run())
			except Exception as e:
				print(f"Error in get_game_analysis: {e}")
				import traceback
				traceback.print_exc()
				return {"error": str(e)}
		self.analysis = analysis
		return analysis
	
	def __make_final_report(self, game_analysis):
		with tqdm(total=100, desc="Generating HTML report") as pbar:
			pbar.write("\n🌍 Generating HTML report")
			hga = HTMLGenerationAgent(game_analysis)
			html = hga.run()
			if isinstance(html, (list)):
				generate_html_report(''.join(html))
			# If it's a string, parse it
			elif isinstance(html, str):
				generate_html_report(html)
			pbar.update(100)

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