import config
import json
import time
from tools.html_generation_tools import generate_html_report
from dotenv import load_dotenv
import os
from openai import OpenAI

class HTMLGenerationAgent:
	def __init__(self, analysis):
		"""Main agent loop"""
		start_time = time.time()
		self.analysis = analysis
		self.debug = False

	def run(self):
		start_time = time.time()
		finished = False

		if self.debug:
			print(f"👨🏻‍💻 Starting HTML Generation Agent")
		
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
				messages = messages
			)
			msg = response.choices[0].message
			messages.append(msg.model_dump(exclude_none=True))
		
			if not msg.content and not msg.tool_calls:
				empty_responses += 1
					
			print(f"\n{'='*80}")
			print(f"👨🏻‍💻 HTML Generation Agent Response")
			print(f"{'='*80}\n")
							
			if msg.content:
				print(f"💬 EXPLANATION:")
				print(f"{msg.content}\n")		
				print(f"👨🏻‍💻 Exiting HTML Generation Agent")
				print(f"Completed in { round(time.time() - start_time, 3) }s")
				finished = True
				return msg.content
						
			print(f"{ '='*80 }\n")
	
	def __get_system_prompt(self):
		"""System prompt with full context"""
		
		return f"""You are an expert HTML designer and web-page builder that will create stunning web pages from a JSON object of NFL analysis.
		
		STEPS
		- You will receive for a list of HTML games
		- Format all game analyses in a single HTML string, do not send a list
		- Return the HTML string and ONLY the HTML string in your response

		HTML LAYOUT
		• Use the following HTML doctype details:
			<!DOCTYPE html>
			<html lang="en">
			  ...
			</html>
		• Design each HTML page as a table of results using Bootstrap for CSS & Javascript. Use the following to include the CSS & Javascript Libraries in the page:
			<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
			<!-- Latest compiled and minified JavaScript -->
			<script src="https://cdn.jsdelivr.net/npm/bootstrap@3.3.7/dist/js/bootstrap.min.js" integrity="sha384-Tc5IQib027qvyjSMfHjOMaLkfuWVxZxUPnCJA7l2mCWNIpG9mGCD8wGNIcPD7Txa" crossorigin="anonymous"></script>
		• For <table>, use the classes "table table-bordered table-striped"
		• For <tbody>, use the classes "table table-hover"
		• If the confidecne for a result is VERY_HIGH, use the classes "table-success" for the <tr>
		• If the confidence for a result is VERY_LOW, use the class "table-danger" for the <tr>
		• If the confidence for a result is LOW, use the class "table-warning" for the <tr>
		• If the confidence for a result is HIGH, use the class "table-info" for the <tr>
		• Wrap the entire page in a div with the class "container-fluid"
			<div class="container-fluid">
			...
			</div>
		
		Each result should include the following information:
			- Column: Matchup, Cell: [Away Team] @ [Home Team]
			- Column: Base Model Prediction, Cell: [winner] by [spread] pts
			- Column: Injury Adjusted Prediction, Cell: [winner] by [spread] pts
			- Column: Final Prediction, Cell: [winner] by [spread] pts
			- Column: Confidence, Cell: [confidence]
			- Column: Analysis, Cell: [A bulleted list of the analysis points]"""
			
	def __get_initial_prompt(self):
		"""Initial user message to start the agent"""
		return f"""Generate HTML from the following game_analysis list
		
		{ self.analysis }"""