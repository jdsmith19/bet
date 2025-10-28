import requests
import json
import os
from tools.odds_tools import create_odds_tools
import config

def test_with_ollama():
	"""Test tools integrations with local Ollama"""
	
	# Load the tool
	odds_tool = create_odds_tools(config.odds_api_key)
	
	# Convert your tool definition to Ollama's format
	ollama_tool = {
		"type": "function",
		"function": {
			"name": odds_tool["definition"]["name"],
			"description": odds_tool["definition"]["description"],
			"paramaters": odds_tool["definition"]["input_schema"],
		}
	}
	
	response = requests.post(
		"http://localhost:11434/api/chat",
		json={
			"model": config.model,
			"messages": [{
				"role": "user",
				"content": "Who are the favorites and what is the spread for the upcoming NFL slate of games?"
			}],
			"tools": [ollama_tool],
			"stream": False
		}
	)
	
	result = response.json()
	message = result["message"]
	
	# Check if it wants to use your tool
	if "tool_calls" in message:
		print("✅ LLM is calling your tool!")
		
		for tool_call in message["tool_calls"]:
			# Execute YOUR function
			tool_result = odds_tool["function"](**tool_call["function"]["arguments"])
			
			print(f"\n📊 Got data: {str(tool_result)[:200]}...")
			
			# Send back to LLM for final answer
			final_response = requests.post(
				"http://localhost:11434/api/chat",
				json={
					"model": config.model,
					"messages": [
						{"role": "user", "content": "Who are the favorites and what is the spread for the upcoming NFL slate of games?"},
						message,
						{"role": "tool", "content": json.dumps(tool_result)}
					],
					"tools": [ollama_tool],
					"stream": False
				}
			)
			
			print("\n💬 Final Answer:")
			print(final_response.json())
			print(final_response.json()["message"]["content"])
	else:
		print("❌ LLM didn't use tool:")
		print(message.get("content"))
	
if __name__ == "__main__":
	test_with_ollama()