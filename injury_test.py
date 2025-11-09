from data_sources.ESPN import NFLDepthChartAnalyzer

analyzer = NFLDepthChartAnalyzer()

# Get injury report for a single team
prompt = analyzer.get_llm_prompt_context('buf')

# Or get JSON
json_data = analyzer.to_json_for_llm('buf')

# Send to your local LLM and get adjustment multipliers back
print(prompt)
print(json_data)
