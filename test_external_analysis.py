from agents.ExternalAnalysisAgent import ExternalAnalysisAgent

eaa = ExternalAnalysisAgent(['Las Vegas Raiders @ Denver Broncos','Atlanta Falcons @ Indianapolis Colts','Baltimore Ravens @ Minnesota Vikings'])
eaa.run()
print(eaa.analysis)