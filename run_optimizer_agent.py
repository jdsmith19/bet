import config
from agents.OptimizerOrchestrationAgent import OptimizerOrchestrationAgent

agent = OptimizerOrchestrationAgent(
	model_name = config.planning_model,
	max_experiments = 500
)
agent.run()