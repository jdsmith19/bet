import config
from FeatureOptimizerAgent import FeatureOptimizerAgent

agent = FeatureOptimizerAgent(
	model_name = config.model,
	max_experiments = 500
)
agent.run()