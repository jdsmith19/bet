import config
from FeatureOptimizerAgent import FeatureOptimizerAgent

agent = FeatureOptimizerAgent(
	model_name = config.model,
	max_experiments = 25
)
agent.run()