import os

ROOT = os.getcwd()

class RunnerParams:
	def __init__(self, env, logfile, config, budget, max_play, snapshot, random_prior):
		# given params
		self.env = env
		self.budget = budget
		self.max_play = max_play
		self.config = config
		self.random_prior = random_prior
		self.snapshot = snapshot
		self.logfile = logfile

		# default params
		self.config_folder = os.path.join(ROOT, 'configs')
		self.env_folder = os.path.join(ROOT, 'environments', 'pomdp')

	@property
	def algo_config(self):
		return os.path.join(self.config_folder, self.config + '.json')

	@property
	def env_config(self):
		return os.path.join(self.env_folder, self.env)
