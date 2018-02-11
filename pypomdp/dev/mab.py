
from matplotlib import pylab as plt
import numpy as np

MAX = np.inf
DEBUG = True
# DEBUG = False

class Arm:
	def __init__(self, name, mean, stddev, cost):
		self.name = name
		self.mean = mean
		self.stddev = stddev
		self.cost = cost
		self.n = 0
		self.hist_reward = 0.0
		self.mean_reward = 0.0

	def pull(self):
		self.n += 1
		reward = np.random.normal(self.mean, self.stddev)
		self.hist_reward += reward
		self.mean_reward = self.hist_reward / self.n

		if DEBUG:
			print(('Arm {} gets pulled'.format(self.name)))
		return self.cost, reward


################
# General Util #
################
def generate_arm_configs(K, mean_bound, var_bound, cost_bound):
	means_ = np.random.uniform(mean_bound[0], mean_bound[1], K)
	vars_ = np.random.uniform(var_bound[0], var_bound[1], K)
	costs_ = np.random.uniform(cost_bound[0], cost_bound[1], K)
	return np.vstack((means_, vars_, costs_)).T


def plot(results, num_plays, regret=True):
	plt.plot(results.T)
	# plt.semilogy(results.T/num_plays)
	plt.title('Simulated Bandit Performance for K = 5')
	plt.ylabel('Cumulative Expected {}'.format('Regret' if regret else 'Reward'))
	plt.xlabel('Round Index')
	plt.legend(('UCB1', 'MAB-BV1','Random'),loc='lower right')
	plt.show()


def confidence(N_h, N_ha):
	if N_h == 0:
		return 0.
	if N_ha == 0:
		return MAX
	return np.sqrt(np.log(N_h) / N_ha)  # UCB1


##############
# Algorithms #
##############
def random_play(i, arms):
	return np.random.choice(arms).pull()


def ucb1(c):
	def runner(i, arms):
		arm_idx = np.argmax([arm.mean_reward + c * confidence(i, arm.n) for arm in arms])
		return arms[arm_idx].pull()
	return runner


def bv1(_lambda):
	def runner(i, arms):
		arm_idx = np.argmax([arm.mean_reward/arm.cost + ((1.+1./_lambda)*confidence(i, arm.n))/(_lambda-confidence(i, arm.n)) for arm in arms])
		return arms[arm_idx].pull()
	return runner


########
# Main #
########
if __name__ == '__main__':
	budget = 5000.
	K = 6
	c = 0.5
	num_plays = 13000
	mean_bound, var_bound, cost_bound = (0.1, 0.8), (0.02, 0.05), (0.3, 1.0)
	configs = generate_arm_configs(K, mean_bound, var_bound, cost_bound)
	_lambda = min(configs[:, 2])

	algorithms = [ucb1(c), bv1(_lambda), random_play]
	n_algos = len(algorithms)

	budgets = np.repeat(budget, n_algos)
	total_regret, total_reward = np.zeros((2, n_algos))
	acc_regrets, acc_rewards = np.zeros((2, n_algos, num_plays))

	best_reward = np.repeat(max(configs[:, 0]), n_algos)
	models = [[Arm(i, *config) for i, config in enumerate(configs)] for j in range(n_algos)]

	for i in range(num_plays):
		results = np.array([fn(i, models[j]) for j, fn in enumerate(algorithms)])
		costs, rewards = results[:, 0], results[:, 1]

		budgets -= costs
		alive = budgets > 0.0
		total_reward[alive] += rewards[alive]
		total_regret[alive] += best_reward[alive] - rewards[alive]
		acc_rewards[:, i] = total_reward
		acc_regrets[:, i] = total_regret

		if DEBUG:
			print(('-'*20))

	print('   mean_reward var         cost')
	print(configs)
	plot(acc_rewards, num_plays, regret=False)
