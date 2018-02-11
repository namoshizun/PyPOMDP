from abc import abstractmethod


class Solver(object):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def solve(self, T):
        """
        To be implemented by algorithms
        :param T: planing horizon
        """

    @abstractmethod
    def update_belief(self, belief, action, obs):
        """
        To be implemented by algorithms
        :param belief: previous belief distribution
        :param action: action name
        :param obs: observation name
        :return: updated new belief distribution
        """

    @abstractmethod
    def get_action(self, belief):
        """
        To be implemented by algorithms
        :param belief: current belief distribution
        :return: the best action to take
        """

    @abstractmethod
    def add_configs(self, *args):
        """
        To be implemented by algorithms
        :param args: any arguments required by a specific POMDP solver algorithm
        :return:
        """

    def take_action(self, action):
        """
        Just a shallow Facade to expose model's take_action method to the external runner
        :param action: action name
        :return:
        """
        return self.model.take_action(action)


    # def draw(self, belief_points):
    #     '''
    #     Draw each of the alpha vectors and the final solution
    #     '''
    #     import matplotlib.pyplot as plt
    #     vals = np.zeros((len(self.alpha_vecs), len(belief_points)))
    #     for aidx in range(len(self.alpha_vecs)):
    #         alpha = self.alpha_vecs[aidx]
    #         vals[aidx, :] = np.dot(alpha, belief_points.transpose())
    #         plt.plot(belief_points[:, 0], vals[aidx, :], '--')
    #         plt.hold(True)

    #     max_vals = np.max(vals, axis=0)
    #     plt.plot(belief_points[:, 0], max_vals, 'k', linewidth=2)
    #     plt.show()
