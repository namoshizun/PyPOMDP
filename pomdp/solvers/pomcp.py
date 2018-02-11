from solvers import Solver
from util.helper import rand_choice, randint, round
from util.helper import elem_distribution, ucb
from util.belief_tree import BeliefTree
from logger import Logger as log
import numpy as np
import time

MAX = np.inf

class UtilityFunction():
    @staticmethod
    def ucb1(c):
        def algorithm(action):
            return action.V + c * ucb(action.parent.N, action.N)
        return algorithm
    
    @staticmethod
    def mab_bv1(min_cost, c=1.0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            ucb_value = ucb(action.parent.N, action.N)
            return action.mean_reward / action.mean_cost + c * ((1. + 1. / min_cost) * ucb_value) / (min_cost - ucb_value)
        return algorithm

    @staticmethod
    def sa_ucb(c0):
        def algorithm(action):
            if action.mean_cost == 0.0:
                return MAX
            return action.V + c0 * action.parent.budget * ucb(action.parent.N, action.N)
        return algorithm


class POMCP(Solver):
    def __init__(self, model):
        Solver.__init__(self, model)
        self.tree = None

        self.simulation_time = None  # in seconds
        self.max_particles = None    # maximum number of particles can be supplied by hand for a belief node
        self.reinvigorated_particles_ratio = None  # ratio of max_particles to mutate 
        self.utility_fn = None

    def add_configs(self, budget=float('inf'), initial_belief=None, simulation_time=0.5,
                    max_particles=350, reinvigorated_particles_ratio=0.1, utility_fn='ucb1', C=0.5):
        # acquaire utility function to choose the most desirable action to try
        self.utility_fn = {
            'ucb1': UtilityFunction.ucb1(C),
            'mab_bv1': UtilityFunction.mab_bv1(min(self.model.costs), C),
            'sa_ucb': UtilityFunction.sa_ucb(C)
        }[utility_fn]

        # other configs
        self.simulation_time = simulation_time
        self.max_particles = max_particles
        self.reinvigorated_particles_ratio = reinvigorated_particles_ratio
        
        # initialise belief search tree
        root_particles = self.model.gen_particles(n=self.max_particles, prob=initial_belief)
        self.tree = BeliefTree(budget, root_particles)

    def compute_belief(self):
        base = [0.0] * self.model.num_states
        particle_dist = elem_distribution(self.tree.root.B)
        for state, prob in particle_dist.items():
            base[self.model.states.index(state)] = round(prob, 6)
        return base

    def rollout(self, state, h, depth, max_depth, budget):
        """
        Perform randomized recursive rollout search starting from 'h' util the max depth has been achived
        :param state: starting state's index
        :param h: history sequence
        :param depth: current planning horizon
        :param max_depth: max planning horizon
        :return:
        """
        if depth > max_depth or budget <= 0:
            return 0

        ai = rand_choice(self.model.get_legal_actions(state))
        sj, oj, r, cost = self.model.simulate_action(state, ai)

        return r + self.model.discount * self.rollout(sj, h + [ai, oj], depth + 1, max_depth, budget-cost)
        
    def simulate(self, state, max_depth, depth=0, h=[], parent=None, budget=None):
        """
        Perform MCTS simulation on a POMCP belief search tree
        :param state: starting state's index
        :return:
        """
        # Stop recursion once we are deep enough in our built tree
        if depth > max_depth:
            return 0

        obs_h = None if not h else h[-1]
        node_h = self.tree.find_or_create(h, name=obs_h or 'root', parent=parent,
                                          budget=budget, observation=obs_h)

        # ===== ROLLOUT =====
        # Initialize child nodes and return an approximate reward for this
        # history by rolling out until max depth
        if not node_h.children:
            # always reach this line when node_h was just now created
            for ai in self.model.get_legal_actions(state):
                cost = self.model.cost_function(ai)
                # only adds affordable actions
                if budget - cost >= 0:
                    self.tree.add(h + [ai], name=ai, parent=node_h, action=ai, cost=cost)

            return self.rollout(state, h, depth, max_depth, budget)

        # ===== SELECTION =====
        # Find the action that maximises the utility value
        np.random.shuffle(node_h.children)
        node_ha = sorted(node_h.children, key=self.utility_fn, reverse=True)[0]

        # ===== SIMULATION =====
        # Perform monte-carlo simulation of the state under the action
        sj, oj, reward, cost = self.model.simulate_action(state, node_ha.action)
        R = reward + self.model.discount * self.simulate(sj, max_depth, depth + 1, h=h + [node_ha.action, oj],
                                                    parent=node_ha, budget=budget-cost)
        # ===== BACK-PROPAGATION =====
        # Update the belief node for h
        node_h.B += [state]
        node_h.N += 1

        # Update the action node for this action
        node_ha.update_stats(cost, reward)
        node_ha.N += 1
        node_ha.V += (R - node_ha.V) / node_ha.N

        return R

    def solve(self, T):
        """
        Solves for up to T steps
        """
        begin = time.time()
        n = 0
        while time.time() - begin < self.simulation_time:
            n += 1
            state = self.tree.root.sample_state()
            self.simulate(state, max_depth=T, h=self.tree.root.h, budget=self.tree.root.budget)
        log.info('# Simulation = {}'.format(n))

    def get_action(self, belief):
        """
        Choose the action maximises V
        'belief' is just a part of the function signature but not actually required here
        """
        root = self.tree.root
        action_vals = [(action.V, action.action) for action in root.children]
        return max(action_vals)[1]

    def update_belief(self, belief, action, obs):
        """
        Updates the belief tree given the environment feedback.
        extending the history, updating particle sets, etc
        """
        m, root = self.model, self.tree.root

        #####################
        # Find the new root #
        #####################
        new_root = root.get_child(action).get_child(obs)
        if new_root is None:
            log.warning("Warning: {} is not in the search tree".format(root.h + [action, obs]))
            # The step result randomly produced a different observation
            action_node = root.get_child(action)
            if action_node.children:
                # grab any of the beliefs extending from the belief node's action node (i.e, the nearest belief node)
                log.info('grabing a bearest belief node...')
                new_root = rand_choice(action_node.children)
            else:
                # or create the new belief node and rollout from there
                log.info('creating a new belief node')
                particles = self.model.gen_particles(n=self.max_particles)
                new_root = self.tree.add(h=action_node.h + [obs], name=obs, parent=action_node, observation=obs,
                                         particle=particles, budget=root.budget - action_node.cost)
        
        ##################
        # Fill Particles #
        ##################
        particle_slots = self.max_particles - len(new_root.B)
        if particle_slots > 0:
            # fill particles by Monte-Carlo using reject sampling
            particles = []
            while len(particles) < particle_slots:
                si = root.sample_state()
                sj, oj, r, cost = self.model.simulate_action(si, action)

                if oj == obs:
                    particles.append(sj)
            new_root.B += particles

        #####################
        # Advance and Prune #
        #####################
        self.tree.prune(root, exclude=new_root)
        self.tree.root = new_root
        new_belief = self.compute_belief()

        ###########################
        # Particle Reinvigoration #
        ###########################
        if any([prob == 0.0 for prob in new_belief]):
            # perform particle re-invigoration when particle deprivation happens
            mutations = self.model.gen_particles(n=int(self.max_particles * self.reinvigorated_particles_ratio))
            for particle in mutations:
                new_root.B[randint(0, len(new_root.B))] = particle

            # re-compute the current belief distribution after reinvigoration
            new_belief =  self.compute_belief()
            log.info(('*** {} random particles are added ***'.format(len(mutations))))
        return new_belief

    def draw(self, beliefs):
        """
        Dummy
        """
        pass
