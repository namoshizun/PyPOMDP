import os

from models import RockSampleModel, Model
from solvers import POMCP, PBVI
from parsers import PomdpParser, GraphViz
from logger import Logger as log

class PomdpRunner:

    def __init__(self, params):
        self.params = params
        if params.logfile is not None:
            log.new(params.logfile)

    def create_model(self, env_configs):
        """
        Builder method for creating model (i,e, agent's environment) instance
        :param env_configs: the complete encapsulation of environment's dynamics
        :return: concrete model
        """
        MODELS = {
            'RockSample': RockSampleModel,
        }
        return MODELS.get(env_configs['model_name'], Model)(env_configs)

    def create_solver(self, algo, model):
        """
        Builder method for creating solver instance
        :param algo: algorithm name
        :param model: model instance, e.g, TigerModel or RockSampleModel
        :return: concrete solver
        """
        SOLVERS = {
            'pbvi': PBVI,
            'pomcp': POMCP,
        }
        return SOLVERS.get(algo)(model)

    def snapshot_tree(self, visualiser, tree, filename):
        visualiser.update(tree.root)
        visualiser.render('./dev/snapshots/{}'.format(filename))  # TODO: parametrise the dev folder path

    def run(self, algo, T, **kwargs):
        visualiser = GraphViz(description='tmp')
        params, pomdp = self.params, None
        total_rewards, budget = 0, params.budget

        log.info('~~~ initialising ~~~')
        with PomdpParser(params.env_config) as ctx:
            # creates model and solver
            model = self.create_model(ctx.copy_env())
            pomdp = self.create_solver(algo, model)

            # supply additional algo params
            belief = ctx.random_beliefs() if params.random_prior else ctx.generate_beliefs()

            if algo == 'pbvi':
                belief_points = ctx.generate_belief_points(kwargs['stepsize'])
                pomdp.add_configs(belief_points)
            elif algo == 'pomcp':
                pomdp.add_configs(budget, belief, **kwargs)

        # have fun!
        log.info('''
        ++++++++++++++++++++++
            Starting State:  {}
            Starting Budget:  {}
            Init Belief: {}
            Time Horizon: {}
            Max Play: {}
        ++++++++++++++++++++++'''.format(model.curr_state, budget, belief, T, params.max_play))

        for i in range(params.max_play):
            # plan, take action and receive environment feedbacks
            pomdp.solve(T)
            action = pomdp.get_action(belief)
            new_state, obs, reward, cost = pomdp.take_action(action)

            if params.snapshot and isinstance(pomdp, POMCP):
                # takes snapshot of belief tree before it gets updated
                self.snapshot_tree(visualiser, pomdp.tree, '{}.gv'.format(i))
            
            # update states
            belief = pomdp.update_belief(belief, action, obs)
            total_rewards += reward
            budget -= cost

            # print ino
            log.info('\n'.join([
              'Taking action: {}'.format(action),
              'Observation: {}'.format(obs),
              'Reward: {}'.format(reward),
              'Budget: {}'.format(budget),
              'New state: {}'.format(new_state),
              'New Belief: {}'.format(belief),
              '=' * 20
            ]))

            if budget <= 0:
                log.info('Budget spent.')
                break


        log.info('{} games played. Toal reward = {}'.format(i + 1, total_rewards))
        return pomdp
