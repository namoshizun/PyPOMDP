"""
    This .POMDP parser is a modified version of this:
    https://github.com/mbforbes/py-pomdp/blob/master/pomdp.py
"""

from copy import deepcopy
from numpy import *
from util.helper import gen_distribution
from numpy import random
import os
import itertools


class PomdpxParser:
    # TODO
    pass


class PomdpParser:
    def __init__(self, config_file):
        '''
        Parses .pomdp file and loads info into this object's fields.
        '''
        self.config_file = config_file
        self.model_name = None
        self.model_spec = None

        self.T, self.Z, self.R = {}, {}, {}
        self.discount, self.start, self.init_state = None, None, None
        self.states, self.actions, self.observations, self.costs = None, None, None, None

    def __enter__(self):
        attrs = ['init_state', 'start', 'discount', 'values', 'states', 'actions', 'costs', 'observations', 'T', 'O', 'R']

        with open(self.config_file, 'r') as f:
            self.contents = [
                x.strip() for x in f.readlines()
                if not x.startswith("#") and not x.isspace()
            ]
            self.__get_model()

            # go through line by line and extract configurations
            i = 0
            while i < len(self.contents):
                line = self.contents[i]
                attr = [a for a in attrs if line.startswith(a)]

                if not attr:
                    raise Exception("Unrecognized line: " + line)
                i = getattr(self, '_PomdpParser__get_' + attr[0])(i)

        return self

    def __exit__(self, ctx_type, ctx_value, ctx_traceback):
        self = None

    def __get_model(self):
        fname = os.path.basename(self.config_file)
        if '-' in fname:
            name, spec = fname.split('-')
            self.model_spec = spec.split('.')[0]
            self.model_name = name.split('.')[0]
        else:
            self.model_name = fname.split('.')[0]

    def __get_discount(self, i):
        line = self.contents[i]
        self.discount = float(line.split()[1])
        return i + 1

    def __get_values(self, i):
        # Currently just supports "values: reward". I.e. currently
        # meaningless.
        line = self.contents[i]
        self.values = line.split()[1]
        return i + 1

    def __parse_line__(self, i, attr):
        parts = self.contents[i].split()

        if len(parts) == 2:
            n = int(parts[1])
            setattr(self, attr, list(map(str, list(range(n)))))
        else:
            setattr(self, attr, parts[1:])
        return i + 1

    def __get_init_state(self, i):
        line = self.contents[i]
        self.init_state = line.split()[1]
        return i + 1

    def __get_states(self, i):
        return self.__parse_line__(i, 'states')

    def __get_actions(self, i):
        return self.__parse_line__(i, 'actions')

    def __get_observations(self, i):
        return self.__parse_line__(i, 'observations')
    
    def __get_start(self, i):
        j = self.__parse_line__(i, 'start')
        self.start = list(map(float, self.start))
        return j

    def __get_costs(self, i):
        j = self.__parse_line__(i, 'costs')
        self.costs = list(map(float, self.costs))
        return j

    def __get_T(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        # try:
        #     pieces = map(float, pieces)
        # except Exception:
        #     pass
        action = pieces[0]

        if len(pieces) == 4:
            # case 1: T: <action> : <start-state> : <next-state> %f
            start_state, next_state = pieces[1], pieces[2]
            self.T[(action, start_state, next_state)] = float(pieces[3])
            return i + 1
        elif len(pieces) == 3:
            # case 2: T: <action> : <start-state> : <next-state>
            # %f
            start_state, next_state = pieces[1], pieces[2]
            self.T[(action, start_state, next_state)] = float(self.contents[i+1])
            return i + 2
        elif len(pieces) == 2:
            # case 3: T: <action> : <start-state>
            # %f %f ... %f
            start_state = pieces[1]
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.states)
            for j, prob in enumerate(probs):
                self.T[(action, start_state, j)] = float(prob)
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: T: <action>
                # identity
                # map(lambda tprob: self.T[tprob] = 1.0 if tprob[1] == tprob[2] else 0.0, [])
                for comb in itertools.product([action], self.states, self.states):
                    self.T[(action, comb[1], comb[2])] = 1.0 if comb[1] == comb[2] else 0.0
                return i + 2
            elif next_line == "uniform":
                # case 5: T: <action>
                # uniform
                prob = 1.0 / float(len(self.states))
                for comb in itertools.product([action], self.states, self.states):
                    self.T[comb] = prob
                return i + 2
            else:
                # case 6: T: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j, sj in enumerate(self.states):
                    probs = next_line.split()
                    assert len(probs) == len(self.states)
                    for k, sk in enumerate(self.states):
                        prob = float(probs[k])
                        self.T[(action, sj, sk)] = prob
                    next_line = self.contents[i+2+j]
                return i+1+len(self.states)
        else:
            raise Exception("Cannot parse line " + line)

    def __get_O(self, i):
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        # try:
        #     pieces = map(float, pieces)
        # except Exception:
        #     pass
        action = pieces[0]

        if len(pieces) == 4:
            # case 1: O: <action> : <next-state> : <obs> %f
            next_state, obs, prob = pieces[1], pieces[2], float(pieces[3])
            self.Z[(action, next_state, obs)] = prob
            return i + 1
        elif len(pieces) == 3:
            # case 2: O: <action> : <next-state> : <obs>
            # %f
            next_state, obs = pieces[1], pieces[2]
            next_line = self.contents[i+1]
            prob = float(next_line)
            self.Z[(action, next_state, obs)] = prob
            return i + 2
        elif len(pieces) == 2:
            # case 3: O: <action> : <next-state>
            # %f %f ... %f
            next_state = pieces[1]
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j, obs in enumerate(self.observations):
                self.Z[(action, next_state, obs)] = float(probs[j])
            return i + 2
        elif len(pieces) == 1:
            next_line = self.contents[i+1]
            if next_line == "identity":
                # case 4: O: <action>
                # identity
                for comb in itertools.product([action], self.states, self.observations):
                    self.Z[comb] = 1.0 if comb[1] == comb[2] else 0.0
                return i + 2
            elif next_line == "uniform":
                # case 5: O: <action>
                # uniform
                prob = 1.0 / float(len(self.observations))
                for comb in itertools.product([action], self.states, self.observations):
                    self.Z[comb] = prob
                return i + 2
            else:
                # case 6: O: <action>
                # %f %f ... %f
                # %f %f ... %f
                # ...
                # %f %f ... %f
                for j, sj in enumerate(self.states):
                    probs = next_line.split()
                    assert len(probs) == len(self.observations)
                    for k, oj in enumerate(self.observations):
                        self.Z[(action, sj, oj)] = prob = float(probs[k])
                    next_line = self.contents[i + 2 + j]
                return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __get_R(self, i):
        '''
        Wild card * are allowed when specifying a single reward
        probability. They are not allowed when specifying a vector or
        matrix of probabilities.
        '''
        line = self.contents[i]
        pieces = [x for x in line.split() if (x.find(':') == -1)]
        # try:
        #     pieces = map(float, pieces)
        # except Exception:
        #     pass
        action = pieces[0]

        if len(pieces) == 5 or len(pieces) == 4:
            # case 1:
            # R: <action> : <start-state> : <next-state> : <obs> %f
            # any of <start-state>, <next-state>, and <obs> can be *
            # %f can be on the next line (case where len(pieces) == 4)
            # start_state_raw = pieces[1]
            # next_state_raw = pieces[2]
            # obs_raw = pieces[3]
            start_state, next_state, obs = pieces[1], pieces[2], pieces[3]
            prob = float(pieces[4]) if len(pieces) == 5 else float(self.contents[i+1])

            # self.__reward_ss(
            #     action, start_state_raw, next_state_raw, obs_raw, prob)

            self.R[(action, start_state, next_state, obs)] = prob
            return i + 1 if len(pieces) == 5 else i + 2
        elif len(pieces) == 3:
            # case 2: R: <action> : <start-state> : <next-state>
            # %f %f ... %f
            start_state, next_state = pieces[1], pieces[2]
            next_line = self.contents[i+1]
            probs = next_line.split()
            assert len(probs) == len(self.observations)
            for j, obs in enumerate(self.observations):
                self.R[(action, start_state, next_state, obs)] = float(probs[j])
            return i + 2
        elif len(pieces) == 2:
            # case 3: R: <action> : <start-state>
            # %f %f ... %f
            # %f %f ... %f
            # ...
            # %f %f ... %f
            start_state = pieces[1]
            next_line = self.contents[i+1]
            for j, sj in enumerate(self.states):
                probs = next_line.split()
                assert len(probs) == len(self.observations)
                for k, oj in enumerate(self.observations):
                    prob = float(probs[k])
                    self.R[(action, start_state, sj, oj)] = prob
                next_line = self.contents[i + 2 + j]
            return i + 1 + len(self.states)
        else:
            raise Exception("Cannot parse line: " + line)

    def __reward_ss(self, a, start_state_raw, next_state_raw, obs_raw, prob):
        '''
        reward_ss means we're at the start state of the unrolling of the
        reward expression. start_state_raw could be * or the name of the
        real start state.
        '''
        if start_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ns(a, i, next_state_raw, obs_raw, prob)
        else:
            start_state = self.states.index(start_state_raw)
            self.__reward_ns(a, start_state, next_state_raw, obs_raw, prob)

    def __reward_ns(self, a, start_state, next_state_raw, obs_raw, prob):
        '''
        reward_ns means we're at the next state of the unrolling of the
        reward expression. start_state is the number of the real start
        state, and next_state_raw could be * or the name of the real
        next state.
        '''
        if next_state_raw == '*':
            for i in range(len(self.states)):
                self.__reward_ob(a, start_state, i, obs_raw, prob)
        else:
            next_state = self.states.index(next_state_raw)
            self.__reward_ob(a, start_state, next_state, obs_raw, prob)

    def __reward_ob(self, a, start_state, next_state, obs_raw, prob):
        '''
        reward_ob means we're at the observation of the unrolling of the
        reward expression. start_state is the number of the real start
        state, next_state is the number of the real next state, and
        obs_raw could be * or the name of the real observation.
        '''
        if obs_raw == '*':
            for i in range(len(self.observations)):
                self.R[(a, start_state, next_state, i)] = prob
        else:
            obs = self.observations.index(obs_raw)
            self.R[(a, start_state, next_state, obs)] = prob

    def copy_env(self):
        return {
            "model_name": self.model_name,
            "model_spec": self.model_spec,
            "discount": self.discount,
            "init_state": self.init_state,
            "values": deepcopy(self.values),
            "start": deepcopy(self.start),
            "states": deepcopy(self.states),
            "costs": deepcopy(self.costs),
            "actions": deepcopy(self.actions),
            "observations": deepcopy(self.observations),
            "T": deepcopy(self.T),
            "Z": deepcopy(self.Z),
            "R": deepcopy(self.R)
        }

    def random_beliefs(self):
        return gen_distribution(len(self.states))

    def generate_beliefs(self):
        if self.start:
            return self.start
        n_states= len(self.states)
        return [1 / n_states for _ in range(n_states)]

    def generate_belief_points(self, stepsize):
        # must be many better ways to do it
        beliefs = [[random.uniform() for s in self.states] for p in arange(0., 1. + stepsize, stepsize)]
        return array(beliefs)
