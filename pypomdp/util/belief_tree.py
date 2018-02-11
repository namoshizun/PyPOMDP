
from util.helper import rand_choice, round
from abc import abstractmethod

class Node(object):
    def __init__(self, nid, name, h, parent=None, V=0, N=0):
        self.h = h
        self.V = V
        self.N = N
        self.id = nid
        self.name = name
        self.parent = parent
        self.children = []

    @abstractmethod
    def add_child(self, node):
        """
         To be implemented.
        """

    @abstractmethod
    def get_child(self, *args):
        """
         To be implemented.
        """

class BeliefNode(Node):
    """
    Represents a node that holds the belief distribution given its history sequence in a belief tree.
    It also holds the received observation after which the belief is updated accordingly
    """
    def __init__(self, nid, name, h, obs_index, parent=None, V=0, N=0, budget=float('inf')):
        Node.__init__(self, nid, name, h, parent, V, N)
        self.observation = obs_index
        self.budget = budget
        self.B = []
        self.action_map = {}

    def add_child(self, node):
        self.children.append(node)
        self.action_map[node.action] = node

    def get_child(self, action):
        return self.action_map.get(action, None)

    def sample_state(self):
        return rand_choice(self.B)

    def add_particle(self, particle):
        if type(particle) is list:
            self.B.extend(particle)
        else:
            self.B.append(particle)

    def __repr__(self):
        return 'Bid = {}, N = {}'.format(self.id, self.N)


class ActionNode(Node):
    """
    represents the node associated with an POMDP action
    """
    def __init__(self, nid, name, h, action_index, cost, parent=None, V=0, N=0):
        Node.__init__(self, nid, name, h, parent, V, N)
        self.mean_reward = 0.0
        self.mean_cost = 0.0
        self.cost = cost
        self.action = action_index
        self.obs_map = {}

    def update_stats(self, cost, reward):
        self.mean_cost = (self.mean_cost * self.N + cost) / (self.N + 1)
        self.mean_reward = (self.mean_reward * self.N + reward) / (self.N + 1)

    def add_child(self, node):
        self.children.append(node)
        self.obs_map[node.observation] = node

    def get_child(self, observation):
        return self.obs_map.get(observation, None)

    def __repr__(self):
        return 'Aid = {}, N = {}, V = {}'.format(self.id, self.N, round(self.V, 6))


class BeliefTree:
    """
    The belief tree decipted in Silver's POMCP paper.
    """
    def __init__(self, total_budget, root_particles):
        """
        :param root_particles: particles sampled from the prior belief distribution; used as initial root's particle set
        """
        self.counter = 0
        self.nodes = {}
        self.root = self.add(h=[], name='root', particle=root_particles, budget=total_budget)

    def __pretty_print__(self, root, depth):
        if not root.children:
            # the leaf
            return

        for node in root.children:
            print('|  ' * depth + str(node))
            self.__pretty_print__(node, depth + 1)

    def add(self, h, name, parent=None, action=None, observation=None,
            particle=None, budget=None, cost=None):
        """
        Creates and adds a new belief node or action node to the belief search tree

        :param h: history sequence
        :param parent: either ActionNode or BeliefNode
        :param action: action name
        :param observation: observation name
        :param particle: new node's particle set
        :param budget: remaining budget of a belief nodde
        :param cost: action cost of an action node
        :return:
        """
        history = h[:]

        # instantiate node
        if action is not None:
            n = ActionNode(self.counter, name, history, parent=parent, action_index=action, cost=cost)
        else:
            n = BeliefNode(self.counter, name, history, parent=parent, obs_index=observation, budget=budget)

        if particle is not None:
            n.add_particle(particle)

        # add the node to belief tree
        self.nodes[n.id] = n
        self.counter += 1

        # register node as parent's child
        if parent is not None:
            parent.add_child(n)
        return n

    def find_or_create(self, h, **kwargs):
        """
        Search for the node corrresponds to given history, otherwise create one using given params
        """
        curr = self.root
        h_len, root_history_len = len(h), len(self.root.h)

        for step in range(root_history_len, h_len):
            curr = curr.get_child(h[step])
            if curr is None:
                return self.add(h, **kwargs)
        return curr

    def prune(self, node, exclude=None):
        """
        Removes the entire subtree subscribed to 'node' with exceptions.
        :param node: root of the subtree to be removed
        :param exclude: exception component
        :return:
        """
        for child in node.children:
            if exclude and exclude.id != child.id:
                self.prune(child, exclude)

        self.nodes[node.id] = None
        del self.nodes[node.id]

    def prune_siblings(self, node):
        siblings = [child for child in node.parent.children if child.id != node.id]
        for sb in siblings:
            self.prune(sb)

    def pretty_print(self):
        """
         pretty prints tree's structure
        """
        print(self.root)
        self.__pretty_print__(self.root, depth=1)
