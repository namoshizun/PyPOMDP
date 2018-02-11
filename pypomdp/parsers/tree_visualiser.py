import json
import graphviz
from abc import abstractmethod
from util.belief_tree import Node, ActionNode, BeliefNode

class TreeVisualiser(object):

    def __init__(self, description):
        self.description = description
        self.graph = None

    @abstractmethod
    def update(self, root, append_to=None):
        """
        To be implemented by visualiser
        """

    @abstractmethod
    def render(self, **kwargs):
        """
        To be implemented by visualiser
        """


class GraphViz(TreeVisualiser):

    def __init__(self, description):
        TreeVisualiser.__init__(self, description)
        # self.graph = graphviz.Digraph(description)

    def update(self, node, parent=None):
        self.graph = graphviz.Digraph(self.description)
        self.graph.attr(rankdir='LR')
        self.__update(node)
        if parent:
            self.graph.edge(parent, node)

    def render(self, fname=None, directory=None):
        self.graph.render(filename=fname, directory=directory)

    def __update(self, node):
        if not node.children:
            return

        for ch in node.children:
            self.graph.edge(str(node), str(ch), label=ch.name )
            self.__update(ch)


