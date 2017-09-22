# coding=utf-8
import pickle
import random
import unittest

import matplotlib.pyplot as plt
import networkx

from explorable_graph import ExplorableGraph
from search_submission import PriorityQueue, a_star, bidirectional_a_star, \
    bidirectional_ucs, breadth_first_search, uniform_cost_search, tridirectional_search, tridirectional_upgraded
from visualize_graph import plot_search


class TestBidirectionalSearch(unittest.TestCase):
    """Test the bidirectional search algorithms: UCS, A*"""

    def setUp(self):
        """Romania map data from Russell and Norvig, Chapter 3."""
        romania = pickle.load(open('romania_graph.pickle', 'rb'))
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()

    def test_ucs_tri(self):
        """TTest and visualize uniform-cost search"""
        start = 'd'
        center = 'o'
        goal = 'n'
        goals = [start, center, goal]

        node_positions = {n: self.romania.node[n]['pos'] for n in
                          self.romania.node.keys()}

        self.romania.reset_search()
        path = tridirectional_search(self.romania, goals)
        print 'Tri', path

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)

        path = tridirectional_upgraded(self.romania, goals)
        print 'TriU', path

        self.draw_graph(self.romania, node_positions=node_positions,
                        start=start, goal=goal, path=path)

    if 0:
        def test_ucs(self):
            """TTest and visualize uniform-cost search"""
            start = 'a'
            goal = 'u'

            node_positions = {n: self.romania.node[n]['pos'] for n in
                              self.romania.node.keys()}

            self.romania.reset_search()
            path = uniform_cost_search(self.romania, start, goal)
            print 'Uni', path

            self.draw_graph(self.romania, node_positions=node_positions,
                            start=start, goal=goal, path=path)

        def test_ucs_bi(self):
            """TTest and visualize uniform-cost search"""
            start = 'a'
            goal = 'u'

            node_positions = {n: self.romania.node[n]['pos'] for n in
                              self.romania.node.keys()}

            self.romania.reset_search()
            path = bidirectional_ucs(self.romania, start, goal)
            print 'Bi', path

            self.draw_graph(self.romania, node_positions=node_positions,
                            start=start, goal=goal, path=path)

    @staticmethod
    def draw_graph(graph, node_positions=None, start=None, goal=None,
                   path=None):
        """Visualize results of graph search"""
        explored = list(graph.explored_nodes)

        labels = {}
        for node in graph:
            labels[node] = node

        if node_positions is None:
            node_positions = networkx.spring_layout(graph)

        networkx.draw_networkx_nodes(graph, node_positions)
        networkx.draw_networkx_edges(graph, node_positions, style='dashed')
        networkx.draw_networkx_labels(graph, node_positions, labels)

        networkx.draw_networkx_nodes(graph, node_positions, nodelist=explored,
                                     node_color='g')

        if path is not None:
            edges = [(path[i], path[i + 1]) for i in range(0, len(path) - 1)]
            networkx.draw_networkx_edges(graph, node_positions, edgelist=edges,
                                         edge_color='b')

        if start:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[start], node_color='b')

        if goal:
            networkx.draw_networkx_nodes(graph, node_positions,
                                         nodelist=[goal], node_color='y')

        plt.plot()
        plt.show()

if __name__ == '__main__':
    unittest.main()
