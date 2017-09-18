# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
from math import sqrt
import itertools


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        if self.queue:
            return heapq.heappop(self.queue)
        return None

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """

        self.queue[node_id] = heapq.heappop(self.queue)
        heapq.heapify(self.queue)

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        heapq.heappush(self.queue, node)

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]

    def isNull(self):
        return len(self.queue) == 0

    def get_node(self, node):
        for index, (_, path) in enumerate(self.queue):
            if node == path[-1]:
                return self.queue[index]
        return None, None


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    frontier = PriorityQueue()
    explored = []
    counter = itertools.count()
    count = next(counter)
    frontier.append((0, count, [start]))

    if start == goal:
        return []

    while not frontier.isNull():
        cost, count, path = frontier.pop()
        curr_node = path[-1]

        if curr_node not in explored:
            explored.append(curr_node)
            for neighbor in graph[curr_node]:
                if neighbor == goal:
                    new_path = list(path)
                    new_path.append(neighbor)
                    return new_path
                if neighbor not in explored:
                    new_path = list(path)
                    new_path.append(neighbor)
                    total_cost = len(new_path) - 1
                    count = next(counter)
                    frontier.append((total_cost, count, new_path))

    return []

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # TODO: finish this function!
    frontier = PriorityQueue()
    explored = []
    frontier.append((0, [start]))

    if start == goal:
        return []

    while not frontier.isNull():
        cost, path = frontier.pop()
        curr_node = path[-1]

        if curr_node == goal:
            return path

        if curr_node not in explored:
            explored.append(curr_node)
            for neighbor in graph[curr_node]:
                new_path = list(path)
                new_path.append(neighbor)
                total_cost = cost + graph[curr_node][neighbor]['weight']
                if neighbor not in explored and neighbor not in frontier:
                    frontier.append((total_cost, new_path))
                elif neighbor in frontier:
                    for last_cost, last_path in frontier:
                        last_node = last_path[-1]
                        if neighbor == last_node:
                            if total_cost < last_cost:
                                frontier.remove(neighbor)
                                frontier.append((total_cost, new_path))
                            break

    return []


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node as a list.
    """

    # TODO: finish this function!
    if v == goal:
        return 0

    curr_node_pos = graph.node[v]['pos']
    goal_node_pos = graph.node[goal]['pos']

    return sqrt(((curr_node_pos[0] - goal_node_pos[0]) ** 2) + ((curr_node_pos[1] - goal_node_pos[1]) ** 2))

def path_cost(graph, path):
    total_cost = 0
    if len(path) > 1:
        for i in range(0,len(path)-1):
            curr_node = path[i]
            next_node = path[i+1]
            total_cost += graph[curr_node][next_node]['weight']

    return total_cost


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    frontier = PriorityQueue()
    explored = []
    frontier.append((0, [start]))

    if start == goal:
        return []

    while not frontier.isNull():
        cost, path = frontier.pop()
        curr_node = path[-1]

        if curr_node == goal:
            return path

        if curr_node not in explored:
            explored.append(curr_node)
            for neighbor in graph[curr_node]:
                new_path = list(path) # makes sure a copy a made
                new_path.append(neighbor)
                total_cost = path_cost(graph, new_path)
                total_cost += heuristic(graph, neighbor, goal)
                if neighbor not in explored and neighbor not in frontier:
                    frontier.append((total_cost, new_path))
                elif neighbor in frontier:
                    for last_cost, last_path in frontier:
                        last_node = last_path[-1]
                        if neighbor == last_node:
                            if total_cost < last_cost:
                                frontier.remove(neighbor)
                                frontier.append((total_cost, new_path))
                            break

    return []


def bidirectional_ucs(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    frontier_s = PriorityQueue()
    frontier_g = PriorityQueue()
    explored_s = []
    explored_g = []
    frontier_s.append((0, [start]))
    frontier_g.append((0, [goal]))
    mu = float('inf')
    best_path = []
    
    if start == goal:
        return []

    while not frontier_s.isNull() and not frontier_g.isNull():
        
        top_s = frontier_s.top()
        top_s_cost = top_s[0]
        top_g = frontier_g.top()
        top_g_cost = top_g[0]
        if (top_s_cost + top_g_cost) >= mu:
            return best_path

        if top_s_cost <= top_g_cost:
        
            if not frontier_s.isNull():
                cost_s, path_s = frontier_s.pop()
                curr_node_s = path_s[-1]
        
            if curr_node_s in explored_g:
                cost_g, path_g = frontier_g.get_node(curr_node_s)
                path_g.reverse()
                if best_cost > cost_s + cost_g:
                    best_cost = cost_s + cost_g
                    best_path = path_s + path_g

            if curr_node_s not in explored_s:
                explored_s.append(curr_node_s)
                for neighbor in graph[curr_node_s]:
                    new_path = list(path_s)
                    new_path.append(neighbor)
                    total_cost = cost_s + graph[curr_node_s][neighbor]['weight']
                    if neighbor not in explored_s and neighbor not in frontier_s:
                        frontier_s.append((total_cost, new_path))
                    elif neighbor in frontier_s:
                        for last_cost, last_path in frontier_s:
                            last_node = last_path[-1]
                            if neighbor == last_node:
                                if total_cost < last_cost:
                                    frontier_s.remove(neighbor)
                                    frontier_s.append((total_cost, new_path))
                                break
        else:
            if not frontier_g.isNull():
                cost_g, path_g = frontier_g.pop()
                curr_node_g = path_g[-1]
        
            if curr_node_g in explored_s:
                cost_s, path_s = frontier_s.get_node(curr_node_g)
                path_s.reverse()
                if best_cost > cost_s + cost_g:
                    best_cost = cost_s + cost_g
                    best_path = path_g + path_s

            if curr_node_g not in explored_g:
                explored_g.append(curr_node_g)
                for neighbor in graph[curr_node_g]:
                    new_path = list(path_g)
                    new_path.append(neighbor)
                    total_cost = cost_g + graph[curr_node_g][neighbor]['weight']
                    if neighbor not in explored_g and neighbor not in frontier_g:
                        frontier_g.append((total_cost, new_path))
                    elif neighbor in frontier_g:
                        for last_cost, last_path in frontier_g:
                            last_node = last_path[-1]
                            if neighbor == last_node:
                                if total_cost < last_cost:
                                    frontier_g.remove(neighbor)
                                    frontier_g.append((total_cost, new_path))
                                break

    return []



def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
