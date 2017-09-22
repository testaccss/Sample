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
import copy


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

        self.queue[node_id] = self.top()
        heapq.heappop(self.queue)
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

    def isNodeLastInQueue(self, key):
        nodes = [n for _, n in self.queue]
        for index, node in enumerate(nodes):
            if key == node[-1]:
                return True, index
        return False, 0

    def isNodeInQueue(self, key):
        nodes = [n for _, n in self.queue]
        for index, node in enumerate(nodes):
            if key in node:
                return True
        return False

    def node_index(self, key):
        cost = float('inf')
        node_id = -1
        nodes = [n for _, n in self.queue]
        for index, node in enumerate(nodes):
            if key in node:
                curr_cost = self.queue[index][0]
                if curr_cost < cost:
                    cost = curr_cost
                    node_id = index
        return node_id

    def get(self,nodeId):
        for index, (c, n) in enumerate(self.queue):
            if nodeId == n[-1]:
                return self.queue[index]
        return []


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
                is_last, node_id = frontier.isNodeLastInQueue(neighbor)
                if (neighbor not in explored) and (not is_last):
                    frontier.append((total_cost, new_path))
                elif is_last:
                    last_cost, last_path = frontier.queue[node_id]
                    if total_cost < last_cost:
                        frontier.remove(node_id)
                        frontier.append((total_cost, new_path))

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
        for i in range(0, len(path) - 1):
            curr_node = path[i]
            next_node = path[i + 1]
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
                new_path = list(path)
                new_path.append(neighbor)
                total_cost = path_cost(graph, new_path)
                total_cost += heuristic(graph, neighbor, goal)
                is_last, node_id = frontier.isNodeLastInQueue(neighbor)
                if (neighbor not in explored) and (not is_last):
                    frontier.append((total_cost, new_path))
                elif is_last:
                    last_cost, last_path = frontier.queue[node_id]
                    if total_cost < last_cost:
                        frontier.remove(node_id)
                        frontier.append((total_cost, new_path))
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
    count = 0

    if start == goal:
        return []

    while not frontier_s.isNull() and not frontier_g.isNull():

        top_s = frontier_s.top()
        top_s_cost = top_s[0]
        top_g = frontier_g.top()
        top_g_cost = top_g[0]

        if (top_s_cost + top_g_cost) >= mu:
            return best_path

        count = count + 1
        if top_s_cost <= top_g_cost:

            if not frontier_s.isNull():
                cost_s, path_s = frontier_s.pop()
                curr_node_s = path_s[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_s == goal:
                    if mu > cost_s:
                        mu = cost_s
                        best_path = copy.deepcopy(path_s)
                if frontier_g.isNodeInQueue(curr_node_s):
                    # if neighbor in explored_g:
                    node_id = frontier_g.node_index(curr_node_s)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu > cost_s + cost_g:
                            mu = cost_s + cost_g
                            best_path = path_s + path_g[1:]

                if curr_node_s not in explored_s:
                    explored_s.append(curr_node_s)
                    for neighbor in graph[curr_node_s]:
                        new_path = list(path_s)
                        new_path.append(neighbor)
                        total_cost = cost_s + graph[curr_node_s][neighbor]['weight']
                        is_last, node_id = frontier_s.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_s) and (not is_last):
                            frontier_s.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_s.queue[node_id]
                            if total_cost < last_cost:
                                frontier_s.remove(node_id)
                                frontier_s.append((total_cost, new_path))
        else:

            if not frontier_g.isNull():
                cost_g, path_g = frontier_g.pop()
                curr_node_g = path_g[-1]

                # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == start:
                    if mu > total_cost:
                        mu = total_cost
                        best_path = copy.deepcopy(path_g)
                        best_path.reverse()
                # if neighbor in explored_s:
                if frontier_s.isNodeInQueue(curr_node_g):
                    node_id = frontier_s.node_index(curr_node_g)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_gg = copy.deepcopy(path_g)
                        path_gg.reverse()
                        if mu > cost_g + cost_s:
                            mu = cost_g + cost_s
                            best_path = path_s[:len(path_s) - 1] + path_gg

                if curr_node_g not in explored_g:
                    explored_g.append(curr_node_g)
                    for neighbor in graph[curr_node_g]:
                        new_path = list(path_g)
                        new_path.append(neighbor)
                        total_cost = cost_g + graph[curr_node_g][neighbor]['weight']
                        is_last, node_id = frontier_g.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_g) and (not is_last):
                            frontier_g.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_g.queue[node_id]
                            if total_cost < last_cost:
                                frontier_g.remove(node_id)
                                frontier_g.append((total_cost, new_path))
    return best_path


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
    frontier_s = PriorityQueue()
    frontier_g = PriorityQueue()
    explored_s = []
    explored_g = []
    frontier_s.append((0, [start]))
    frontier_g.append((0, [goal]))
    mu = float('inf')
    best_path = []
    count = 0

    if start == goal:
        return []

    while not frontier_s.isNull() and not frontier_g.isNull():

        top_s = frontier_s.top()
        top_s_cost = top_s[0]
        top_g = frontier_g.top()
        top_g_cost = top_g[0]

        if (top_s_cost + top_g_cost) >= mu:
            return best_path

        count = count + 1
        # if top_s_cost < (mu + cost):
        if top_s_cost <= top_g_cost:

            if not frontier_s.isNull():
                cost_s, path_s = frontier_s.pop()
                curr_node_s = path_s[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_s == goal:
                    if mu > cost_s:
                        mu = cost_s
                        best_path = copy.deepcopy(path_s)
                if frontier_g.isNodeInQueue(curr_node_s):
                    # if neighbor in explored_g:
                    node_id = frontier_g.node_index(curr_node_s)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu > cost_s + cost_g:
                            mu = cost_s + cost_g
                            best_path = path_s + path_g[1:]

                if curr_node_s not in explored_s:
                    explored_s.append(curr_node_s)
                    for neighbor in graph[curr_node_s]:
                        new_path = list(path_s)
                        new_path.append(neighbor)
                        total_cost = path_cost(graph, new_path)
                        hs = heuristic(graph, neighbor, start)
                        hg = heuristic(graph, neighbor, goal)
                        total_cost = total_cost + ((hg - hs) / 2)
                        is_last, node_id = frontier_s.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_s) and (not is_last):
                            frontier_s.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_s.queue[node_id]
                            if total_cost < last_cost:
                                frontier_s.remove(node_id)
                                frontier_s.append((total_cost, new_path))

        # if top_g_cost < (mu + cost):
        else:

            if not frontier_g.isNull():
                cost_g, path_g = frontier_g.pop()
                curr_node_g = path_g[-1]

                # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == start:
                    if mu > cost_g:
                        mu = cost_g
                        best_path = copy.deepcopy(path_g)
                        best_path.reverse()
                # if neighbor in explored_s:
                if frontier_s.isNodeInQueue(curr_node_g):
                    node_id = frontier_s.node_index(curr_node_g)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_gg = copy.deepcopy(path_g)
                        path_gg.reverse()
                        if mu > cost_g + cost_s:
                            mu = cost_g + cost_s
                            best_path = path_s[:len(path_s) - 1] + path_gg

                if curr_node_g not in explored_g:
                    explored_g.append(curr_node_g)
                    for neighbor in graph[curr_node_g]:
                        new_path = list(path_g)
                        new_path.append(neighbor)
                        total_cost = path_cost(graph, new_path)
                        hs = heuristic(graph, neighbor, start)
                        hg = heuristic(graph, neighbor, goal)
                        total_cost = total_cost + ((hs - hg) / 2)
                        is_last, node_id = frontier_g.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_g) and (not is_last):
                            frontier_g.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_g.queue[node_id]
                            if total_cost < last_cost:
                                frontier_g.remove(node_id)
                                frontier_g.append((total_cost, new_path))
    return best_path

def get_min_path_1(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg):

    print mu_sg, path_sg
    print mu_cg, path_cg
    print mu_sc, path_sc
    if mu_sc + mu_sg < mu_sc + mu_cg and mu_sc + mu_sg < mu_cg + mu_sg:
        return path_sg + path_sc[1::]
    elif mu_sc + mu_cg < mu_cg + mu_sg:
        return path_sc + path_cg[1::]
    else:
        return path_cg + path_sg[1::]

def get_min_path(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg):

    print mu_sg, path_sg
    print mu_cg, path_cg
    print mu_sc, path_sc
    sc = set(copy.deepcopy(path_sc))
    cg = set(copy.deepcopy(path_cg))
    sg = set(copy.deepcopy(path_sg))

    if mu_sc + mu_sg < mu_sc + mu_cg and mu_sc + mu_sg < mu_cg + mu_sg: # and path_sc[0] == path_sg[-1]:
        if sg.issubset(sc):
            return path_sc
        elif sc.issubset(sg):
            return path_sg
        return path_sg + path_sc[1::]
    elif mu_sc + mu_cg < mu_cg + mu_sg:# and path_cg[0] == path_sc[-1]:
        if sc.issubset(cg):
            return path_cg
        elif cg.issubset(sc):
            return path_sc
        return path_sc + path_cg[1::]
    else:
        if sg.issubset(cg):
            return path_cg
        elif cg.issubset(sg):
            return path_sg
        return path_cg + path_sg[1::]

def get_min_path_old(flag_sg, flag_cg, flag_sc, path_sc, path_sg, path_cg):

    if flag_sc + flag_cg + flag_sg > 1:
        if flag_sc and flag_cg:
            for n in path_cg:
                if n not in path_sc:
                    path_sc.append(n)
            return path_sc
        if flag_sc and flag_sg:
            for n in path_sg:
                if n not in path_sc:
                    path_sc.append(n)
            return path_sc

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

    if len(goals) != len(set(goals)):
        return []

    start = goals[0]
    center = goals[1]
    goal = goals[2]
    frontier_s = PriorityQueue()
    frontier_c = PriorityQueue()
    frontier_g = PriorityQueue()
    explored_s = []
    explored_c = []
    explored_g = []
    frontier_s.append((0, [start]))
    frontier_c.append((0, [center]))
    frontier_g.append((0, [goal]))
    mu_sc = float('inf')
    mu_cg = float('inf')
    mu_sg = float('inf')
    flag_sg = 0
    flag_cg = 0
    flag_sc = 0
    count = 0
    path_sc = []
    path_sg = []
    path_cg = []
    total_explored = 0

    while not frontier_s.isNull() and not frontier_g.isNull() and not frontier_c.isNull():

        top_s = frontier_s.top()
        top_s_cost = top_s[0]
        top_c = frontier_c.top()
        top_c_cost = top_c[0]
        top_g = frontier_g.top()
        top_g_cost = top_g[0]

        #if (top_s_cost >= mu_sc and top_c_cost >= mu_cg) or (top_c_cost >= mu_cg and top_g_cost >= mu_sg)\
        #        or (top_s_cost >= mu_sc and top_g_cost >= mu_sg):
        #if top_s_cost >= mu_sc and top_c_cost >= mu_cg and top_g_cost >= mu_sg:
        #if (top_s_cost + top_c_cost + top_g_cost >= (mu_sg + mu_sc + mu_cg)):

        if (top_s_cost + top_c_cost >= mu_sc) and (top_c_cost + top_g_cost >= mu_cg) and (top_s_cost + top_g_cost >= mu_sg):
            return get_min_path(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg)

        count = count + 1
        if top_s_cost <= top_g_cost and top_s_cost <= top_c_cost:
        #if top_s_cost < mu_sc:
        #if top_s_cost < mu_sc or top_s_cost < mu_sg:

            if not frontier_s.isNull():
                cost_s, path_s = frontier_s.pop()
                curr_node_s = path_s[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_s == center:
                    if mu_sc > cost_s:
                        mu_sc = cost_s
                        path_sc = copy.deepcopy(path_s)
                        flag_sc = 1
                if curr_node_s == goal:
                    if mu_sg > cost_s:
                        mu_sg = cost_s
                        path_sg = copy.deepcopy(path_s)
                        flag_sg = 1
                        if path_sg[-1] == goal:
                            path_sg.reverse()
                if frontier_c.isNodeLastInQueue(curr_node_s)[0]:
                    node_id = frontier_c.node_index(curr_node_s)
                    if node_id > -1:
                        cost_c, path_c = copy.deepcopy(frontier_c.queue[node_id])
                        path_c.reverse()
                        if mu_sc > cost_s + cost_c:
                            mu_sc = cost_s + cost_c
                            path_sc = path_s[0:-1] + path_c
                            flag_sc = 1
                if frontier_g.isNodeLastInQueue(curr_node_s)[0]:
                    node_id = frontier_g.node_index(curr_node_s)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu_sg > cost_s + cost_g:
                            mu_sg = cost_s + cost_g
                            path_sg = path_s[0:-1] + path_g
                            if path_sg[-1] == goal:
                                path_sg.reverse()
                            flag_sg = 1

                if curr_node_s not in explored_s:
                    explored_s.append(curr_node_s)
                    for neighbor in graph[curr_node_s]:
                        new_path = list(path_s)
                        new_path.append(neighbor)
                        if neighbor in explored_s:
                            continue
                        total_cost = cost_s + graph[curr_node_s][neighbor]['weight']
                        is_last, node_id = frontier_s.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_s) and (not is_last):
                            frontier_s.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_s.queue[node_id]
                            if total_cost < last_cost:
                                frontier_s.remove(node_id)
                                frontier_s.append((total_cost, new_path))

        if top_c_cost <= top_s_cost and top_c_cost <= top_g_cost:
        #if top_c_cost < mu_cg:
        #if top_c_cost < mu_cg or top_c_cost < mu_sc:

            if not frontier_c.isNull():
                cost_c, path_c = frontier_c.pop()
                curr_node_c = path_c[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_c == goal:
                    if mu_cg > cost_c:
                        mu_cg = cost_c
                        path_cg = copy.deepcopy(path_c)
                        flag_cg = 1
                if curr_node_c == start:
                    if mu_sc > cost_c:
                        mu_sc = cost_c
                        path_sc = copy.deepcopy(path_c)
                        flag_sc = 1
                        if path_sc[-1] == start:
                            path_sc.reverse()
                if frontier_s.isNodeLastInQueue(curr_node_c)[0]:
                    node_id = frontier_s.node_index(curr_node_c)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_s.reverse()
                        if mu_sc > cost_s + cost_c:
                            mu_sc = cost_s + cost_c
                            path_sc = path_c[0:-1] + path_s
                            if path_sc[-1] == start:
                                path_sc.reverse()
                            flag_sc = 1
                if frontier_g.isNodeLastInQueue(curr_node_c)[0]:
                    node_id = frontier_g.node_index(curr_node_c)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu_cg > cost_c + cost_g:
                            mu_cg = cost_c + cost_g
                            path_cg = path_c[0:-1] + path_g
                            flag_cg = 1

                if curr_node_c not in explored_c:
                    explored_c.append(curr_node_c)
                    for neighbor in graph[curr_node_c]:
                        new_path = list(path_c)
                        new_path.append(neighbor)
                        if neighbor in explored_c:
                            continue
                        total_cost = cost_c + graph[curr_node_c][neighbor]['weight']
                        is_last, node_id = frontier_c.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_c) and (not is_last):
                            frontier_c.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_c.queue[node_id]
                            if total_cost < last_cost:
                                frontier_c.remove(node_id)
                                frontier_c.append((total_cost, new_path))
        #else:
        if top_g_cost <= top_s_cost and top_g_cost <= top_c_cost:
        #if top_g_cost < mu_sg:

        #if top_g_cost < mu_sg or top_g_cost < mu_cg:
            if not frontier_g.isNull():
                cost_g, path_g = frontier_g.pop()
                curr_node_g = path_g[-1]

                # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == start:
                    if mu_sg > cost_g:
                        mu_sg = cost_g
                        path_sg = copy.deepcopy(path_g)
                        flag_sg = 1
                    # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == center:
                    if mu_cg > cost_g:
                        mu_cg = cost_g
                        path_cg = copy.deepcopy(path_g)
                        flag_cg = 1
                        if path_cg[-1] == center:
                            path_cg.reverse()

                # if neighbor in explored_s:
                if frontier_s.isNodeLastInQueue(curr_node_g)[0]:
                    node_id = frontier_s.node_index(curr_node_g)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_gg = copy.deepcopy(path_s)
                        path_gg.reverse()
                        if mu_sg > cost_g + cost_s:
                            mu_sg = cost_g + cost_s
                            path_sg = path_g[0:-1] + path_gg
                            flag_sg = 1
                if frontier_c.isNodeLastInQueue(curr_node_g)[0]:
                    node_id = frontier_c.node_index(curr_node_g)
                    if node_id > -1:
                        cost_c, path_c = copy.deepcopy(frontier_c.queue[node_id])
                        path_gg = copy.deepcopy(path_c)
                        path_gg.reverse()
                        if mu_cg > cost_g + cost_c:
                            mu_cg = cost_g + cost_c
                            path_cg = path_g[0:-1] + path_gg
                            if path_cg[-1] == center:
                                path_cg.reverse()
                            flag_cg = 1

                if curr_node_g not in explored_g:
                    explored_g.append(curr_node_g)
                    for neighbor in graph[curr_node_g]:
                        new_path = list(path_g)
                        new_path.append(neighbor)
                        if neighbor in explored_g:
                            continue
                        total_cost = cost_g + graph[curr_node_g][neighbor]['weight']
                        is_last, node_id = frontier_g.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_g) and (not is_last):
                            frontier_g.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_g.queue[node_id]
                            if total_cost < last_cost:
                                frontier_g.remove(node_id)
                                frontier_g.append((total_cost, new_path))

    return get_min_path(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg)


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
    if len(goals) != len(set(goals)):
        return []

    start = goals[0]
    center = goals[1]
    goal = goals[2]
    frontier_s = PriorityQueue()
    frontier_c = PriorityQueue()
    frontier_g = PriorityQueue()
    explored_s = []
    explored_c = []
    explored_g = []
    frontier_s.append((0, [start]))
    frontier_c.append((0, [center]))
    frontier_g.append((0, [goal]))
    mu_sc = float('inf')
    mu_cg = float('inf')
    mu_sg = float('inf')
    flag_sg = 0
    flag_cg = 0
    flag_sc = 0
    count = 0
    path_sc = []
    path_sg = []
    path_cg = []
    total_explored = 0

    while not frontier_s.isNull() and not frontier_g.isNull() and not frontier_c.isNull():

        top_s = frontier_s.top()
        top_s_cost = top_s[0]
        top_c = frontier_c.top()
        top_c_cost = top_c[0]
        top_g = frontier_g.top()
        top_g_cost = top_g[0]

        #if (top_s_cost >= mu_sc and top_c_cost >= mu_cg) or (top_c_cost >= mu_cg and top_g_cost >= mu_sg)\
        #        or (top_s_cost >= mu_sc and top_g_cost >= mu_sg):
        #if (top_s_cost + top_c_cost >= mu_sc) and (top_c_cost  + top_g_cost >= mu_cg)\
        #       and (top_s_cost + top_g_cost >= mu_sg):
        #if top_s_cost >= mu_sc and top_c_cost >= mu_cg and top_g_cost >= mu_sg:
        if (top_s_cost + top_c_cost >= mu_sc) and (top_c_cost + top_g_cost >= mu_cg) and (top_s_cost + top_g_cost >= mu_sg):
            return get_min_path(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg)

        count = count + 1
        #if top_s_cost <= top_g_cost and top_s_cost <= top_c_cost:
        #if top_s_cost < mu_sc:
        if top_s_cost < mu_sc or top_s_cost < mu_sg:

            if not frontier_s.isNull():
                cost_s, path_s = frontier_s.pop()
                curr_node_s = path_s[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_s == center:
                    if mu_sc > cost_s:
                        mu_sc = cost_s
                        path_sc = copy.deepcopy(path_s)
                        flag_sc = 1
                if curr_node_s == goal:
                    if mu_sg > cost_s:
                        mu_sg = cost_s
                        path_sg = copy.deepcopy(path_s)
                        flag_sg = 1
                        if path_sg[-1] == goal:
                            path_sg.reverse()
                if frontier_c.isNodeLastInQueue(curr_node_s)[0]:
                    node_id = frontier_c.node_index(curr_node_s)
                    if node_id > -1:
                        cost_c, path_c = copy.deepcopy(frontier_c.queue[node_id])
                        path_c.reverse()
                        if mu_sc > cost_s + cost_c:
                            mu_sc = cost_s + cost_c
                            path_sc = path_s[0:-1] + path_c
                            flag_sc = 1
                if frontier_g.isNodeLastInQueue(curr_node_s)[0]:
                    node_id = frontier_g.node_index(curr_node_s)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu_sg > cost_s + cost_g:
                            mu_sg = cost_s + cost_g
                            path_sg = path_s[0:-1] + path_g
                            if path_sg[-1] == goal:
                                path_sg.reverse()
                            flag_sg = 1

                if curr_node_s not in explored_s:
                    explored_s.append(curr_node_s)
                    for neighbor in graph[curr_node_s]:
                        new_path = list(path_s)
                        new_path.append(neighbor)
                        if neighbor in explored_s:
                            continue
                        total_cost = path_cost(graph, new_path)
                        hs = heuristic(graph, neighbor, start)
                        hc = heuristic(graph, neighbor, center)
                        hg = heuristic(graph, neighbor, goal)
                        total_cost = total_cost - ((hg - hs) / 2)
                        #total_cost = total_cost + ((hc - hs) / 8)
                        is_last, node_id = frontier_s.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_s) and (not is_last):
                            frontier_s.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_s.queue[node_id]
                            if total_cost < last_cost:
                                frontier_s.remove(node_id)
                                frontier_s.append((total_cost, new_path))

        #if top_c_cost <= top_s_cost and top_c_cost <= top_g_cost:
        #if top_c_cost < mu_cg:
        if top_c_cost < mu_cg or top_c_cost < mu_sc:

            if not frontier_c.isNull():
                cost_c, path_c = frontier_c.pop()
                curr_node_c = path_c[-1]

                # if frontier_g.isNodeInQueue(neighbor):
                if curr_node_c == goal:
                    if mu_cg > cost_c:
                        mu_cg = cost_c
                        path_cg = copy.deepcopy(path_c)
                        flag_cg = 1
                if curr_node_c == start:
                    if mu_sc > cost_c:
                        mu_sc = cost_c
                        path_sc = copy.deepcopy(path_c)
                        flag_sc = 1
                        if path_sc[-1] == start:
                            path_sc.reverse()
                if frontier_s.isNodeLastInQueue(curr_node_c)[0]:
                    node_id = frontier_s.node_index(curr_node_c)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_s.reverse()
                        if mu_sc > cost_s + cost_c:
                            mu_sc = cost_s + cost_c
                            path_sc = path_c[0:-1] + path_s
                            if path_sc[-1] == start:
                                path_sc.reverse()
                            flag_sc = 1
                if frontier_g.isNodeLastInQueue(curr_node_c)[0]:
                    node_id = frontier_g.node_index(curr_node_c)
                    if node_id > -1:
                        cost_g, path_g = copy.deepcopy(frontier_g.queue[node_id])
                        path_g.reverse()
                        if mu_cg > cost_c + cost_g:
                            mu_cg = cost_c + cost_g
                            path_cg = path_c[0:-1] + path_g
                            flag_cg = 1

                if curr_node_c not in explored_c:
                    explored_c.append(curr_node_c)
                    for neighbor in graph[curr_node_c]:
                        new_path = list(path_c)
                        new_path.append(neighbor)
                        if neighbor in explored_c:
                            continue
                        total_cost = path_cost(graph, new_path)
                        hs = heuristic(graph, neighbor, start)
                        hc = heuristic(graph, neighbor, center)
                        hg = heuristic(graph, neighbor, goal)
                        total_cost = total_cost + ((hg - hc) / 2)
                        #total_cost = total_cost - ((hs - hc) / 8)
                        is_last, node_id = frontier_c.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_c) and (not is_last):
                            frontier_c.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_c.queue[node_id]
                            if total_cost < last_cost:
                                frontier_c.remove(node_id)
                                frontier_c.append((total_cost, new_path))
        #else:
        #if top_g_cost <= top_s_cost and top_g_cost <= top_c_cost:
        #if top_g_cost < mu_sg:

        if top_g_cost < mu_sg or top_g_cost < mu_cg:
            if not frontier_g.isNull():
                cost_g, path_g = frontier_g.pop()
                curr_node_g = path_g[-1]

                # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == start:
                    if mu_sg > cost_g:
                        mu_sg = cost_g
                        path_sg = copy.deepcopy(path_g)
                        flag_sg = 1
                    # if frontier_s.isNodeInQueue(neighbor):
                if curr_node_g == center:
                    if mu_cg > cost_g:
                        mu_cg = cost_g
                        path_cg = copy.deepcopy(path_g)
                        flag_cg = 1
                        if path_cg[-1] == center:
                            path_cg.reverse()

                # if neighbor in explored_s:
                if frontier_s.isNodeLastInQueue(curr_node_g)[0]:
                    node_id = frontier_s.node_index(curr_node_g)
                    if node_id > -1:
                        cost_s, path_s = copy.deepcopy(frontier_s.queue[node_id])
                        path_gg = copy.deepcopy(path_s)
                        path_gg.reverse()
                        if mu_sg > cost_g + cost_s:
                            mu_sg = cost_g + cost_s
                            path_sg = path_g[0:-1] + path_gg
                            flag_sg = 1
                if frontier_c.isNodeLastInQueue(curr_node_g)[0]:
                    node_id = frontier_c.node_index(curr_node_g)
                    if node_id > -1:
                        cost_c, path_c = copy.deepcopy(frontier_c.queue[node_id])
                        path_gg = copy.deepcopy(path_c)
                        path_gg.reverse()
                        if mu_cg > cost_g + cost_c:
                            mu_cg = cost_g + cost_c
                            path_cg = path_g[0:-1] + path_gg
                            if path_cg[-1] == center:
                                path_cg.reverse()
                            flag_cg = 1

                if curr_node_g not in explored_g:
                    explored_g.append(curr_node_g)
                    for neighbor in graph[curr_node_g]:
                        new_path = list(path_g)
                        new_path.append(neighbor)
                        if neighbor in explored_g:
                            continue
                        total_cost = path_cost(graph, new_path)
                        hs = heuristic(graph, neighbor, start)
                        hc = heuristic(graph, neighbor, center)
                        hg = heuristic(graph, neighbor, goal)
                        total_cost = total_cost + ((hs - hg) / 2)
                        #total_cost = total_cost - ((hc - hg) / 8)
                        is_last, node_id = frontier_g.isNodeLastInQueue(neighbor)
                        if (neighbor not in explored_g) and (not is_last):
                            frontier_g.append((total_cost, new_path))
                        elif is_last:
                            last_cost, last_path = frontier_g.queue[node_id]
                            if total_cost < last_cost:
                                frontier_g.remove(node_id)
                                frontier_g.append((total_cost, new_path))

    return get_min_path(mu_sg,mu_cg,mu_sc, path_sc,path_sg, path_cg)


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return 'Sridhar Sampath'


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
