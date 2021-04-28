from aima3.search import *
from utils import *
from collections import deque
from blocks_world import BlocksWorld

# def get_item(queue, key):
#     """Returns the first node associated with key in PriorityQueue.
#     Raises KeyError if key is not present."""
#     for _, item in queue.heap:
#         if item == key:
#             return item
#     raise KeyError(str(key) + " is not in the priority queue")

def show_solution(name_algo, node):
    try:
        print(name_algo + ":", node.solution())
    except:
        if type(Node) == str:
            print(name_algo + ":", node)
        else:
            print(name_algo + ":", "No solution found") 
    

def tree_bfs(problem):
    frontier = deque([Node(problem.initial)])
    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        for child_node in node.expand(problem):
            frontier.append(child_node)
        # Alternative for the expansion of the extracted node
        # frontier.extend(node.expand(problem))


# Graph Breadth First Search
def graph_bfs(problem):
    frontier = deque([Node(problem.initial)])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        if problem.goal_test(node.state):
            return node
        for child_node in node.expand(problem):
            if child_node.state not in explored and child_node not in frontier:
                frontier.append(child_node)


# Tree Depth First Search
def tree_dfs(problem):
    frontier = deque([Node(problem.initial)])
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        for child_node in node.expand(problem):
            frontier.append(child_node)
        # Alternative for the expansion of the extracted node
        # frontier.extend(node.expand(problem))


# Graph Depth First Search
def graph_dfs(problem):
    frontier = deque([Node(problem.initial)])
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node.state)
        if problem.goal_test(node.state):
            return node
        for child_node in node.expand(problem):
            if child_node.state not in explored and child_node not in frontier:
                frontier.append(child_node)


# Uniform Cost Search
def ucs(problem, f):
    if problem.goal_test(problem.initial):
        return Node(problem.initial)
    f = memoize(f,'f')
    frontier = PriorityQueue('min',f)
    frontier.append(Node(problem.initial))
    explored = set()
    while frontier:
        node = frontier.pop()
        # print(node, f(node))
        if problem.goal_test(node.state):
            return node 
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                next_node = frontier.get_item(child)
                if f(child) < f(next_node):
                    del frontier[next_node]
                    frontier.append(child)
     


# Depth Limited Search
def dls(problem, limit = 20):
    
    # Recursive routine of DLS
    def recursive_dls(problem, node, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        cutoff_occurred = False
        for child_node in node.expand(problem):
            result = recursive_dls(problem, child_node, limit-1)
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        return 'cutoff' if cutoff_occurred else None

    return recursive_dls(problem, Node(problem.initial), limit)


# Iterative Deepening Search
def ids(problem):
    for depth in range(sys.maxsize):
        result = dls(problem, depth)
        if result != 'cutoff':
            return result
    return None


def a_star(problem : BlocksWorld, h=None) -> Node:
    h = memoize(h or problem.h)
    return ucs(problem, lambda n: n.depth + h(n))


def greedy_best_first(problem : BlocksWorld, h=None) -> Node:
    h = memoize(h or problem.h)
    return ucs(problem, lambda n: h(n))