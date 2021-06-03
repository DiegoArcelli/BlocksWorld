from aima3.search import *
from utils import *
from collections import deque
from blocks_world import BlocksWorld
import sys

# file che contiene le implementazioni degli algoritmi di ricerca



node_expanded = 0 # numero di nodi espansi durante la ricerca
max_node = 0 # massimo numero di nodi presenti nella frontiera durante la ricerca
f_dim = 0 # dimensione della frontiera in un dato momento 
total_node = 0 


def init_param():
    global node_expanded, total_node, max_node, f_dim
    node_expanded = 0
    max_node = 0
    total_node = 0
    f_dim = 0


def print_param():
    print(f"Nodi espansi: {node_expanded}")
    print(f"Max dimensione della frontiera: {max_node}")
    print(f"Dim media della frontiera: {int(total_node/node_expanded)}")


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


# Graph Breadth First Search
def graph_bfs(problem):
    global node_expanded, total_node, max_node, f_dim
    init_param()
    frontier = deque([Node(problem.initial)])
    f_dim += 1
    explored = set()
    while frontier:
        node_expanded += 1
        total_node += f_dim
        node = frontier.popleft()
        f_dim -= 1
        explored.add(node.state)
        if problem.goal_test(node.state):
            # print(node_expanded)
            print_param()
            return node
        for child_node in node.expand(problem):
            if child_node.state not in explored and child_node not in frontier:
                f_dim += 1
                max_node = f_dim if f_dim > max_node else max_node
                frontier.append(child_node)


# Graph Depth First Search
def graph_dfs(problem):
    global node_expanded, total_node, max_node, f_dim
    init_param()
    frontier = deque([Node(problem.initial)])
    f_dim += 1
    explored = set()
    while frontier:
        total_node += f_dim
        node = frontier.pop()
        node_expanded += 1
        f_dim -= 1
        if problem.goal_test(node.state):
            print_param()
            return node
        explored.add(node.state)
        for child_node in node.expand(problem):
            if child_node.state not in explored and child_node not in frontier:
                f_dim += 1
                max_node = f_dim if f_dim > max_node else max_node
                frontier.append(child_node)


# Uniform Cost Search
def ucs(problem, f):
    global node_expanded, total_node, max_node, f_dim
    init_param()
    if problem.goal_test(problem.initial):
        return Node(problem.initial)
    f = memoize(f, 'f')
    node_expanded += 1
    frontier = PriorityQueue('min', f)
    frontier.append(Node(problem.initial))
    f_dim += 1
    explored = set()
    while frontier:
        total_node += f_dim
        node_expanded += 1
        node = frontier.pop()
        f_dim -= 1
        # print(node, f(node))
        if problem.goal_test(node.state):
            print_param()
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                f_dim += 1
                frontier.append(child)
                max_node = f_dim if f_dim > max_node else max_node
            elif child in frontier:
                next_node = frontier.get_item(child)
                if f(child) < f(next_node):
                    del frontier[next_node]
                    frontier.append(child)


# Depth Limited Search
def dls(problem, limit):

    def recursive_dls(problem, node, limit):
        global node_expanded, total_node, max_node, f_dim
        node_expanded += 1
        total_node += f_dim
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        cutoff_occurred = False
        for child_node in node.expand(problem):
            f_dim+=1
            max_node = f_dim if f_dim > max_node else max_node
            result = recursive_dls(problem, child_node, limit-1)
            f_dim -= 1
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        return 'cutoff' if cutoff_occurred else None

    return recursive_dls(problem, Node(problem.initial), limit)


# Iterative Deepening Search
def ids(problem):
    global node_expanded, total_node, max_node, f_dim
    init_param()
    prevexp = 0
    for depth in range(sys.maxsize):
        f_dim += 1
        result = dls(problem, depth)
        print(node_expanded - prevexp)
        prevexp = node_expanded
        f_dim = 0
        if result != 'cutoff':
            print_param()
            return result
    return None

# A*
def a_star(problem: BlocksWorld, h=None):
    global node_expanded
    h = memoize(h or problem.h)
    return ucs(problem, lambda n: problem.depth(n) + h(n))


# Recursive Best First Search
def rbfs(problem, h):
    global node_expanded, total_node, max_node, f_dim
    init_param()

    h = memoize(h or problem.h, 'h')
    g = memoize(lambda n: problem.depth(n), 'g')
    f = memoize(lambda n: g(n) + h(n), 'f')
    

    def rbfs_search(problem, node, f_limit=np.inf):
        global node_expanded, total_node, max_node, f_dim

        node_expanded += 1
        if problem.goal_test(node.state):
            print_param()
            return node, 0

        successors = [*node.expand(problem)]
        f_dim += len(successors)
        total_node += f_dim
        max_node = f_dim if f_dim > max_node else max_node

        if len(successors) == 0:
            return None, np.inf

        for child in successors:
            child.f = max(f(child), node.f)

        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]

            if best.f > f_limit:
                f_dim -= len(successors)
                return None, best.f

            alt = successors[1].f if len(successors) > 1 else np.inf
            # importante, sovrascrivere best.f
            result, best.f = rbfs_search(problem, best, min(f_limit, alt))
            # return result
            if result is not None:
                f_dim -= len(successors)
                return result, best.f

    node = Node(problem.initial)
    f(node)
    f_dim += 1
    return rbfs_search(problem, node)[0]