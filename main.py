from PIL import Image, ImageTk
from load_state import prepare_image
from utils import draw_state
from blocks_world import BlocksWorld
from search_algs import *
import argparse
from inspect import getfullargspec

# file che definisce lo script da linea di comando per utilizzare il programma


if __name__ == "__main__":

    search_algs = {
        "astar": a_star,
        "ucs": ucs,
        "rbfs": rbfs,
        "bfs": graph_bfs,
        "dfs": graph_dfs,
        "ids": ids
    }

    parser = argparse.ArgumentParser(description="Blocks World")
    parser.add_argument("--initial", "-i", type=str, default=None, required=True, help="The image representing the initial state")
    parser.add_argument("--goal", "-g", type=str, default=None, required=True, help="The image representing the goal state")
    parser.add_argument("--algorithm", "-a", type=str, default=None, required=True, help="The search algorithm used")
    parser.add_argument("--debug", "-d", default=False, required=False, action='store_true', help="Shows the steps of the image processing")
    parser.add_argument("--output", "-o", default=False, required=False, action='store_true', help="The solution is printed graphically")
    args = vars(parser.parse_args())

    initial_state_path = args["initial"]
    goal_state_path = args["goal"]
    search_alg = args["algorithm"]
    debug = args["debug"]
    output = args["output"]

    initial_state = prepare_image(initial_state_path, debug)
    goal_state = prepare_image(goal_state_path, debug)
    print(initial_state)
    print(goal_state)

    functions = {
        "ucs": lambda n: problem.depth(n),
        "astar": lambda n: problem.misplaced_blocks(n),
        "rbfs": lambda n: problem.misplaced_blocks(n)
    }

    problem = BlocksWorld(initial_state, goal_state)

    if len(getfullargspec(search_algs[search_alg]).args) == 2:
        problem.solution(search_algs[search_alg](problem, functions[search_alg]).solution(), output)
    else:
        problem.solution(search_algs[search_alg](problem).solution(), output)