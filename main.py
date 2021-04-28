from PIL import Image, ImageTk
from load_state import prepare_image
from utils import draw_state
from blocks_world import BlocksWorld
from search_algs import *
import argparse
from inspect import getfullargspec

if __name__ == "__main__":

    search_algs = {
        "a_star": a_star,
        "ucs": ucs,
        "graph_bfs": graph_bfs,
        "graph_dfs": graph_dfs,
        "greedy_best_first": greedy_best_first
    }

    parser = argparse.ArgumentParser(description="Blocks World")
    parser.add_argument("--initial", "-i", type=str, default=None, required=True)
    parser.add_argument("--goal", "-g", type=str, default=None, required=True)
    parser.add_argument("--algorithm", "-a", type=str, default=None, required=True)
    args = vars(parser.parse_args())

    initial_state_path = args["initial"]
    goal_state_path = args["goal"]
    search_alg = args["algorithm"]

    initial_state = prepare_image(initial_state_path)
    goal_state = prepare_image(goal_state_path)
    print(initial_state)
    print(goal_state)
    
    problem = BlocksWorld(initial_state, goal_state)
    if (len(getfullargspec(search_algs[search_alg]).args) == 2):
        problem.solution(search_algs[search_alg](problem, problem.misplaced_blocks).solution())
    else:
        problem.solution(search_algs[search_alg](problem).solution())



    