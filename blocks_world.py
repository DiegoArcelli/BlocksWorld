from aima3.search import *
from utils import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class BlocksWorld(Problem):

    def __init__(self, initial, goal):
        super().__init__(initial, goal)


    def get_blocks_number(self):
        return len(self.initial)


    def actions(self, state):
        blocks = [*state[0:-1]]
        size = state[-1]
        columns = {}
        tops = []
        for block in blocks:
            n, i, j = block
            if j not in columns:
                columns[j] = (n, i, j)
            else:
                if i > columns[j][1]:
                    columns[j] = (n, i, j)
        for col in columns:
            tops.append(columns[col])
        actions = []
        for block in tops:
            n, i, j = block
            for col in range(size):
                if col != j:
                    if col in columns:
                        actions.append((n, columns[col][1]+1, col))
                    else:
                        actions.append((n, 0, col))
        return actions


    def result(self, state, actions):
        blocks = [*state[0:-1]]
        size = state[-1]
        to_delete = ()
        for block in blocks:
            if block[0] == actions[0]:
                to_delete = block
        blocks.remove(to_delete)
        blocks.append((actions))
        blocks.append(size)
        return tuple(blocks)


    def goal_test(self, state):
        op_1 = [*state[0:-1]]
        op_2 = [*self.goal[0:-1]]
        op_1.sort(key=lambda l: l[0])
        op_2.sort(key=lambda l: l[0])
        return str(op_1) == str(op_2)


    def get_movable(self, state):
        blocks = [*state[0:-1]]
        size = state[-1]
        columns = {}
        tops = []
        for block in blocks:
            n, i, j = block
            if j not in columns:
                columns[j] = (n, i, j)
            else:
                if i > columns[j][1]:
                    columns[j] = (n, i, j)
        for col in columns:
            tops.append(columns[col])
        return tops


    def misplaced_blocks(self, node):
        blocks = [*node.state[0:-1]]
        target = [*self.goal[0:-1]]
        target.sort(key=lambda l: l[0])
        value = 0
        for block in blocks:
            n, i, j = block
            if target[n-1][1:3] != (i, j):
                value += 1
                # if block not in self.get_movable(node.state):
                #     value += 1
        return value


    def solution(self, actions):
        if len(actions) is None:
            return
        state = self.initial
        successor = None
        for action in actions:
            successor = self.result(state, action)
            figue_1 = self.draw_state(state)
            figue_2 = self.draw_state(successor)
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(figue_1, cmap=plt.cm.binary)
            axarr[1].imshow(figue_2, cmap=plt.cm.binary)
            figManager = plt.get_current_fig_manager()
            figManager.full_screen_toggle()
            plt.show()
            state = successor


    def draw_state(self, state):
        blocks = [*state[0:-1]]
        w = state[-1]
        blocks.sort(key=lambda l: l[1], reverse=True)
        h = blocks[0][1]
        image = np.zeros(((h+1)*100, w*100), np.uint8)
        for block in blocks:
            n, i, j = block
            i = h - i
            digit = cv.imread("./images/digits/" + str(n) + ".jpg", 0)
            digit = cv.resize(digit, (100, 100))
            image[i*100:i*100 + 100, j*100:j*100 + 100] = ~digit
        size = (len(state) - 1)*100
        adjust = np.zeros((size, w*100), np.uint8)
        adjust[size - (h+1)*100 : size, :] = image
        return adjust
