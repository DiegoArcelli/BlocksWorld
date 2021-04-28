import heapq
import functools
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class PriorityQueue:
    """A Queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first.
    If order is 'min', the item with minimum f(x) is
    returned first; if order is 'max', then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':  # now item with max f(x)
            self.f = lambda x: -f(x)  # will be popped first
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item at its correct position."""
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        """Insert each item in items at its correct position."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item (with min or max f(x) value)
        depending on the order."""
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    def __len__(self):
        """Return current capacity of PriorityQueue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if the key is in PriorityQueue."""
        return any([item == key for _, item in self.heap])

    def __getitem__(self, key):
        """Returns the first value associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    def get_item(self, key):
        """Returns the first node associated with key in PriorityQueue.
        Raises KeyError if key is not present."""
        for _, item in self.heap:
            if item == key:
                return item
        raise KeyError(str(key) + " is not in the priority queue")


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, use lru_cache for caching the values."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn


def draw_state(state, file_path):
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
    padded = np.zeros((size, w*100), np.uint8)
    padded[size - (h+1)*100 : size, :] = image
    h =  len(state) - 1
    bg = np.zeros((h*100 + 40, w*100 + 40), np.uint8)
    bg[20: h*100 + 20, 20: w*100 + 20] = padded
    bg[0:10, :] = 255
    bg[h*100 + 30 : h*100 + 40, :] = 255
    bg[:, 0:10] = 255
    bg[h*100 + 30 : h*100 + 40, :] = 255
    bg[:,w*100 + 30 : w*100 + 40] = 255
    w, h = (w*100 + 40, h*100 + 40)
    l = max(w, h)
    adjust = np.zeros((l, l), np.uint8)
    d_w = (l - w) // 2
    d_h = (l - h) // 2
    adjust[d_h: d_h + h, d_w: d_w + w] = bg
    cv.imwrite("./images/temp/" + str(file_path) + ".jpg", ~adjust)