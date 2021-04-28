import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow import keras
from math import ceil

BLOCK_SIZE = 50
THRESHOLD = 40
deteced = [np.array([]) for x in range(6)]
poisitions = [None for x in range(6)]
explored = []
model = keras.models.load_model("./model/model.h5")


def find_area(img_pos, org):
    temp = org.copy()
    x, y, w, h = img_pos
    temp[y:y+h, x:x+w] = 0



def get_black_pixel(img):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255:
                return (i, j)
    return False


def get_adjacents(img, i, j):
    h, w = img.shape
    actions = []
    if img[i, j] == 255:
        for k in range(i-1, i+2):
            for l in range(j-1, j+2):
                if k >= 0 and l >= 0 and k < h and l < w:
                    if img[k, l] == 255:
                        actions.append((k, l))
    return actions


def check_continuity(img):
    image = img.copy()
    image = ~image
    res = get_black_pixel(image)
    if type(res) == bool:
        return False
    i, j = res
    frontier = [(i, j)]
    explored = set()
    while frontier:
        i, j = frontier.pop()
        explored.add((i, j))
        for action in get_adjacents(image, i, j):
            k, l = action
            if (k, l) not in frontier and (k, l) not in explored:
                frontier.append((k, l))
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            if image[i, j] == 255 and (i, j) not in explored:
                return False
    return True


def predict(image, previous):
    img = image.copy()
    img = ~img
    h, w = image.shape
    l = max(h, w)
    size = (int(l + l/3), int(l + l/3))
    mask = np.zeros(size, np.int8)
    x = int((size[0]-h)/2)
    y = int((size[1]-w)/2)
    mask[x:x+h, y:y+w] = img.copy()
    mask = mask.astype('float32')
    mask = cv.resize(mask, (28, 28))
    for i in range(28):
        for j in range(28):
            if mask[i, j] != 0:
                mask[i, j] = 255
    _in = np.expand_dims(mask, -1)
    _in = _in / 255
    _list = [_in]
    _list = np.array(_list)
    digit = np.argmax(model.predict(_list[:1])) - 1
    # print(digit)
    # show(mask)
    if deteced[digit].size == 0:
        # print(digit)
        # show(mask)
        deteced[digit] = previous.copy()


def show(img):
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.imshow(img)
    plt.show()


def preprocess(image):
    image = cv.medianBlur(image, 3)
    image = cv.GaussianBlur(image, (3, 3), 0)
    return 255 - image


def postprocess(image):
    image = cv.medianBlur(image, 5)
    image = cv.medianBlur(image, 5)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    image = cv.erode(image, kernel, iterations=2)
    return image


def get_block_index(image_shape, yx, block_size):
    y = np.arange(max(0, yx[0]-block_size),
                  min(image_shape[0], yx[0]+block_size))
    x = np.arange(max(0, yx[1]-block_size),
                  min(image_shape[1], yx[1]+block_size))
    return np.meshgrid(y, x)


def adaptive_median_threshold(img_in):
    med = np.median(img_in)
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < THRESHOLD] = 255
    return img_out


def block_image_process(image, block_size):
    out_image = np.zeros_like(image)
    for row in range(0, image.shape[0], block_size):
        for col in range(0, image.shape[1], block_size):
            idx = (row, col)
            block_idx = get_block_index(image.shape, idx, block_size)
            out_image[block_idx] = adaptive_median_threshold(image[block_idx])
    return out_image


def clean(image):
    contours, hierarchy = cv.findContours(
        image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv.approxPolyDP(
            contour, 0.001 * cv.arcLength(contour, True), True)
        x, y, w, h = cv.boundingRect(approx)
        if search_noise(contour, approx, image.shape[::-1]):
            cv.drawContours(image, [approx], 0, 255, -1)
    return image


def search_noise(contour, approx, image_size):
    i_h, i_w = image_size
    x, y, w, h = cv.boundingRect(approx)
    image_area = i_w*i_h
    if cv.contourArea(contour) >= image_area/1000:
        return False
    if w >= i_w/50 or h >= i_h/50:
        return False
    return True


def compare_images(img, images):
    for image in images:
        if np.array_equal(image, img):
            return True
    return False


def find_digits(image, previous, org, depth):
    if compare_images(image, explored):
        return
    # show(image)
    explored.append(image)
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    print(len(contours))
    valid_contour = 0
    if len(contours) <= 3 and check_continuity(image):
        # print(depth)
        # show(image)
        predict(image, previous)
    mask = np.zeros(image.shape, np.uint8)
    count = 0
    contours.sort(reverse=True, key=lambda c: cv.contourArea(c))
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.001 * cv.arcLength(contour, True), True)
        x, y, w, h = cv.boundingRect(approx)
        if validate_figure(contour, approx, image.shape[::-1], org.shape[::-1]):
            find_digits(image[y:y+h, x:x+w], image.copy(), org, depth+1)


def find_box(image, org):
    oh, ow = org.shape[0:2]
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours.sort(reverse=True, key=lambda c: cv.contourArea(c))
    tot = np.zeros(shape=(oh, ow))
    for contour in contours:
        temp = np.zeros(shape=(oh, ow))
        approx = cv.approxPolyDP(contour, 0.001 * cv.arcLength(contour, True), True)
        x, y, w, h = cv.boundingRect(approx)
        cv.drawContours(temp, [approx], 0, 255, 2)
        cv.drawContours(tot, [approx], 0, 255, 2)
        # show(temp)
    # contours.sort(reverse=True, key=lambda c: cv.contourArea(c))
    # show(tot)
    contour = contours[1]
    approx = cv.approxPolyDP(contour, 0.001 * cv.arcLength(contour, True), True)
    x, y, w, h = cv.boundingRect(approx)
    # show(image[y:y+h, x:x+w])
    # print((x, y, x + w, y + h))
    # return image[y:y+h, x:x+w].copy()
    return (x, y, x + w, y + h)
    

def validate_figure(contour, approx, image_size, org_size):
    x, y, w, h = cv.boundingRect(approx)
    o_w, o_h = org_size
    i_w, i_h = image_size
    image_area = o_h*o_w
    if i_w == w or i_h == h:
        return False
    if cv.contourArea(contour) >= image_area/1000:
        return True
    if w >= o_w/100 or h >= o_h/100:
        return True
    return False


def process_image_file(filename):
    global deteced, poisitions, explored
    deteced = [np.array([]) for x in range(6)]
    poisitions = [None for x in range(6)]
    explored = []
    image_in = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2GRAY)
    # show(image_in)
    # show(image_in)
    image_in = preprocess(image_in)
    image_out = block_image_process(image_in, BLOCK_SIZE)
    image_out = postprocess(image_out)
    # show(image_out)
    image_out = clean(image_out)
    # show(image_out)
    box = find_box(image_out, image_out.copy())
    x, y, w, h = box
    find_digits(image_out[y:h, x:w], None, image_out.copy(), 0)
    # for i in range(6):
    #     if deteced[i].size > 0:
    #         # print(i)
    #         # show(deteced[i])
    for i in range(6):
        if deteced[i].size > 0:
            image = deteced[i]
            w, h = image.shape[::-1]
            res = cv.matchTemplate(image_out, image, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            x, y = max_loc
            bottom_right = (x + w, y + h)
            cv.rectangle(image_in, (x, y), bottom_right, 255, 2)
            show(image_in)
            poisitions[i] = (x, y, x + w, y + h)
    return box


def check_intersection(values):
    v1_i, v1_f, v2_i, v2_f = values
    if v1_i <= v2_f and v1_i >= v2_i:
        return True
    if v1_f <= v2_f and v1_f >= v2_i:
        return True
    if v2_i <= v1_f and v2_i >= v1_i:
        return True
    if v2_f <= v1_f and v2_f >= v1_i:
        return True
    return False


def create_state(poisitions, box):
    cols = [[] for x in range(6)]
    mean_points = []
    for i in range(6):
        if poisitions[i] is not None:
            x1_i, y1_i, x1_f, y1_f = poisitions[i]
            mean_points.append(((x1_f + x1_i) // 2, ((y1_f + y1_i) // 2)))
            c = [i+1]
            for j in range(6):
                if poisitions[j] is not None and j != i:
                    x2_i, y2_i, x2_f, y2_f = poisitions[j]
                    if check_intersection((x1_i, x1_f, x2_i, x2_f)):
                        c.append(j+1)
            c.sort()
            cols[i] = tuple([*c])
        else:
            cols[i] = ()

    temp_cols = list(set(tuple(cols)))
    if () in temp_cols:
        temp_cols.remove(())

    cols = []

    for t_col in temp_cols:
        col = list(t_col)
        col.sort(reverse=True, key=lambda e: mean_points[e-1][1])
        cols.append(tuple(col))

    cols.sort(key=lambda e: mean_points[e[0]-1][0])


    bottoms = [col[0] for col in cols]

    distances = []

    xb_i, _, xb_f, _ = box

    x_i, _, x_f, _  = poisitions[bottoms[0]-1]
    dist = abs(x_i - xb_i)
    dist = dist / (x_f - x_i)
    distances.append(dist)

    for i in range(len(bottoms)-1):
        x1_i, _, x1_f, _  = poisitions[bottoms[i]-1]
        x2_i, _, _, _  = poisitions[bottoms[i+1]-1]
        dist = abs(x2_i - x1_f)
        dist = dist / (x1_f - x1_i)
        distances.append(dist)
        
    x_i, _, x_f, _  = poisitions[bottoms[-1]-1]
    dist = abs(xb_f - x_f)
    dist = dist / (x_f - x_i)
    distances.append(dist)

    for i in range(len(distances)):
        dist = distances[i]
        if dist - int(dist) >= 0.5:
            distances[i] = int(dist) + 1
        else:
            distances[i] = int(dist)

    n = sum(distances) + len(cols)
    i = distances[0]
    state = []
    pos = 1
    for col in cols:
        j = 0
        for block in col:
            state.append((block, j, i))
            j += 1
        i += distances[pos] + 1
        pos += 1
    # print(distances)
    state.append(n)
    return tuple(state)


def prepare_image(file_path):
    box = process_image_file(file_path)
    state = create_state(poisitions, box)
    print(state)
    return state
