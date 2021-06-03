import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow import keras
from math import ceil

deteced = [np.array([]) for x in range(6)] # lista che contiene le immagini delle cifre
poisitions = [None for x in range(6)] # lista che contiene la posizione delle cifre nell'immagine
debug_mode = False
model = keras.models.load_model("./model/model.h5") # carica il modello allenato sul datase del MNIST

# funzione che si occupa del riconoscimento della cifra presente nell'immagine
# che gli viene passato come parametro
def predict(image):
    h, w = image.shape
    l = int(max(image.shape)*1.2)
    n_h = int((l - h)/2)
    n_w = int((l - w)/2)
    img = np.zeros((l, l), np.uint8)
    img[n_h : n_h + h, n_w : n_w + w] = image
    img = (img / 255).astype('float64')
    img = cv.resize(img, (28, 28), interpolation = cv.INTER_AREA)
    _in = np.array([img])
    _in = np.expand_dims(_in, -1)
    digit = np.argmax(model.predict(_in))
    if debug_mode:
        print(digit)
        show(img)
    return digit - 1 if digit > 0 else -1


# stampa a schermo l'immagine che gli veiene passata come parametro
def show(img):
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()


# prime modifiche all'immagine che consistono nell'applicazione di blur
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
    threshold = 40
    img_out = np.zeros_like(img_in)
    img_out[img_in - med < threshold] = 255
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


def find_digits(image, org_image, org):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i = 0
    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.001 * cv.arcLength(contour, True), True)
        x, y, w, h = cv.boundingRect(approx)
        if hierarchy[0][i][3] == -1:
            prev = predict(org_image[y:y+h, x:x+w])
            if prev != -1:
                deteced[prev] = org[y:y+h, x:x+w]
                poisitions[prev] = (x, y, x + w, y + h)  
        i += 1


# funzione che individua il box che contiene i blocchi ed individua le cifre
def find_box(image):
    o_h, o_w = image.shape[0:2]
    contours, hierarchy = cv.findContours(
        image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours.sort(reverse=True, key=lambda c: cv.contourArea(c))
    contour = contours[1]
    approx = cv.approxPolyDP(
        contour, 0.001 * cv.arcLength(contour, True), True)
    x, y, w, h = cv.boundingRect(approx)
    box = (x, y, x + w, y + h)
    img = image[y:y+h, x:x+w]
    sub = img.copy()
    bg = ~np.zeros((h + 50, w + 50), np.uint8)
    bg[25: 25 + h, 25: 25 + w] = img
    img = bg
    i = 0
    i_h, i_w = img.shape[0:2]
    tot = np.zeros(shape=(i_h, i_w))
    if debug_mode:
        print(image)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        approx = cv.approxPolyDP(
            contour, 0.001 * cv.arcLength(contour, True), True)
        if hierarchy[0][i][3] == 0:
            cv.drawContours(tot, [approx], 0, 255, -1)
        if hierarchy[0][i][3] == 1:
            cv.drawContours(tot, [approx], 0, 0, -1)
        i += 1
    tot = tot[25: 25 + h, 25: 25 + w]
    kernel = np.ones((5, 5), np.uint8)
    tot = cv.dilate(tot, kernel, iterations=3)
    tot = tot.astype('uint32')
    sub = sub.astype('uint32')
    res = sub + tot
    res = np.where(res == 0, 255, 0)
    result = np.zeros((o_h, o_w), np.uint8)
    result[y:y+h, x:x+w] = res
    if debug_mode:
        show(result)
    return (result, box)


def get_block_borders(dims, image):
    x_i, y_i, x_f, y_f = dims
    kernel = np.ones((5, 5), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    y_m = (y_f + y_i) // 2
    x_m = (x_f + x_i) // 2
    t = x_i - 1
    while image[y_m, t] != 255:
        t-=1
    x_i = t
    t = x_f + 1
    while image[y_m, t] != 255:
        t+=1
    x_f = t
    t = y_i - 1
    while image[t, x_m] != 255:
        t-=1
    y_i = t
    t = y_f + 1
    while image[t, x_m] != 255:
        t+=1
    y_f = t
    return (x_i, y_i, x_f, y_f)


def process_image_file(filename):
    global deteced, poisitions, explored, debug_mode
    block_size = 50
    deteced = [np.array([]) for x in range(6)]
    poisitions = [None for x in range(6)]
    explored = []
    image_in = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2GRAY)
    if debug_mode:
        show(image_in)
    image_in_pre = preprocess(image_in)
    image_out = block_image_process(image_in_pre, block_size)
    image_out = postprocess(image_out)
    image_out = clean(image_out)
    if debug_mode:
        show(image_out)
    digits, box = find_box(image_out)
    find_digits(digits, ~image_out, image_in)
    for i in range(6):
        if deteced[i].size > 0:
            image = deteced[i]
            x, y, w, h = get_block_borders(poisitions[i], ~image_out)
            poisitions[i] = (x, y, w, h)
            cv.rectangle(image_in, (x, y), (w, h), 255, 2)
    if debug_mode:
        show(image_in)
    return box

def check_intersection(values):
    v1_i, v1_f, v2_i, v2_f = values
    v2_m = (v2_i + v2_f) // 2
    if v1_i < v2_m and v1_f > v2_m:
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

    x_i, _, x_f, _ = poisitions[bottoms[0]-1]
    dist = abs(x_i - xb_i)
    dist = dist / (x_f - x_i)
    distances.append(dist)

    for i in range(len(bottoms)-1):
        x1_i, _, x1_f, _ = poisitions[bottoms[i]-1]
        x2_i, _, _, _ = poisitions[bottoms[i+1]-1]
        dist = abs(x2_i - x1_f)
        dist = dist / (x1_f - x1_i)
        distances.append(dist)

    x_i, _, x_f, _ = poisitions[bottoms[-1]-1]
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
    state.append(n)
    return tuple(state)


def prepare_image(file_path, debug):
    global debug_mode
    debug_mode = True if debug else False
    box = process_image_file(file_path)
    state = create_state(poisitions, box)
    return state
