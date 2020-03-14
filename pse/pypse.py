import numpy as np
import cv2
import queue


def pse(region, center,  min_area):
    pred = np.zeros(center.shape, dtype='int32')

    label_num, label = cv2.connectedComponents(center.astype(np.uint8), connectivity=4)

    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    queue_base = queue.Queue(maxsize=0)
    points = np.array(np.where(label > 0)).transpose((1, 0))

    for point_idx in range(points.shape[0]):
        x, y = points[point_idx, 0], points[point_idx, 1]
        l = label[x, y]
        queue_base.put((x, y, l))
        pred[x, y] = l

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    while not queue_base.empty():
        (x, y, l) = queue_base.get()

        for j in range(4):
            tmpx = x + dx[j]
            tmpy = y + dy[j]
            if tmpx < 0 or tmpx >= region.shape[0] or tmpy < 0 or tmpy >= region.shape[1]:
                continue
            if region[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:
                continue

            queue_base.put((tmpx, tmpy, l))
            pred[tmpx, tmpy] = l

    return pred, label_values