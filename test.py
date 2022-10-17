from collections import Counter
import math
from os import cpu_count
from sqlitedict import SqliteDict
from asift import affine_detect
from find_obj import init_feature
from multiprocessing.pool import ThreadPool
from phash import triangles_from_keypoints, hash_triangles

import cv2
import redis
import numpy as np
import sys
import multiprocessing
import time

start_time = time.time()
feature_name = "orb"
detector, matcher = init_feature(feature_name)


def phash_triangles(img, triangles, batch_size=None):
    n = len(triangles)

    if batch_size is None:
        batch_size = n // cpu_count()

    array = np.asarray(triangles, dtype="d")
    tasks = [(img, array[i : i + batch_size]) for i in range(0, n, batch_size)]
    results = []

    with multiprocessing.Pool(processes=cpu_count()) as p:
        for result in p.starmap(hash_triangles, tasks):
            results += result

    return results


filename = (
    "transformationInvariantImageSearch/fullEndToEndDemo/inputImages/cat_original.png"
)
img = cv2.imread(filename)
im_h, im_w = img.shape[:2]
img_area = im_h * im_w
pool = ThreadPool(processes=cv2.getNumberOfCPUs())
keypoints = [
    (round(k.pt[0]), round(k.pt[1])) for k in affine_detect(detector, img, pool=pool)[0]
]
# Adapted from https://stackoverflow.com/a/42046567
a = np.array(keypoints)
a = a[np.lexsort(a[:, ::-1].T)]
a = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:], axis=0)

i = 0
points = np.reshape(np.asarray(a[0][0]), (-1, 2))

while not i == len(a) - 1:
    if len(a[i + 1]) > 1:
        # Following line calculates the euclidean distance between current point and the points in the next group
        min_dist_point_addr = np.argmin(np.linalg.norm(points[i] - a[i + 1], axis=1))
        # Next group is reassigned with the element to whom the distance is the least
        a[i + 1] = a[i + 1][min_dist_point_addr]
    # The element is concatenated to points
    points = np.concatenate((points, np.reshape((a[i + 1]), (1, 2))), axis=0)
    i += 1

triangles = triangles_from_keypoints(points, lower=50, upper=400)
b = [[t[0].tolist(), t[1].tolist(), t[2].tolist()] for t in triangles]
c = []
d = {}
for t in b:
    if str(t[0]) not in c:
        c.append(str(t[0]))
        print(str(t[0]), len(c))
for _c in c:
    if str(_c) not in d:
        d[str(_c)] = [t for t in b if str(t[0]) == str(_c)]
        print(d[str(_c)], len(list(d)))

_d = []
for i, g in enumerate(dict.values(d)):
    _gh = g[0]  # Group head
    # Area(ΔABC) = (1/2)|x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    head_area = round(
        (
            abs(
                round(
                    _gh[0][0] * (_gh[1][1] - _gh[2][1])
                    + _gh[1][0] * (_gh[2][1] - _gh[0][1])
                    + _gh[2][0] * (_gh[0][1] - _gh[1][1])
                )
            )
        )
        / 2
    )
    print(i, _gh)
    for t in g:
        t_area = round(
            (
                abs(
                    round(
                        t[0][0] * (t[1][1] - t[2][1])
                        + t[1][0] * (t[2][1] - t[0][1])
                        + t[2][0] * (t[0][1] - t[1][1])
                    )
                )
            )
            / 2
        )
        if t_area < head_area and t_area > math.ceil(img_area / 5):
            # Change the head of group if smaller
            # This way, it's unlikely for it to be removed
            print(i, _gh)
            _gh = t
            head_area = t_area
    print(i, _gh)
    _d.append(_gh)

hashes = hash_triangles(img, _d)

print("tris: ", triangles[0:10], len(triangles))
print(f"processed: {len(_d)}")
print(f"preview hash [{len(hashes)}]: {hashes[0:5]}")
print("--- %s seconds ---" % (time.time() - start_time))
