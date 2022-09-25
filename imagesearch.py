from collections import Counter
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


def pipeline(r, data, chunk_size):
    npartitions = len(data) // chunk_size
    pipe = r.pipeline()

    for chunk in np.array_split(data, npartitions or 1):
        yield pipe, chunk


def insert(chunks, filename):
    n = 0

    for pipe, keys in chunks:
        for key in keys:
            pipe.sadd(key, filename)

        n += sum(pipe.execute())

    print(f"added {n} fragments for {filename}")


def lookup(chunks, filename):
    count = Counter()

    for pipe, keys in chunks:
        for key in keys:
            pipe.smembers(key)

        for result in pipe.execute():
            count.update(result)

    print(f"matches for {filename}:")

    for key, num in count.most_common():
        print(f'{num:<10d} {key.decode("utf-8")}')


def search(command: str):
    if len(sys.argv) < 3:
        print(__doc__)
        exit(1)

    command, *filenames = sys.argv[1:]
    command = insert if command == "insert" else lookup

    r = redis.StrictRedis(host="localhost", port=6379, db=0)
    try:
        r.ping
    except redis.ConnectionError:
        print("You need to install redis.")
        return

    for filename in filenames:
        print("loading", filename)
        img = cv2.imread(filename)
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        keypoints = [
            (round(k.pt[0]), round(k.pt[1]))
            for k in affine_detect(detector, img, pool=pool)[0]
        ]

        # Adapted from https://stackoverflow.com/a/42046567
        a = np.array(keypoints)
        a = a[np.lexsort(a[:, ::-1].T)]
        a = np.split(a, np.unique(a[:, 0], return_index=True)[1][1:], axis=0)

        i = 0
        points = np.reshape(np.asarray(a[0][0]), (-1, 2))

        while not i == len(a) - 1:
            if len(a[i + 1]) > 1:
                print(i)
                # Following line calculates the euclidean distance between current point and the points in the next group
                min_dist_point_addr = np.argmin(
                    np.linalg.norm(points[i] - a[i + 1], axis=1)
                )

                # Next group is reassigned with the element to whom the distance is the least
                a[i + 1] = a[i + 1][min_dist_point_addr]

            # The element is concatenated to points
            points = np.concatenate((points, np.reshape((a[i + 1]), (1, 2))), axis=0)
            i += 1

        triangles = triangles_from_keypoints(points, lower=50, upper=400)
        hashes = phash_triangles(img, triangles)
        chunks = pipeline(r, hashes, chunk_size=1e5)

        print()
        command(chunks, filename)
