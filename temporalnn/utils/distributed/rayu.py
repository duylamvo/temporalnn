"""Module to support distributed using ray"""
import ray

ray.init()


@ray.remote
def f():
    return 1


@ray.remote
def g():
    # Call f 4 times and return the resulting object IDs.
    return [f.remote() for _ in range(4)]


@ray.remote
def h():
    # Call f 4 times, block until those 4 tasks finish,
    # retrieve the results, and return the values.
    return ray.get([f.remote() for _ in range(4)])

ray.get(ray.get(g.remote()))