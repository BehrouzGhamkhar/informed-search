from Search import *
import time


def run_a_star():
    t0 = time.time()
    p = Puzzle([0, 1, 2, 3, 4, 5, 8, 6, 7])
    path, expanded_nodes = astar_search(p, manhattan_distance)
    t1 = time.time()
    assert len(path) == 23
    # The execution time is limited to 10min, so please keep this in mind when implementing your solution!
    assert t1 - t0 <= 600


def run_greedy():
    # Run this test cell in order to get an INDICATION of whether your implementation is working as expected.
    import time
    t0 = time.time()
    p = Puzzle([0, 1, 2, 3, 4, 5, 8, 6, 7])
    path, expanded_nodes = greedy_search(p, manhattan_distance)
    t1 = time.time()
    # The execution time is limited to 10min, so please keep this in mind when implementing your solution!
    assert t1 - t0 <= 600


if __name__ == "__main__":
    run_a_star()
