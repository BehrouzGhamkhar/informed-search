from Search import *
import time


def run_a_star(puzzle_configs, heuristic):
    records = []
    for i in puzzle_configs.keys():
        t0 = time.time()
        p = Puzzle(puzzle_configs[i])
        path, expanded_nodes = astar_search(p, heuristic)
        t1 = time.time()

        records.append({
            'puzzle': i,
            'algorithm': 'astar',
            'heuristic': heuristic.__name__,
            'expanded nodes': expanded_nodes,
            'execution time': t1 - t0,
            'path cost': len(path)})
    return records


def run_greedy(puzzle_configs, heuristic):
    # Run this test cell in order to get an INDICATION of whether your implementation is working as expected.
    records = []
    for i in puzzle_configs.keys():
        t0 = time.time()
        p = Puzzle(puzzle_configs[i])
        path, expanded_nodes = greedy_search(p, heuristic)
        t1 = time.time()

        records.append({
            'puzzle': i,
            'algorithm': 'greedy',
            'heuristic': heuristic.__name__,
            'expanded nodes': expanded_nodes,
            'execution time': t1 - t0,
            'path cost': len(path)})
    return records


def evaluate():
    puzzle_configs = {1: [0, 1, 2, 3, 4, 5, 8, 6, 7],
                      2: [8, 7, 6, 5, 1, 4, 2, 0, 3],
                      3: [1, 5, 7, 3, 6, 2, 0, 4, 8]}

    evaluation_results = [*run_a_star(puzzle_configs, manhattan_distance),
                          *run_a_star(puzzle_configs, hamming_distance),
                          *run_greedy(puzzle_configs, manhattan_distance),
                          *run_greedy(puzzle_configs, hamming_distance)]

    for record in evaluation_results:
        print(record)


if __name__ == "__main__":
    evaluate()
