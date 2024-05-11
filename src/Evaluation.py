from Search import *
import time
import random


def run_evaluation(puzzle_configs, heuristic, algorithm):

    records = []
    for i in puzzle_configs.keys():
        t0 = time.time()
        p = Puzzle(puzzle_configs[i])
        path, expanded_nodes = algorithm(p, heuristic)
        t1 = time.time()

        records.append({
            'puzzle': i,
            'algorithm': algorithm.__name__,
            'heuristic': heuristic.__name__,
            'expanded nodes': expanded_nodes,
            'execution time': t1 - t0,
            'path cost': len(path),
            'complexity': inversion(puzzle_configs[i])})
    return records


def evaluate(puzzle_configs):
    heuristics = [manhattan_distance, hamming_distance]
    search_algorithms = [astar_search, greedy_search]
    evaluation_results = []

    for heuristic in heuristics:
        for algorithm in search_algorithms:
            eval_records = run_evaluation(puzzle_configs, heuristic, algorithm)
            for record in eval_records:
                evaluation_results.append(record)

    for record in evaluation_results:
        print(record)


def inversion(configuration: List[int]) -> int:
    """
    Finds all inversions in a puzzle configuration and returns their total number
    :param configuration: the configuration of the 8 puzzle to count its inversions
    :returns: number of inversions in the puzzle configuration
    """
    # YOUR CODE HERE
    inversion_count = 0
    for i in range(len(configuration)):
        for j in range(i + 1, len(configuration)):
            if configuration[i] != 0 and configuration[j] != 0 and configuration[i] > configuration[j]:
                inversion_count += 1
    return inversion_count


def is_solvable(configuration: List[int]) -> bool:
    """
    Checks whether a given puzzle configuration is solvable or not
    :param configuration: the initial configuration of the puzzle to check for solvability
    :returns: True if the configuration is solvable; False otherwise
    """
    # YOUR CODE HERE
    inversion_count = inversion(configuration)

    return inversion_count % 2 == 0


def generate_puzzle():
    """
    Generate a random solvable puzzle configuration.

    Returns:
    :puzzle: A list representing the puzzle configuration
    """
    puzzle = list(range(9))
    random.shuffle(puzzle)
    while not is_solvable(puzzle):
        random.shuffle(puzzle)
    return puzzle


def generate_solvable_puzzles(total):
    return [generate_puzzle() for i in range(total)]


if __name__ == "__main__":
    # puzzle_configs = {1: [0, 1, 2, 3, 4, 5, 8, 6, 7],
    #                   2: [8, 7, 6, 5, 1, 4, 2, 0, 3],
    #                   3: [1, 5, 7, 3, 6, 2, 0, 4, 8]}

    puzzle_configs = generate_solvable_puzzles(5)
    puzzle_dict = {}

    for index, config in enumerate(puzzle_configs):
        puzzle_dict[index] = config

    evaluate(puzzle_dict)
