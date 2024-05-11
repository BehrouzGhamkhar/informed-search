from Puzzle import Puzzle
from typing import Callable, List, Tuple


# Heuristics
def hamming_distance(state: List[int]) -> int:
    """Calculate the Hamming distance for a particular configuration
    :param state: current configuration of the puzzle
    :return: the number of misplaced tiles in the given configuration
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def manhattan_distance(state: List[int]) -> int:
    """Function to calculate the manhattan distance for a
    particular configuration
    :param state: current configuration of the puzzle
    :return: the accumulated manhattan distance between each tile and its goal position in the given configuration
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def test_heuristics():
    # Run this test cell in order to get an indication of whether your implementation is working as expected.
    assert hamming_distance([0, 1, 2, 3, 4, 5, 8, 6, 7]) == 9
    assert manhattan_distance([1, 2, 3, 4, 0, 5, 6, 8, 7]) == 8


def astar_search(board: Puzzle, heuristic: Callable) -> Tuple[List[int], int]:
    """
    :param board: the 8-puzzle to solve
    :param heuristic: the heuristic function to use
    :return: an ordered list with the solution path and the number of total nodes expanded
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def greedy_search(board: Puzzle, heuristic: Callable) -> Tuple[List[int], int]:
    """Implementation of the greedy search algorithm.
    :param board: the 8-puzzle to solve
    :param heuristic: the heuristic function to use
    :return: an ordered list with the solution path and the number of total nodes expanded
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# YOUR CODE HERE
raise NotImplementedError()
