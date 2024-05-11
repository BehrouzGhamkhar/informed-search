from Puzzle import *
from typing import Callable, List, Tuple


# Heuristics
def hamming_distance(state: List[int]) -> int:
    """Calculate the Hamming distance for a particular configuration
    :param state: current configuration of the puzzle
    :return: the number of misplaced tiles in the given configuration
    """
    # YOUR CODE HERE
    goal_state = [i for i in range(1, 9)] + [0]
    return len([x for (x, y) in zip(state, goal_state) if x != y])


def manhattan_distance(state: List[int]) -> int:
    """Function to calculate the manhattan distance for a
    particular configuration
    :param state: current configuration of the puzzle
    :return: the accumulated manhattan distance between each tile and its goal position in the given configuration
    """
    # YOUR CODE HERE
    goal_state = [i for i in range(1, 9)] + [0]
    distance = 0
    for i in state:
        goal_i = goal_state.index(i)
        state_i = state.index(i)
        goal_x, goal_y = goal_i // 3, goal_i % 3
        state_x, state_y = state_i // 3, state_i % 3

        distance += abs(goal_x - state_x) + abs(goal_y - state_y)

    return distance


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
    start_node = Node(board.init_state, 0, heuristic(board.init_state))
    board.fringe = [start_node]

    while board.fringe:
        current_node = heappop(board.fringe)
        current_state = current_node.state

        if board.goal_test(current_state):
            # Reconstruct the solution path
            solution_path = [current_state]
            while current_node.parent:
                current_node = current_node.parent
                solution_path.append(current_node.state)
            solution_path.reverse()
            return solution_path, len(board.explored_set)

        board.explored_set.add(tuple(current_state))

        for new_state in get_possible_moves(current_state):
            if tuple(new_state) not in board.explored_set:
                new_g = current_node.g + 1
                new_f = new_g + heuristic(new_state)
                new_node = Node(new_state, new_g, new_f, parent=current_node)
                heappush(board.fringe,  new_node)

    return [], len(board.explored_set)


def greedy_search(board: Puzzle, heuristic: Callable) -> Tuple[List[int], int]:
    """Implementation of the greedy search algorithm.
    :param board: the 8-puzzle to solve
    :param heuristic: the heuristic function to use
    :return: an ordered list with the solution path and the number of total nodes expanded
    """
    # YOUR CODE HERE
    start_node = Node(board.init_state, 0, heuristic(board.init_state))
    heappush(board.fringe,  start_node)

    while board.fringe:
        current_node = heappop(board.fringe)
        current_state = current_node.state

        if board.goal_test(current_state):
            # Reconstruct the solution path
            solution_path = [current_state]
            while current_node.parent:
                current_node = current_node.parent
                solution_path.append(current_node.state)
            solution_path.reverse()
            return solution_path, len(board.explored_set)

        board.explored_set.add(tuple(current_state))

        for new_state in get_possible_moves(current_state):
            if tuple(new_state) not in board.explored_set:
                new_f = heuristic(new_state)
                new_node = Node(new_state, 0, new_f, parent=current_node)
                heappush(board.fringe, new_node)

    return [], len(board.explored_set)

# YOUR CODE HERE
#test_heuristics()
