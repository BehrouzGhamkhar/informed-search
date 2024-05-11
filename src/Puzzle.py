from typing import List, Tuple
from utils import print_puzzle
from heapq import heappop, heappush


class Node:
    def __init__(self, state: List[int], g: int, f: int, parent=None):
        self.state = state
        self.g = g  # Cost to reach this state
        self.f = f  # Estimated total cost. (g + h) for A* and h for best first search
        self.parent = parent  # Parent node

    def __lt__(self, other):
        return self.f < other.f


class Puzzle:

    def __init__(self, init_state: List[int], puzzle_type: int = 8):
        """
        Initialize a new 8-puzzle board
        :param init_state: Initial configuration of the board
        """
        self.init_state = init_state
        self.goal_state = [i for i in range(1, 9)] + [0]
        self.explored_set = set()
        self.fringe = []
        self.puzzle_type = puzzle_type

    def goal_test(self, state: List[int]):
        """Test if goal state is reached
        :param state: board configuration to check
        :return: true if the passed configuration is equal to goal configuration
        """
        # YOUR CODE HERE
        return state == self.goal_state

    def is_explored(self, state: List[int]):
        """Check if a particular board configuration has already been explored
        :param state: board configuration to check
        :return: true if a particular configuration has already been explored
        """
        # YOUR CODE HERE
        return state in self.explored_set

    def reset(self):
        # YOUR CODE HERE
        self.explored_set = set()


def move_left(position: int) -> int:
    """Move one position left in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving to the left
    """
    # YOUR CODE HERE
    return position + 1 if (position % 3) < 2 else position


def move_right(position: int) -> int:
    """Move one position right in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving to the right
    """
    # YOUR CODE HERE
    return position - 1 if (position % 3) > 0 else position


def move_up(position: int) -> int:
    """Move one position up in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving upwards
    """
    # YOUR CODE HERE
    return position + 3 if (position // 3) < 2 else position


def move_down(position: int):
    """Move one position down in 8 puzzle if possible
    :param position: current position of the 0 tile.
    :return: new position of the 0 tile after moving downwards
    """
    # YOUR CODE HERE
    return position - 3 if (position // 3) > 0 else position


def get_possible_moves(state: List[int]) -> List[List[int]]:
    """Check whether a move is possible in left, right, up, down direction and store it.
    :param state: current configuration of the puzzle as one dimensional list
    :return: list containing the new configurations after applying all possible moves
    """
    # YOUR CODE HERE
    current_empty_tile = state.index(0)

    def apply_move(move):
        new_empty_tile = move(current_empty_tile)
        if new_empty_tile != current_empty_tile:
            new_state = state.copy()
            new_state[current_empty_tile], new_state[new_empty_tile] = new_state[new_empty_tile], new_state[
                current_empty_tile]
            return new_state

    move_lists = [apply_move(move_left), apply_move(move_right), apply_move(move_up), apply_move(move_down)]
    possible_moves = [move for move in move_lists if move is not None]

    return possible_moves


# YOUR CODE HERE
state = [1, 2, 3
    , 4, 0, 6
    , 7, 5, 8]

moves = get_possible_moves(state)
#print(moves)
