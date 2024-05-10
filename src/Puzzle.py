from typing import List, Tuple
from utils import print_puzzle


class Puzzle:

    def __init__(self, init_state: List[int], puzzle_type: int = 8):
        """
        Initialize a new 8-puzzle board
        :param init_state: Initial configuration of the board
        """
        self.init_state = init_state
        self.goal_state = [i for i in range(1, 9)] + [0]
        # YOUR CODE HERE
        raise NotImplementedError()

    def goal_test(self, state: List[int]):
        """Test if goal state is reached
        :param state: board configuration to check
        :return: true if the passed configuration is equal to goal configuration
        """
        # YOUR CODE HERE
        raise NotImplementedError()

    def is_explored(self, state: List[int]):
        """Check if a particular board configuration has already been explored
        :param state: board configuration to check
        :return: true if a particular configuration has already been explored
        """
        # YOUR CODE HERE
        raise NotImplementedError()

    def reset(self):
        # YOUR CODE HERE
        raise NotImplementedError()


def move_left(position: int) -> int:
    """Move one position left in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving to the left
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def move_right(position: int) -> int:
    """Move one position right in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving to the right
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def move_up(position: int) -> int:
    """Move one position up in 8 puzzle if possible
    :param position: current position of the 0 tile
    :return: new position of the 0 tile after moving upwards
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def move_down(position: int):
    """Move one position down in 8 puzzle if possible
    :param position: current position of the 0 tile.
    :return: new position of the 0 tile after moving downwards
    """
    # YOUR CODE HERE
    raise NotImplementedError()


def get_possible_moves(state: List[int]) -> List[List[int]]:
    """Check whether a move is possible in left, right, up, down direction and store it.
    :param state: current configuration of the puzzle as one dimensional list
    :return: list containing the new configurations after applying all possible moves
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# YOUR CODE HERE
raise NotImplementedError()