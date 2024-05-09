from typing import Tuple

def print_puzzle(puzzle: Tuple[(int,) * 9], row_size: int = 3):
    """Function to print the puzzle to console

    Parameters
    ----------
    puzzle : tuple
        8 puzzle configuration
    row_size: int
        row size of the puzzle, 3 for a classic 8 puzzle
    """

    for idx, val in enumerate(puzzle):

        if (idx + 1) % row_size == 0:
            print("  ", val)
        else:
            print("  ", val, end="")