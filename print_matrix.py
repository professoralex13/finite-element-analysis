"""
Pretty numpy Matrix/Vector visualization

Author: Alex Cutforth
August 2026
"""

import numpy as np

ZERO_THRESHOLD = 1e-10


def format_number(x, decimals, max_len=None):
    if abs(x) > ZERO_THRESHOLD:
        return (
            f"{x:>{max_len}.{decimals}f}"
            if max_len is not None
            else f"{x:.{decimals}f}"
        )

    return f"{0:^{max_len}}" if max_len is not None else str(0)


def print_matrix(matrix, decimals=4, scale_coef=None):
    """Prints a matrix with nice formatting and automatic scientific scaling"""
    if scale_coef is None:
        max_val = np.max(np.abs(matrix))
        exp = np.floor(np.log10(max_val)) if max_val > 0 else 0
        scale_coef = 10**exp

    # Scale matrix values
    printed_matrix = matrix / scale_coef

    max_len = max(len(format_number(x, decimals)) for x in printed_matrix.flatten())

    for i, row in enumerate(printed_matrix):
        print("|", end="")
        if isinstance(row, np.float64):
            print(format_number(row, decimals, max_len=max_len), end="")
        else:
            print(
                " ".join(format_number(x, decimals, max_len=max_len) for x in row),
                end="",
            )
        if i == len(printed_matrix) // 2:
            print(f"| x {scale_coef:.0e}")
        else:
            print("|")


def print_matrix_rounded(matrix):
    """Logs a matrix of integers"""
    for row in matrix:
        print("|", end="")
        print(" ".join(str(int(x)) for x in row), end="")
        print("|")
