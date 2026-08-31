"""
Pretty numpy Matrix/Vector visualization

Author: Alex Cutforth
August 2026
"""

import numpy as np

ZERO_THRESHOLD = 1e-10


def print_table(table: list[list[str]], suffix: str | None = None):
    max_len = max(len(item) for sublist in table for item in sublist)

    for i, row in enumerate(table):
        print("|", end="")

        for j, item in enumerate(row):
            if j != 0:
                print(" ", end="")
            if str(item) == "0":
                print(f"{item:^{max_len}}", end="")
            else:
                print(f"{item:>{max_len}}", end="")
        if suffix is not None and i == len(table) // 2:
            print(f"| {suffix}")
        else:
            print("|")


def format_number(x, decimals):
    """Formats a number with a given number of decimals and padding"""
    if abs(x) > ZERO_THRESHOLD:
        return f"{x:.{decimals}f}"

    return "0"


def print_matrix(matrix, decimals=4, scale_coef=None):
    """Prints a matrix with nice formatting and automatic scientific scaling"""
    if scale_coef is None:
        max_val = np.max(np.abs(matrix))
        exp = np.floor(np.log10(max_val)) if max_val > 0 else 0
        scale_coef = 10**exp

    # Scale matrix values
    printed_matrix = matrix / scale_coef

    table = [
        (
            [format_number(row, decimals)]
            if isinstance(row, np.float64)
            else [format_number(x, decimals) for x in row]
        )
        for row in printed_matrix
    ]

    print_table(table, f"x {scale_coef:.0e}")


def print_matrix_rounded(matrix):
    """Logs a matrix of integers"""
    for row in matrix:
        print("|", end="")
        print(" ".join(str(int(x)) for x in row), end="")
        print("|")
