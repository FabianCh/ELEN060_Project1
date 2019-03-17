import math
import numpy as np


def single_square_entropy(number_of_possibility=9):
    ans = 0
    p = 1/number_of_possibility
    for _ in range(number_of_possibility):
        ans += -p * math.log2(p)
    return ans


def square_entropy(square):
    number_saw = []
    counter_saw = 0
    for i in range(3):
        for j in range(3):
            n = square[i][j]
            if n !=0:
                if n not in number_saw:
                    number_saw.append(n)
                    counter_saw += 1
    number_of_possibility = 9 - counter_saw
    return number_of_possibility * single_square_entropy(number_of_possibility)


def sudoku_entropy(sudoku_grid):
    ans = 0
    for i in range(9):
        for j in range(9):
            if sudoku_grid[i][j] != 0:
                number_saw = []
                counter_saw = 0

                # count the number on the same line

                for k in range(9):
                    n = sudoku_grid[i][k]
                    if n != 0:
                        if n not in number_saw:
                            number_saw.append(n)
                            counter_saw += 1

                # count the number on the same line

                for k in range(9):
                    n = sudoku_grid[k][j]
                    if n != 0:
                        if n not in number_saw:
                            number_saw.append(n)
                            counter_saw += 1

                # count the number on the same square

                    # Identification of the square

                i_square = i//3
                j_square = j//3

                for i2 in range(3):
                    for j2 in range(3):
                        n = sudoku_grid[3*i_square + i2][j_square + j2]
                        if n != 0:
                            if n not in number_saw:
                                number_saw.append(n)
                                counter_saw += 1

                number_of_possibility = 9 - counter_saw
                ans += single_square_entropy(number_of_possibility)
    return ans





