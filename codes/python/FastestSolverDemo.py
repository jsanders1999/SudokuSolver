import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from utils.timer import Timer

from numba import njit

from utils.constraintUtils import *
from utils.formattingUtils import *
from utils.constructILPUtils import *
from utils.solveILPUtils import *

#The sudoku from the problem
empty_sudoku = np.array(
   [[0, 0, 1, 0, 0, 0, 0, 0, 2],
	[0, 0, 0, 3, 0, 4, 5, 0, 0],
	[6, 0, 0, 7, 0, 0, 1, 0, 0],
	[0, 4, 0, 5, 0, 0, 0, 0, 0],
	[0, 2, 0, 0, 0, 0, 0, 8, 0],
	[0, 0, 0, 0, 0, 6, 0, 9, 0],
	[0, 0, 5, 0, 0, 9, 0, 0, 4],
	[0, 0, 8, 2, 0, 1, 0, 0, 0],
	[3, 0, 0, 0, 0, 0, 7, 0, 0]]
)

print()
print("Sudoku that will be solved:")
print(empty_sudoku)
print()

#call the constuctor once so numba compiles the function
c, A_full_sparse = constuct_A_sparse_and_c(empty_sudoku)

#repeat the solution finding several times to get more accurate timings
repeats = 10
for i in range(10):
	with Timer("1. Construct ILP from sudoku clues:"):
		c, A_full_sparse = constuct_A_sparse_and_c(empty_sudoku)
		
	with Timer("2. Solve ILP using scipy:"):
		res = solve_sudoku_scipy(c, A_full_sparse)
	
	with Timer("3. Full construct and solve time:"):
		c, A_full_sparse = constuct_A_sparse_and_c(empty_sudoku)
		res = solve_sudoku_scipy(c, A_full_sparse)
		

#show the solution
print("Solution to sudoku:")
print(vector_to_array(res.x))
print()

#print the timings
print("CPU times of the different processes:")
print()
np.set_printoptions(precision=6)
keys = np.sort(list(Timer.timers.keys()))
for key in keys:
	print(key, "\t\t", np.array(Timer.timers[key]))


