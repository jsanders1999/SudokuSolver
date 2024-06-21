import numpy as np
import scipy as sp
from numba import njit
	
#you can also just turn the matrix into a sparse one in index format, that might be faster
def sudoku_array_to_clues(clue_array):
	clues_inds_rows, clues_inds_cols = np.where(clue_array>0)
	clues_vals = clue_array[clues_inds_rows, clues_inds_cols]
	return clues_inds_rows, clues_inds_cols, clues_vals

@njit
def sudoku_array_to_clues_njit(clue_array):
	clues_inds_rows, clues_inds_cols = np.where(clue_array>0)
	clues_vals = np.empty_like(clues_inds_rows) 
	for i, (row_ind, col_ind) in enumerate(zip(clues_inds_rows, clues_inds_cols)):
		clues_vals[i] = clue_array[row_ind, col_ind]
	return clues_inds_rows, clues_inds_cols, clues_vals

def array_to_vector(sudoku_array):
	#turn a sudoku values array into a binary vector
	flat_values = sudoku_array.flatten()
	binary_values = np.zeros((flat_values.shape[0], 9), dtype = int)
	indices = np.arange(flat_values.shape[0], dtype = int)
	binary_values[indices, flat_values-1] = 1
	return binary_values.flatten()

def vector_to_array(binary_solution):
	#turn a binary vector into a soduku values array
	binary_values = binary_solution.reshape((81,9)) #magig numbers are sudoky dims and number of values possible
	_, flat_values = np.where(binary_values)
	sudoku_array = (flat_values+1).reshape(9,9)
	return sudoku_array
