import numpy as np
import scipy as sp
from numba import njit

from utils.formattingUtils import *
from utils.constraintUtils import *
	

def setup_ILP_sparse(clue_array, format = "csr", dtype = int):
    clues_inds_rows, clues_inds_cols, clues_vals = sudoku_array_to_clues(clue_array)
    
    A_one_val = generate_one_value_constr_sparse(format = format, dtype = dtype)
    A_row = generate_row_uniq_constr_sparse(format = format, dtype = dtype)
    A_col = generate_col_uniq_constr_sparse(format = format, dtype = dtype)
    A_block = generate_block_uniq_constr_sparse(format = format, dtype = dtype)
    A_clues = generate_clues_contr_sparse(clues_inds_rows, clues_inds_cols, clues_vals, format = format, dtype = dtype)

    A_full = sp.sparse.vstack((A_one_val, A_row, A_col, A_block, A_clues), format = format)
    c = np.linspace(0,1,A_full.shape[1]) #np.zeros(solution_size, dtype = np.float64)
    return c, A_full

def setup_ILP_dense(clue_array, dtype = int):
    clues_inds_rows, clues_inds_cols, clues_vals = sudoku_array_to_clues(clue_array)
    
    A_one_val = generate_one_value_constr_dense(dtype = dtype)
    A_row = generate_row_uniq_constr_dense(dtype = dtype)
    A_col = generate_col_uniq_constr_dense(dtype = dtype)
    A_block = generate_block_uniq_constr_dense(dtype = dtype)
    A_clues = generate_clues_contr_dense(clues_inds_rows, clues_inds_cols, clues_vals, dtype = dtype)

    A_full = np.vstack((A_one_val, A_row, A_col, A_block, A_clues))
    c = np.linspace(0,1,A_full.shape[1]) #np.zeros(solution_size, dtype = np.float64)
    return c, A_full

def setup_ILP_sparse_end(clue_array, format = "csr", dtype = int):
    clues_inds_rows, clues_inds_cols, clues_vals = sudoku_array_to_clues(clue_array)
    
    A_one_val = generate_one_value_constr_dense(dtype = dtype)
    A_row = generate_row_uniq_constr_dense(dtype = dtype)
    A_col = generate_col_uniq_constr_dense(dtype = dtype)
    A_block = generate_block_uniq_constr_dense(dtype = dtype)
    A_clues = generate_clues_contr_dense(clues_inds_rows, clues_inds_cols, clues_vals, dtype = dtype)

    A_full = sp.sparse.vstack((A_one_val, A_row, A_col, A_block, A_clues), format = format)
    c = np.linspace(0,1,A_full.shape[1]) #np.zeros(solution_size, dtype = np.float64)
    return c, A_full

@njit
def setup_ILP_njit(clue_array):
    clues_inds_rows, clues_inds_cols, clues_vals = sudoku_array_to_clues_njit(clue_array)
    
    A_one_val = generate_one_value_constr_njit()
    A_row = generate_row_uniq_constr_njit()
    A_col = generate_col_uniq_constr_njit()
    A_block = generate_block_uniq_constr_njit()
    A_clues = generate_clues_contr_njit(clues_inds_rows, clues_inds_cols, clues_vals)

    A_full = np.vstack((A_one_val, A_row, A_col, A_block, A_clues))
    c = np.linspace(0,1,A_full.shape[1]) #np.zeros(solution_size, dtype = np.float64)
    return c, A_full
    