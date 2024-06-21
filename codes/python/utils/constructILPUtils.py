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




@njit
def kron_sum(a,b):
    return (a[:,None] + b[None,:]).flatten()

@njit
def generate_sparse_one_val(dtype = numba.uint32):
    indices = np.arange(0,NCOLS*NROWS*NVALS,)
    indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS)
    data = np.ones((NCOLS*NROWS*NVALS), dtype = dtype)
    return data, indices, indptr

@njit
def generate_sparse_row(dtype = numba.uint32):
    indices = kron_sum(np.arange(0,9**3, 9**2), kron_sum(np.arange(0,9,1), np.arange(0,81,9)))   
    indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS)
    data = np.ones((NCOLS*NROWS*NVALS), dtype = dtype)
    return data, indices, indptr

@njit
def generate_sparse_col(dtype = numba.uint32):
    indices = kron_sum(np.arange(0,81,1), np.arange(0,9**3, 9**2))
    indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS)
    data = np.ones((NCOLS*NROWS*NVALS), dtype = dtype)
    return data, indices, indptr

@njit
def generate_sparse_block(dtype = numba.uint32):
    indices = kron_sum(np.arange(0,9**3, 9**2*3), kron_sum( np.arange(0, 9*9, 9*3), kron_sum(np.arange(0,9,1), kron_sum(np.arange(0,9**2*3,9**2), np.arange(0,3*9, 9)))))
    indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS)
    data = np.ones((NCOLS*NROWS*NVALS), dtype = dtype)
    return data, indices, indptr

@njit
def generate_sparse_clues(clues_inds_rows, clues_inds_cols, clues_vals, dtype = numba.uint32):
    row_inds = np.arange(clues_vals.shape[0])
    col_inds = NVALS*NCOLS*clues_inds_rows + NVALS*clues_inds_cols + clues_vals-1
    indices = col_inds
    indptr = np.arange(row_inds.shape[0]+1)
    data = np.ones((len(col_inds)), dtype = dtype)
    return data, indices, indptr

@njit
def generate_sparse_constaints(clues_array):
    data_0, indices_0, indptr_0 = generate_sparse_one_val()
    data_1, indices_1, indptr_1 = generate_sparse_row()
    data_2, indices_2, indptr_2 = generate_sparse_col()
    data_3, indices_3, indptr_3 = generate_sparse_block()
    clues_inds_rows, clues_inds_cols, clues_vals = sudoku_array_to_clues_njit(clues_array)
    data_4, indices_4, indptr_4 = generate_sparse_clues(clues_inds_rows, clues_inds_cols, clues_vals)

    data_list = [data_0, data_1, data_2, data_3, data_4]
    indices_list = [indices_0, indices_1, indices_2, indices_3, indices_4]
    indptr_list = [indptr_0, indptr_1, indptr_2, indptr_3, indptr_4]
    shape_list = [(9**2, 9**3), (9**2, 9**3), (9**2, 9**3), (9**2, 9**3), (len(indptr_list[4]), 9**3)]
    
    return data_list, indices_list, indptr_list, shape_list

@njit
def concat(arr_list):
    return [j for arr in arr_list for j in arr]

@njit
def count_concat(arr_list):
    offset = 0
    new_arr_list = arr_list #[None]*len(arr_list)#np.empty((len(arr_list))).tolist()
    for i, arr in enumerate(arr_list):
        new_arr_list[i] =  arr + offset
        offset += arr[-1]
    return concat(new_arr_list)

@njit
def stack_sparse_matrices(data_list, indices_list, indptr_list, shape_list):
    data = concat(data_list)
    indices = concat(indices_list)
    indptr = count_concat([indptr_list[1]] + [indices[1:] for indices in indptr_list[1:]])
    shape = (sum( [shape[0] for shape in shape_list])-1, 9**3)
    return data, indices, indptr, shape

@njit
def construct_c(size):
    return np.linspace(0,1,size)

def constuct_A_sparse_and_c(clues_array):
    data_list, indices_list, indptr_list, shape_list = generate_sparse_constaints(clues_array)
    data, indices, indptr, shape = stack_sparse_matrices(data_list, indices_list, indptr_list, shape_list)
    A_full_sparse = sp.sparse.csr_matrix((data, indices, indptr), shape = shape)
    c = construct_c(shape[1])

    return c, A_full_sparse
    
    