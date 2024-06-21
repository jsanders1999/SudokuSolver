import numpy as np
import scipy as sp
from numba import njit
import numba

from utils.formattingUtils import *
	

#these are operatiors that should apply to a 729 long solution vector x
#x 
NCOLS = 9
NROWS = 9
NVALS = 9

def generate_one_value_constr_sparse(format = "csr", dtype = int):
	#[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,....]
	#[0,0,0,0,0,0,0,0,0, 1,1,1,1,1,....
	#...
	return sp.sparse.kron(
		sp.sparse.eye(NROWS*NCOLS, dtype = dtype, format=format),
		np.ones((1,NVALS), dtype = dtype),
		format=format
	)

def generate_one_value_constr_dense(dtype = int):
	#[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,....]
	#[0,0,0,0,0,0,0,0,0, 1,1,1,1,1,....
	#...
	return np.kron(
		np.eye(NROWS*NCOLS, dtype = dtype),
		np.ones((1,NVALS), dtype = dtype)
	)

@njit
def generate_one_value_constr_njit():
	#[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,....]
	#[0,0,0,0,0,0,0,0,0, 1,1,1,1,1,....
	#...
	return np.kron(
		np.eye(NROWS*NCOLS, dtype=numba.boolean),
		np.ones((1,NVALS), dtype=numba.boolean)
	)


def generate_row_uniq_constr_sparse(format = "csr", dtype = int):
	#[1,..,0, 1,..,0, ... 1,... 0,  0,1,..,0, 0, 
	return sp.sparse.kron(
		sp.sparse.eye(NROWS, dtype = dtype, format=format),
		sp.sparse.kron(np.ones((1,NCOLS), dtype = dtype), 
					   sp.sparse.eye(NVALS, dtype = dtype),
					   format=format),
		format=format
	)

def generate_row_uniq_constr_dense(dtype = int):
	#[1,..,0, 1,..,0, ... 1,... 0,  0,1,..,0, 0, 
	return np.kron(
		np.eye(NROWS, dtype = dtype),
		np.kron(np.ones((1,NCOLS), dtype = dtype), 
					   np.eye(NVALS, dtype =dtype))
	)

@njit
def generate_row_uniq_constr_njit():
	#[1,..,0, 1,..,0, ... 1,... 0,  0,1,..,0, 0, 
	return np.kron(
		np.eye(NROWS, dtype = numba.boolean),
		np.kron(np.ones((1,NCOLS), dtype = numba.boolean), 
					   np.eye(NVALS, dtype = numba.boolean))
	)


def generate_col_uniq_constr_sparse(format = "csr", dtype = int):
	return sp.sparse.kron(
		np.ones((1,NROWS), dtype = dtype),
		sp.sparse.eye(NCOLS*NVALS, dtype = dtype, format=format),
		format=format
	)

def generate_col_uniq_constr_dense(dtype = int):
	return np.kron(
		np.ones((1,NROWS), dtype = dtype),
		np.eye(NCOLS*NVALS, dtype = dtype)
	)

@njit
def generate_col_uniq_constr_njit():
	return np.kron(
		np.ones((1,NROWS), dtype = numba.boolean),
		np.eye(NCOLS*NVALS, dtype = numba.boolean)
	)
	

def generate_block_uniq_constr_sparse(format = "csr", dtype = int):
	return sp.sparse.kron(
		sp.sparse.eye(3, dtype = dtype, format=format),
		sp.sparse.kron(
			np.ones((1,3), dtype = dtype), 
			sp.sparse.kron(
				sp.sparse.eye(3, dtype = dtype, format=format),
				sp.sparse.kron( 
					np.ones((1,3), dtype = dtype),
					sp.sparse.eye(9, dtype = dtype, format=format)
				),
				format=format
			),
			format=format   
		),
		format=format
	)

def generate_block_uniq_constr_dense(dtype = int):
	return np.kron(
		np.eye(3, dtype = dtype),
		np.kron(
			np.ones((1,3), dtype = dtype), 
			np.kron(
				np.eye(3, dtype = dtype),
				np.kron( 
					np.ones((1,3), dtype = dtype),
					np.eye(9, dtype = dtype)
				),
			),
		),
	)

@njit
def generate_block_uniq_constr_njit():
	return np.kron(
		np.eye(3, dtype = numba.boolean),
		np.kron(
			np.ones((1,3), dtype = numba.boolean), 
			np.kron(
				np.eye(3, dtype = numba.boolean),
				np.kron( 
					np.ones((1,3), dtype = numba.boolean),
					np.eye(9, dtype = numba.boolean)
				),
			),
		),
	)

def generate_clues_contr_sparse(clues_inds_rows, clues_inds_cols, clues_vals,
						 format = "csr", dtype = int):
	#x[i,j,m] = 1
	op_shape = (clues_vals.shape[0], NROWS*NCOLS*NVALS)

	data = np.ones((clues_vals.shape[0]), dtype=dtype)
	row_inds = np.arange(clues_vals.shape[0])
	col_inds = NVALS*NCOLS*clues_inds_rows + NVALS*clues_inds_cols + clues_vals-1
	op_sparse = sp.sparse.csr_matrix( (data, (row_inds, col_inds)), shape = op_shape)
	return op_sparse

def generate_clues_contr_dense(clues_inds_rows, clues_inds_cols, clues_vals, dtype = int):
	#x[i,j,m] = 1
	op_shape = (clues_vals.shape[0], NROWS*NCOLS*NVALS)
	op_dense = np.zeros(op_shape, dtype = dtype)

	row_inds = np.arange(clues_vals.shape[0])
	col_inds = NVALS*NCOLS*clues_inds_rows + NVALS*clues_inds_cols + clues_vals-1
	op_dense[row_inds, col_inds] = 1
	return op_dense

@njit
def generate_clues_contr_njit(clues_inds_rows, clues_inds_cols, clues_vals):
	#x[i,j,m] = 1
	op_shape = (clues_vals.shape[0], NROWS*NCOLS*NVALS)
	op_dense = np.zeros(op_shape, dtype = numba.boolean)
	
	col_inds = NVALS*NCOLS*clues_inds_rows + NVALS*clues_inds_cols + clues_vals-1
	for i, col_ind in enumerate(col_inds):
		op_dense[i, col_ind] = True
	return op_dense


@njit
def kron_sum(a,b):
	return (a.reshape(-1, 1) + b.reshape(1, -1)).flatten()

@njit
def generate_sparse_one_val():
	indices = np.arange(0,NCOLS*NROWS*NVALS, dtype = numba.u2)
	indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS, dtype = numba.u2)
	data = np.ones((NCOLS*NROWS*NVALS), dtype = numba.b1)
	return data, indices, indptr

@njit
def generate_sparse_row():
	indices = kron_sum(np.arange(0,9**3, 9**2, dtype = numba.u2),
					   kron_sum(np.arange(0,9,1, dtype = numba.u2),
								np.arange(0,81,9, dtype = numba.u2)
							   )
					  )   
	indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS, dtype = numba.u2)
	data = np.ones((NCOLS*NROWS*NVALS), dtype = numba.b1)
	return data, indices, indptr

@njit
def generate_sparse_col():
	indices = kron_sum(np.arange(0,81,1, dtype = numba.u2), np.arange(0,9**3, 9**2, dtype = numba.u2))
	indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS, dtype = numba.u2)
	data = np.ones((NCOLS*NROWS*NVALS), dtype = numba.b1)
	return data, indices, indptr

@njit
def generate_sparse_block():
	indices = kron_sum(np.arange(0,9**3, 9**2*3, dtype = numba.u2),
					   kron_sum( np.arange(0, 9*9, 9*3, dtype = numba.u2),
								kron_sum(np.arange(0,9,1, dtype = numba.u2),
										 kron_sum(np.arange(0,9**2*3,9**2, dtype = numba.u2),
												  np.arange(0,3*9, 9, dtype = numba.u2)
												 )
										)
							   )
					  )
	indptr = np.arange(0, (NCOLS*NROWS*NVALS+1), NVALS, dtype = numba.u2)
	data = np.ones((NCOLS*NROWS*NVALS), dtype = numba.b1)
	return data, indices, indptr

@njit
def generate_sparse_clues(clues_inds_rows, clues_inds_cols, clues_vals):
	row_inds = np.arange(clues_vals.shape[0], dtype = numba.u2)
	col_inds = NVALS*NCOLS*clues_inds_rows + NVALS*clues_inds_cols + clues_vals-1
	col_inds = col_inds.astype(numba.u2)
	indices = col_inds
	indptr = np.arange(row_inds.shape[0]+1, dtype = numba.u2)
	data = np.ones((len(col_inds)), dtype = numba.b1)
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

#@njit
def concat(arr_list):
	return [j for arr in arr_list for j in arr]

#@njit
def count_concat(arr_list):
	offset = 0
	new_arr_list = arr_list #[None]*len(arr_list)#np.empty((len(arr_list))).tolist()
	for i, arr in enumerate(arr_list):
		new_arr_list[i] =  arr + offset
		offset += arr[-1]
	return concat(new_arr_list)

#@njit
def stack_sparse_matrices(data_list, indices_list, indptr_list, shape_list):
	data = concat(data_list)
	indices = concat(indices_list)
	indptr = count_concat([indptr_list[1]] + [indices[1:] for indices in indptr_list[1:]])
	shape = (sum( [shape[0] for shape in shape_list])-1, 9**3)
	return data, indices, indptr, shape

