import numpy as np
import scipy as sp
from numba import njit
import numba


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

