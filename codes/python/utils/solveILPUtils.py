import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import optimize as spopt
from scipy.optimize import milp


def solve_sudoku_scipy(c, A_full):
	solution_size = A_full.shape[1]
	integrality = np.ones(solution_size, dtype = int)
	bounds = spopt.Bounds(lb=0, ub=1) #somthing with 0 and 
	constraints = spopt.LinearConstraint(A_full, lb=1, ub=1) #somthing with A
	res = milp(c, integrality = integrality, bounds = bounds, constraints = constraints)
	return res

# import gurobipy as gp
# from gurobipy import GRB
# import highspy

# env = gp.Env(empty=True)
# env.setParam('OutputFlag', 0)
# env.start()
		
# def solve_sudoku_gurobi(c, A_full):
#	 # Create a new model
#	 m = gp.Model("matrix1", env = env) 
#	 # Create variables
#	 solution_size = A_full.shape[1]
#	 x = m.addMVar(shape=solution_size, vtype=GRB.BINARY, name="x")
#	 # Build rhs vector
#	 rhs = np.ones(A_full.shape[0], dtype = int)
#	 # Add constraints
#	 m.addConstr(A_full @ x == rhs, name="constraints")
#	 # Set objective
#	 m.setObjective(c @ x, GRB.MINIMIZE)
#	 # Optimize model
#	 m.optimize()
#	 return x.X
	

# def solve_sudoku_highspy(c, A_full):

#	 rows, cols = A_full.shape
#	 # Highs h
#	 h = highspy.Highs()

#	 # Define a HighsLp instance
#	 lp = highspy.HighsLp()
	
#	 lp.num_col_ = cols
#	 lp.num_row_ = rows
#	 lp.col_cost_ = c
#	 lp.col_lower_ = np.zeros(c.shape, dtype = int)
#	 lp.col_upper_ = np.ones(c.shape, dtype = int)
#	 lp.row_lower_ = np.ones((rows), dtype = int)
#	 lp.row_upper_ = np.ones((rows), dtype = int)
#	 lp.integrality_ = [highspy._core.HighsVarType.kInteger]*cols # np.ones(c.shape, dtype = int).tolist()
#	 #lp.integrality_ = 1#np.ones(c.shape, dtype = int)#highspy.HighsVarType.kInteger
	
#	 # In a HighsLp instsance, the number of nonzeros is given by a fictitious final start
#	 lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
#	 lp.a_matrix_.start_ = A_full.indptr
#	 lp.a_matrix_.index_ = A_full.indices
#	 lp.a_matrix_.value_ = A_full.data
#	 h.passModel(lp)

#	 # Get and set options
#	 options = h.getOptions()
#	 options.log_to_console = False
#	 h.passOptions(options)

#	 h.run()
	
#	 solution = h.getSolution()
#	 x= np.array(solution.col_value)
#	 return x
