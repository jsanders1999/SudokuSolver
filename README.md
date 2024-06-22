# SudokuSolver
Code to solve a Sudoku in parallel on CPUs by modelling it as an Integer Linear Problem (ILP).
This was written as part of the second birthday celebration of the DelftBlue supercomputer of Delft University of Technology.

## How to run

### Locally

Simply navigate to the codes/python folder and run ```fastestSolverDemo.py``` or run the ```sandbox.ipynb``` jupyter notebook.

You need an evironment with python >3.8, numba, numpy and scipy. 

### Submitting to hpc05

```fastestSolverDemo.py``` can be submitted to DelftBlue by running the command ```sbatch submitDemo.slurm```. 
The resulting output will be in a file with the name ```slurm-XXXXXXX.out``` where X are numbers.
In this file, the soduku is solved 10 times, after which the mean and standart deviation of how long the solver took are calculated.
One can see that it takes less than 1 ms to constuct the ILP, and about 3-4 ms to solve the ILP. 

## Algoritm

The approach to solving the sudoku was taken from [here](https://www.mathworks.com/help/optim/ug/sudoku-puzzles-problem-based.html)

The idea is to model the solution of a sudocu as a binary vector of 729 elements. Here each element sais whether there is a certain value (1-9) present in a certain cell.
The constaints of the sudoku can then be modelled as linear constraints. So the constraints "each block/row/column must only have one of each value" can be reduces to linear constraints of the form:
$A \mathbf{x} = \mathbf{1}$. The same can be done for the constrains that the solution must satisfy the intial clues of the sudoku. We are then left with the ILP:

$$\min_{\mathbf{x} \in \{0,1\}^{729}} \mathbf{c}^T x$$
$$\text{st. } A\mathbf{x} = \mathbf{1}$$

The choice of the cost function vector $\mathbf{c}$ is arbitrary, as for a sudoku with sufficient clues the problem is constraint bound. 

One can then solve this ILP using plug and play MILP solvers such as those included in ```scipy```, ```HiGHs``` or ```Gurobi```.

This procedure can be sped up even further by determining the constraint matrix $A$ as a sparse matrix is CSR format. 
Another trick the python implementation uses is to compile the code that constructs the ILP using ```numba```
