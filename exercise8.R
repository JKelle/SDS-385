
library(Matrix)
source("common.R")

setwd("~/Dropbox/University of Texas/SDS 385; Statistical Models for Big Data/code")

# load data
data = as.matrix(read.csv('../data/fmri_z.csv'))

# plot heat map (from Giorgio's code)
image(Matrix(t(data)), sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)


###########################
##  laplacian smoothing  ##
###########################


# This function constructs the D matrix for a 2D grid.
# From James' makeD2_sparse.R function
makeD2_sparse = function (dim1, dim2)  {
  D1 = bandSparse(
    dim1 * dim2,
    m = dim1 * dim2, 
    k = c(0, 1), 
    diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * dim2 - 1)))
  
  D1 = D1[(seq(1, dim1 * dim2)%%dim1) != 0, ]
  
  D2 = bandSparse(
    dim1 * dim2 - dim1, 
    m = dim1 * dim2, 
    k = c(0, dim1), 
    diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * dim2 - 1)))
  
  return(rBind(D1, D2))
}

# convert pixel grid into a sparse vector
x_raw = Matrix(as.vector(t(data)))

# construct the D matrix = the "oriented edge matrix"
dim1 = dim(data)[1]
dim2 = dim(data)[2]
D = makeD2_sparse(dim1, dim2)
m = dim(D)[1]  # number of edges
n = dim(D)[2]  # number of vertices

# define lambda - smoothing parameter
lambda = 10

# direct solver.
# solve the system (I - lambda * L) * x_smooth = x_raw
L = crossprod(D)
A = lambda * L
diag(A) = diag(A) + 1
x_smooth = Matrix::solve(A, x_raw)

# plot the smoothed image
data_smooth = Matrix(as.vector(x_smooth), nrow=dim1, ncol=dim2)
data_smooth[t(data) == 0] <- 0
image(data_smooth, sub = '', xlab = '', ylab = '', cuts = 80, lwd = 0)
