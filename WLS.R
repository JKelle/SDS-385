
# Author: Josh Kelle

# This script implements the closed form solution to weighted least squares.
# It shows the difference in efficiency between the naive solution (explicitly
# computing an inverse) versus solving the linear system via a decomposition.
#
# This script also experiments with sparse matrices.

library(microbenchmark)
library('Matrix')

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

N = 2000
P = 500
y = matrix(rnorm(N), nrow=N)
X = matrix(rnorm(N*P), nrow=N)
mask = matrix(rbinom(N*P,1,0.05), nrow=N)
X = mask*X
Xsparse = Matrix(X, sparse=TRUE)
weights = rnorm(N)^2
W = diag(weights, N);

inversionMethod <- function(X, y, W) {
  return(solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y)
}

# solves X'WX * beta = X'Wy
myMethod <- function(X, y, weights) {
  # X'WX -> (W^(1/2) X)'(W^(1/2) X)
  A = crossprod(sqrt(weights) * X)
  b = t(X * weights) %*% y
  beta = symmetricPosDefSolve(A, b)
  return(beta)
}

microbenchmark(
  inversionMethod(X, y, W),
  myMethod(X, y, weights),
  myMethod(Xsparse, y, weights),
  times=1
)
