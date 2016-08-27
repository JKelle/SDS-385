library(microbenchmark)

N = 2000
P = 500
X = matrix(rnorm(N*P), nrow=N)
y = matrix(rnorm(N), nrow=N)
weights = rnorm(N)^2
W = diag(weights, N);

inversionMethod <- function(X, y, W) {
  return(solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y)
}

myMethod <- function(X, y, W) {
  # tXW = t(X) %*% W
  # don't waste time multiplying by a bunch of zeros
  tXW = t(X * weights)
  
  # cholesky decomposition
  # L'L = X'WX
  # L is upper right triangular
  # L' is lower left triangular
  L = chol(tXW %*% X)
  
  # L'b = X'Wy
  b = forwardsolve(t(L), tXW %*% y)
  
  # L*beta = b
  beta = backsolve(L, b)
  
  return(beta)
}

microbenchmark(
  inversionMethod(X, y, W),
  myMethod(X, y, W),
  times=5
)
