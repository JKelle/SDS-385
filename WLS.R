N = 8
P = 3
X = matrix(rnorm(N*P), nrow=N)
y = matrix(rnorm(N), nrow=N)
W = diag(1, N, N)


inversionMethod <- function(X, y, W) {
  return(solve(t(X)%*%W%*%X)%*%t(X)%*%W%*%y)
}

myMethod <- function(X, y, W) {
  # L'L = X'WX
  # L is upper right triangular
  # L' is lower left triangular
  L = chol(t(X) %*% W %*% X)
  
  # L'b = XWy
  b = forwardsolve(t(L), t(X) %*% W %*% y)
  
  # L*beta = b
  beta = backsolve(L, b)
  
  return(beta)
}

inversionMethod(X, y, W)
myMethod(X, y, W)

