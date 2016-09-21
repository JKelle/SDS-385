# Author: Josh Kelle

# This script implements a grab-bag of math functions shared among other files.
#   1. symmetricPosDefSolve()
#   2. computeNll()
#   3. computeGradient()
#   4. computeHessian()
#   5. computeStepSize()

symmetricPosDefSolve <- function(A, b) {
  # An efficient linear systems solver for Ax = b,
  # given A is symmetric and positive definite.
  #
  # Args:
  #   A: matrix (or Matrix) that is symmetric and positive definite
  #   b: the vector on the right hand side of the equation Ax = b
  
  # cholesky decomposition
  # L'L = A
  # L is upper right triangular
  # L' is lower left triangular
  L = chol(A)
  
  # L'Lx = b
  Lx = forwardsolve(t(L), b)
  
  # Lx = b
  x = backsolve(L, Lx)
  
  return(x)
}

computeNll <- function(beta, y, X, m) {  
  # Computes the negative log likelihood of beta given X, y, and m, (minus some
  # value constant in beta).
  # Implements the expresion: (m - y)' X beta + m' log(1 + exp(-X beta))
  #
  # Assumes yi ~ Binomial(mi, wi), where wi = 1/(1 + exp(-xi' * beta)).
  #
  # Args:
  #   beta: the weight vector
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  Xbeta = X %*% beta
  nll = t(m - y) %*% Xbeta + t(m) %*% log(1 + exp(-Xbeta))
  return(as.numeric(nll))
}

computeGradient <- function(beta, y, X, m) {
  # Computes the gradient of the negative log-likelihood function.
  # Implements the expression: X' (y_hat - y)
  #
  # Args:
  #   beta: the weight vector
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  w = 1/(1 + exp(-(X %*% beta)))
  gradient = t(X) %*% (m*w - y)
  return(gradient)
}

computeHessian <- function(beta, X, m) {
  # Computes the Hessian matrix of the negative log likelihood function.
  # Implements the expression: X'DX, where D is diagonal
  # with elements Dii = mi * wi * (1 - wi)
  #
  # Args:
  #   beta: the weight vector
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  w = 1/(1 + exp(-(X %*% beta)))
  D = as.numeric(m * w * (1 - w))
  hessian = crossprod(sqrt(D) * X)
  return(hessian)
}

computeStepSize <- function(beta, y, X, m, direction) {
  # Computes an appropriate step size when doing gradient descent.
  # Implements the backtracking algorithm for line search.
  #
  # Args:
  #   beta: the weight vector
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  #   direction: vector pointing in the direction of the line
  #              usually is the negative gradient
  stepsize = 1
  shrinkfactor = 0.5
  c = 1e-4
  
  # bogus values to start the loop
  lhs = 1
  rhs = 0
  
  while (lhs > rhs) {
    stepsize = shrinkfactor * stepsize
    lhs = computeNll(beta + stepsize*direction, y, X, m)
    rhs = computeNll(beta, y, X, m) - c*stepsize*(t(direction) %*% direction)
  }
  
  print(paste("step size =", stepsize))
  return(stepsize)
}
