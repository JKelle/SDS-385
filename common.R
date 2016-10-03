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

computeHessianApprox <- function(B, s, y) {
  # Computes the BFGS approximation to Hessian matrix
  # of the negative log likelihood function.
  #
  # Args:
  #   B: the hessian approximation at time step k
  #     B is assumed to be symmetric and positive definite.
  #   s: the difference between two successive values of x: (x_k+1 - x_k)
  #   y: the difference between two successive values of the gradient
  #
  # Returns:
  #   updated_B: the hessian approximation at time step k+1
  term1 = crossprod(t(s) %*% B) / as.numeric(t(s) %*% B %*% s)
  term2 = crossprod(t(y)) / as.numeric(t(y) %*% s)
  updated_B = B - term1 + term2
  return(updated_B)
}

computeInvHessianApprox <- function(H, s, y) {
  # Computes the BFGS approximation to Hessian matrix
  # of the negative log likelihood function.
  #
  # Args:
  #   H: the inverse hessian approximation at time step k
  #     H is assumed to be symmetric and positive definite.
  #   s: the difference between two successive values of x: (x_k+1 - x_k)
  #   y: the difference between two successive values of the gradient
  #
  # Returns:
  #   updated_H: the hessian approximation at time step k+1
  p = 1 / as.numeric(t(y) %*% s)
  I = diag(1, nrow(H))
  updated_H = (I - p * s %*% t(y)) %*% H %*% (I - p * y %*% t(s)) + p * crossprod(t(s))
  return(updated_H)
}

linesearch <- function(beta, y, X, m, direction) {
  # Computes an appropriate step size when doing gradient descent.
  # Implements the backtracking line search algorithm.
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
  
  return(stepsize)
}

AdaGradUpdate <- function(beta, single_y, single_x, single_m, hessian_approx, learning_rate) {
  gradient = computeGradient(beta, single_y, single_x, single_m)
  hessian_approx = hessian_approx + gradient ^ 2
  beta = beta - learning_rate * gradient / sqrt(hessian_approx + 1e-8)
  return(list(beta=beta, hessian_approx=hessian_approx))
}