
# Author: Josh Kelle

# This script implements the quasi-Newton method for minimizing the objective
# function (negative log likelihood) of a logistic regression model.

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

quasiNewtonMethod <- function(y, X, m, max_iterations, convergence_threshold) {
  # quasi-Newton's method - Like Newton's method, but uses an approximation
  #   to the Hessian instead of the exact Hessian.
  #
  # Uses backtracking line search to select a dynamic step size.
  # 
  # Args:
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  #   max_iterations: the maximum number of iterations allowed.
  #     If convergence is not reached after max_iterations, then the
  #     'current' beta vector is returned anyway.
  #   convergence_threshold: convergence is reached if two adjacent iterations
  #     have negative log-likelihoods with percent difference less than
  #     convergence_threshold
  
  # initialize beta to be all 0
  beta = matrix(0, ncol(X), 1)
  
  # create vector to keep track negative log-likelihood values
  i = 1
  likelihoods = vector(mode = "list", length = max_iterations + 1)
  cur_nll = computeNll(beta, y, X, m)
  likelihoods[[i]] = cur_nll
  
  # bogus values to start the loop
  prev_nll = cur_nll + convergence_threshold + 1
  gradient = matrix(1, ncol(X), 1)  # all ones
  prev_beta = beta + 0.1
  inv_hessian_approx = diag(0.001, ncol(X))
  
  print(paste("iteration #", i, "likelihood =", cur_nll))
  
  while ((prev_nll - cur_nll)/prev_nll > convergence_threshold && i <= max_iterations) {
    # update hessian approximation
    prev_gradient = gradient
    gradient = computeGradient(beta, y, X, m)
    beta_diff = beta - prev_beta
    gradient_diff = gradient - prev_gradient
    inv_hessian_approx = computeInvHessianApprox(inv_hessian_approx, beta_diff, gradient_diff)
    
    # update beta
    direction = -(inv_hessian_approx %*% gradient)
    prev_beta = beta
    stepsize = linesearch(beta, y, X, m, direction)
    beta = beta + stepsize * direction
    
    # get new log likelihood
    prev_nll = cur_nll
    cur_nll = computeNll(beta, y, X, m)
    i = i + 1
    likelihoods[[i]] <- cur_nll
    
    print(paste("iteration #", i, "likelihood =", cur_nll))
  }
  
  return(list(beta=beta, likelihoods=likelihoods[1:i]))
}
