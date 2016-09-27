
# Author: Josh Kelle

# This script implements Newton's method for minimizing the objective
# function (negative log likelihood) of a logistic regression model.

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

computeNewtonDirection <- function(beta, y, X, m) {
  # Computes descent direction by Newton's method.
  #
  # Args:
  #   beta: the weight vector
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  hessian = computeHessian(beta, X, m)
  gradient = computeGradient(beta, y, X, m)
  direction = symmetricPosDefSolve(hessian, -gradient)
  return(direction)
}

newtonMethod <- function(y, X, m, max_iterations, convergence_threshold) {
  # Newton's method - TODO: explain newton's method
  # Step size is 1.
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
  cur_likelihood = computeNll(beta, y, X, m)
  
  # create vector to keep track negative log-likelihood values
  i = 1
  likelihoods = vector(mode = "list", length = max_iterations + 1)
  likelihoods[[i]] = cur_likelihood
  # bogus value to start the loop
  prev_likelihood = cur_likelihood + convergence_threshold + 1
  
  print(paste("iteration #", i, "likelihood =", cur_likelihood))
  
  # bogus value to start the loop
  prev_nll = cur_nll + convergence_threshold + 1

  while ((prev_nll - cur_nll)/prev_nll > convergence_threshold && i <= max_iterations) {
    # update beta
    direction = computeNewtonDirection(beta, y, X, m)
    beta = beta + direction
    
    # get new log likelihood
    prev_nll = cur_nll
    cur_nll = computeNll(beta, y, X, m)
    i = i + 1
    likelihoods[[i]] <- cur_nll
    
    print(paste("iteration #", i, "likelihood =", cur_nll))
  }
  
  return(list(beta, likelihoods[1:i]))
}
