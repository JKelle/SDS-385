
# Author: Josh Kelle

# This script implements gradient descent for minimizing the objective
# function (negative log likelihood) of a logistic regression model.

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

gradientDescent <- function(y, X, m, max_iterations, convergence_threshold) {
  # Gradient descent algorithm - update direction is simply the
  # negative gradient of the negative log likelihood function.
  # Step size is chosen by line search (back tracking).
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
  cur_nll = computeNll(beta, y, X, m)
  
  # create vector to keep track negative log-likelihood values
  i = 1
  likelihoods = vector(mode = "list", length = max_iterations + 1)
  likelihoods[[i]] = cur_nll
  # bogus value to start the loop
  prev_nll = cur_nll + convergence_threshold + 1
  
  print(paste("iteration #", i, "likelihood =", cur_nll))
  
  while ((prev_nll - cur_nll)/prev_nll > convergence_threshold && i <= max_iterations) {
    # update beta
    direction = -computeGradient(beta, y, X, m)
    stepsize = linesearch(beta, y, X, m, direction)
    beta = beta + stepsize * direction
    
    # get new negative log-likelihood
    prev_nll = cur_nll
    cur_nll = computeNll(beta, y, X, m)
    i = i + 1
    likelihoods[[i]] <- cur_nll
    
    print(paste("iteration #", i, "likelihood =", cur_nll))
  }
  
  return(c(beta, likelihoods[1:i]))
}
