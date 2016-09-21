
# Author: Josh Kelle

# This script implements stochastic gradient descent (SGD) for minimizing the
# objective function (negative log likelihood) of a logistic regression model.

# TODO: stopping condition?
# TODO: how to return two values? (beta, likelihoods)
# TODO: plot multiple data series on one plot

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

stochasticGradientDescent <- function(y, X, m, max_iterations, convergence_threshold) {
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
  
  # fix a small step size
  stepsize = 0.1
  
  # create vector to keep track negative log-likelihood values
  i = 1
  full_likelihoods = vector(mode = "list", length = max_iterations + 1)
  full_likelihoods[[i]] = computeNll(beta, y, X, m)
  
  # bogus value to start the loop
  sample_likelihoods = vector(mode = "list", length = max_iterations + 1)
  avg_sample_likelihoods = vector(mode = "list", length = max_iterations + 1)
  burn_period = 1000
  
  while (i <= max_iterations) {
    # sample a single data point (with replacement)
    index = sample(1:nrow(X), 1)
    
    # update beta
    gradient = computeGradient(beta, y[index], t(X[index, ]), m[index])
    beta = beta - stepsize * gradient
    
    # get new negative log-likelihood
    i = i + 1
    full_likelihoods[[i]] <- computeNll(beta, y, X, m)
    sample_likelihoods[[i]] <- computeNll(beta, y[index], t(X[index, ]), m[index])
    
    # update moving average
    if (i == burn_period) {
      avg_sample_likelihoods[[i]] <- sample_likelihoods[[i]]
    } else if (i > burn_period) {
      num_samples = i - burn_period
      avg_sample_likelihoods[[i]] <- (num_samples * avg_sample_likelihoods[[i-1]] + sample_likelihoods[[i]]) / (num_samples + 1)
    }
    
    print(paste("iteration #", i, "likelihood =", full_likelihoods[[i]]))
  }
  
  plot(seq(i), full_likelihoods[1:i])
  plot(seq(i), nrow(y)*sample_likelihoods[1:i])
  plot(seq(i), avg_sample_likelihoods[1:i])
  
  return(beta)
}
