
# Author: Josh Kelle

# This script implements "mini-batch linesearch" variant of  stochastic
# gradient descent (SGD) for minimizing the objective function (negative log
# likelihood) of a logistic regression model.

# import common math functions
setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

SGD_linesearch <- function(y, X, m, num_epochs, mini_batch_size, linesearch_freq) {
  # Stochastic gradient descent with linesearch to select step size.
  # Line search is invoked on a mini batch of data points. The returned
  # stepsize is used for 100 iterations before recomputing a new stepsize.
  #
  # Args:
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  #   num_epochs: number of times the whole dataset is iterated through
  #   mini_batch_size: number of data points used during line search
  #   linesearch_freq: frequency with which a new stepsize is computed
  #
  # Returns:
  #   a list with the following elements
  #   beta: regression coefficients
  #   sample_likelihoods: vector of negative log-likelihood of beta value for individual data points
  #   avg_likelihoods: vector of exponentially weighted running average of sample_likelihoods
  #   full_likelihoods: vector of the negative log-likelihood of beta given the entire dataset
  
  # initialize beta to be all 0
  beta = matrix(0, ncol(X))
  
  # create vector to keep track negative log-likelihood values
  max_iterations = num_epochs * nrow(X)
  sample_likelihoods = vector(mode = "numeric", length = max_iterations + 1)
  avg_likelihoods = vector(mode = "numeric", length = max_iterations + 1)
  full_likelihoods = vector(mode = "numeric", length = max_iterations + 1)
  i = 0
  
  while (num_epochs > 0) {
    for (index in seq(nrow(X))) {
      
      # set stepsize for a minibatch every 100 iterations
      if (i %% linesearch_freq == 0) {
        indices = sample(1:nrow(X), mini_batch_size)
        mini_batch_X = X[indices, ]
        mini_batch_y = y[indices]
        mini_batch_m = m[indices]
        mini_batch_gradient = computeGradient(beta, mini_batch_y, mini_batch_X, mini_batch_m)
        stepsize = linesearch(beta, mini_batch_y, mini_batch_X, mini_batch_m, -mini_batch_gradient)
      }
      
      # sample a single data point
      single_y = y[index]
      single_x = t(X[index, ])
      single_m = m[index]
      
      # update beta
      gradient = computeGradient(beta, single_y, single_x, single_m)
      beta = beta - stepsize * gradient
      
      # get new negative log-likelihood
      i = i + 1
      sample_likelihoods[[i]] <- computeNll(beta, single_y, single_x, single_m)
      full_likelihoods[[i]] <- computeNll(beta, y, X, m)
      
      # update exponentially weighted moving average
      if (i == 1) {
        avg_likelihoods[[i]] <- sample_likelihoods[[i]]
      } else {
        avg_likelihoods[[i]] <- 0.99 * avg_likelihoods[[i-1]] + 0.01 * sample_likelihoods[[i]]
      }
      
      #print(paste("iteration #", i, "likelihood =", avg_likelihoods[[i]]))
    }
    
    num_epochs = num_epochs - 1
  }
  
  return(list(beta=beta, avg_likelihoods=avg_likelihoods[1:i], sample_likelihoods=sample_likelihoods[1:i], full_likelihoods=full_likelihoods[1:i]))
}
