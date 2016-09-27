
# Author: Josh Kelle

# This script implements AdaGrad variant of  stochastic gradient descent (SGD)
# for minimizing the objective function (negative log likelihood) of a
# logistic regression model.

# import common math function
setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("common.R")

SGD_AdaGrad <- function(y, X, m, num_epochs, learning_rate) {
  # Stochastic gradient descent with AdaGrad.
  #
  # Args:
  #   y: the vector of labels
  #   X: the feature matrix (each row is one data point)
  #   m: the vector of number of trials for each data point
  #   num_epochs: number of times the whole dataset is iterated through
  #   learning_rate: learning rate for AdaGrad algorithm
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
  
  # initialize estimate of diagonal approx to Hessian
  hessian_approx = matrix(0.001, ncol(X))
  
  # set 'fudge factor' that avoids dividing by zero
  epsilon = 1e-8
  
  while (num_epochs > 0) {
    for (index in seq(nrow(X))) {
      # sample a single data point
      single_y = y[index]
      single_x = t(X[index, ])
      single_m = m[index]
      
      # update beta
      gradient = computeGradient(beta, single_y, single_x, single_m)
      hessian_approx = hessian_approx + gradient ^ 2
      beta = beta - learning_rate * gradient / sqrt(hessian_approx + epsilon)
      
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
      
      print(paste("iteration #", i, "likelihood =", avg_likelihoods[[i]]))
    }
    
    num_epochs = num_epochs - 1
  }
  
  return(list(beta=beta, avg_likelihoods=avg_likelihoods[1:i], sample_likelihoods=sample_likelihoods[1:i], full_likelihoods=full_likelihoods[1:i]))
}
