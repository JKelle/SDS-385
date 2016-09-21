# Author: Josh Kelle
#
# This script compares various algorithms for fitting parameters
# of a logistic regression model.
#
# import the WDBC dataset from
#     https://github.com/jgscott/SDS385/blob/master/data/wdbc.csv
#
## TODO
# try diff levels of sparsity
# gradientDecent(..., maxIteration, )
# change convergent criterion to percent difference
# compare to glm

library(microbenchmark)
library('Matrix')

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
source("gradientDescent.R")
source("newtonMethod.R")


##################
#  loading data  #
##################

# loading data
wdbc <- read.csv("../data/wdbc.csv", header=FALSE)

# load y
y = matrix(as.numeric(wdbc[, 2]) - 1)

# load X
X = as.matrix(wdbc[3:12])
X = scale(X)
X = cbind(X, matrix(1, nrow(X), 1))
# Use Matrix (with a capital M) to handle sparce vectors and matrices
#X = Matrix(X)

# m is all ones
m = matrix(1, nrow(y), 1)


#####################
#  gradientDescent  #
#####################

max_iterations = 10000
convergence_threshold = 1e-6

# gd = gradient descent
list[gd_beta, gd_likelihoods] = gradientDescent(y, X, m, 10000, 1e-6)

# nm = newton method
list[nm_beta, nm_likelihoods] = newtonMethod(y, X, m, 10000, 1e-6)

# glm = R's built-in glm (generalized linear model) function 
glm1 = glm(y~X, family='binomial')
glm_beta = glm1$coefficients


##############################
#  test classifier accuracy  #
##############################

# test accuracy
probs = 1/(1 + exp(-X %*% beta))
plot(probs)
y[probs < 0.5]
sum(y[probs < 0.5])
y[probs > 0.5]
sum(1 - y[probs > 0.5])

# now add column of ones
X2 = cbind(X, matrix(1, nrow(X), 1))
beta = gradientDescent(y, X2, m, 10000, 1e-6)

# test accuracy
probs2 = 1/(1 + exp(-X2 %*% beta))
plot(probs2)
y[probs2 < 0.5]
sum(y[probs2 < 0.5])
y[probs2 > 0.5]
sum(1 - y[probs2 > 0.5])
