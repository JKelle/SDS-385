# gradient descent for fitting logistic regression
#
# import the WDBC dataset from
#     https://github.com/jgscott/SDS385/blob/master/data/wdbc.csv

# load y
y = as.matrix(wdbc[2])
y[y == "B"] <- 0
y[y == "M"] <- 1
y = as.matrix(as.numeric(y))

# load X
X = as.matrix(wdbc[3:12])
X = scale(X)

# m is all ones
m = matrix(1, nrow(y), 1)

computeGradient <- function(beta, y, X, m) {
  # computes the gradient of l(beta)
  # where l is the negative log likelihood of beta
  # given that yi ~ Binomial(mi, wi)
  # and wi = 1/(1 + exp(-xi' * beta))
  w = 1/(1 + exp(-(X %*% beta)))
  return(-t(X) %*% (y - m*w))
}

computeLogLikelihood <- function(beta, y, X, m) {
  # returns the negative log likelihood, minus some value constant in beta
  # (m - y)' X beta + m' log(1 + exp(X beta))
  Xbeta = X %*% beta
  return(t(m - y) %*% Xbeta + t(m) %*% log(1 + exp(Xbeta)))
}

computeStepSize <- function(beta, y, X, m, direction) {
  # implements backtracking
  stepsize = 1
  shrinkfactor = 0.5
  c = 1e-4
  
  # bogus values to start the loop
  lhs = 1
  rhs = 0
  
  while (lhs >= rhs) {
    stepsize = shrinkfactor * stepsize
    lhs = computeLogLikelihood(beta + stepsize*direction, y, X, m)
    rhs = computeLogLikelihood(beta, y, X, m) + c*stepsize*(t(computeGradient(beta, y, X, m)) %*% direction)
  }
  
  return(stepsize)
}

gradientDescent <- function(y, X, m) {
  # initialize beta to be all 1
  likelihoods = vector(mode = "list", length = 10000)
  i = 1
  
  convergence_threshold = 1e-5
  beta = matrix(1, ncol(X), 1)
  cur_likelihood = computeLogLikelihood(beta, y, X, m)
  likelihoods[[i]] <- cur_likelihood
  # bogus value to start the loop
  prev_likelihood = cur_likelihood + convergence_threshold + 1
  
  #print(paste("current log likelihood", cur_likelihood))
  while (prev_likelihood - cur_likelihood > convergence_threshold) {
    # update beta
    direction = -computeGradient(beta, y, X, m)
    stepsize = computeStepSize(beta, y, X, m, direction)
    beta = beta + stepsize * direction
    
    # get new log likelihood
    prev_likelihood = cur_likelihood
    cur_likelihood = computeLogLikelihood(beta, y, X, m)
    i = i + 1
    likelihoods[[i]] <- cur_likelihood
    
    print(paste("iteration #", i, "likelihood =", cur_likelihood))
  }
  
  plot(seq(i), likelihoods[1:i])
}

gradientDescent(y, X, m)
