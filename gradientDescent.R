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


# when using augmented data matrix (extra columns of ones), when step size is large,
# [1] "iteration # 1 likelihood = 787.198735495068"
# [1] "step size = 1.1972515182562"
# [1] "iteration # 2 likelihood = 642.00621606391"
# [1] "step size = 0.0955004950796826"
# [1] "iteration # 3 likelihood = -21145.2296902593"
# [1] "step size = 3.6299899369485e-15"
# [1] "iteration # 4 likelihood = -12898.2483175053"

computeStepSize <- function(beta, y, X, m, direction) {
  # implements backtracking
  stepsize = 0.01
  shrinkfactor = 0.5
  c = 1e-4
  
  # bogus values to start the loop
  lhs = 1
  rhs = 0
  
  while (lhs > rhs) {
    stepsize = shrinkfactor * stepsize
    lhs = computeLogLikelihood(beta + stepsize*direction, y, X, m)
    rhs = computeLogLikelihood(beta, y, X, m) - c*stepsize*(t(direction) %*% direction)
  }
  
  print(paste("step size =", stepsize))
  return(stepsize)
}

gradientDescent <- function(y, X, m) {
  # initialize beta to be all 1
  likelihoods = vector(mode = "list", length = 10000)
  i = 1
  
  convergence_threshold = 1e-5
  beta = matrix(1, ncol(X), 1)
  cur_likelihood = computeLogLikelihood(beta, y, X, m)
  likelihoods[[i]] = cur_likelihood
  # bogus value to start the loop
  prev_likelihood = cur_likelihood + convergence_threshold + 1
  
  print(paste("iteration #", i, "likelihood =", cur_likelihood))
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
  return(beta)
}

beta = gradientDescent(y, X, m)

# test accuracy
probs = 1/(1 + exp(-X %*% beta))
plot(probs)
y[probs < 0.5]
sum(y[probs < 0.5])
y[probs > 0.5]
sum(1 - y[probs > 0.5])

# now add column of ones
X2 = cbind(X, matrix(1, nrow(X), 1))
beta = gradientDescent(y, X2, m)

# test accuracy
probs2 = 1/(1 + exp(-X2 %*% beta))
plot(probs2)
y[probs2 < 0.5]
sum(y[probs2 < 0.5])
y[probs2 > 0.5]
sum(1 - y[probs2 > 0.5])
