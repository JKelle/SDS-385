# exercise 5
# part B

# take aways: optimal value of lambda increases as true sparsity increases

n = 10000
sparsity = 0.02
lambdas = seq(0.1, 3, 0.1)
theta = rbinom(n, 4, 0.5) * rbinom(n, 4, sparsity)
stdev = rep(1, n)
z = rnorm(n, mean=theta, sd=stdev)

estimator_func <- function(zi, lambda) {
  return(sign(zi) * max(abs(zi) - lambda, 0))
}

mse_func <- function(lambda) {
  theta_hat = mapply(estimator_func, z, lambda)
  return(sum((theta_hat - theta) ^ 2)/ length(theta))
}

mse = mapply(mse_func, lambdas)
plot(lambdas, mse)

# Lasso
# take away: in-sample mse increases as lambda increases. The fit gets worse.

library(glmnet)

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
X = data.matrix(read.csv("../data/diabetesX.csv", header=TRUE), rownames.force=FALSE)
y = data.matrix(read.csv("../data/diabetesY.csv", header=FALSE), rownames.force=FALSE)

model = glmnet(X, y)
plot(model, xvar="lambda")

mse = rep(0, length(model$lambda))
for (i in seq(length(model$lambda), 1)) {
  beta = as.matrix(model$beta[, i])
  mse[i] = sum((y - X %*% beta) ^ 2) / length(y)
}
plot(model$lambda, mse, log='x')

# cross validation
# take away: out-of-sample mse has a non-zero optimial value for lambda

n = length(y)
ntrain = round(0.8 * n)
trainX = X[1:ntrain, ]
trainY = y[1:ntrain]
testX = X[(ntrain + 1):n, ]
testY = y[(ntrain + 1):n]

model = glmnet(trainX, trainY)
predictY = predict(model, testX)

mse = rep(0, length(model$lambda))
for (i in seq(length(model$lambda), 1)) {
  y_hat = as.matrix(predictY[, i])
  mse[i] = sum((testY - y_hat) ^ 2) / n
}
plot(model$lambda, mse, log='x')

# Cp statistic

mse = rep(0, length(model$lambda))
for (i in seq(length(model$lambda), 1)) {
  beta = as.matrix(model$beta[, i])
  mse = sum((y - X %*% beta) ^ 2) / n
  
}
plot(model$lambda, mse, log='x')