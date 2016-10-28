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
