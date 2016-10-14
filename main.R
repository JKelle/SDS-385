
library(microbenchmark)
library('Matrix')
library("Rcpp")
library("RcppEigen")
library("readr")

setwd("~/Google Drive/University of Texas/SDS 385; Statistical Models for Big Data/code")
sourceCpp(file='SGD_AdaGrad.cpp')

# These .rds files are from Matteo's Piazza post
#X_train = readRDS("../data/URL_preprocessed/url_X_training.rds")
#y_train = readRDS("../data/URL_preprocessed/url_y_training.rds")
X_test = readRDS("../data/URL_preprocessed/url_X_test.rds")
y_test = readRDS("../data/URL_preprocessed/url_y_test.rds")

num_epochs = 1
learning_rate = 0.9

m = rep(1, length(y_test))

result = SGD_AdaGrad_cpp(y_test, X_test, m, num_epochs, learning_rate)
