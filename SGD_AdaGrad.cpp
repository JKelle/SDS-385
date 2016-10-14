#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
#include <cmath>
#include <ctime>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MappedSparseMatrix;
using Eigen::SparseMatrix;
using Eigen::RowMajor;

/*
 * Fast inverse square root approximation.
 * Copy/Pasted from Wikipedia:
 *   https://en.wikipedia.org/wiki/Fast_inverse_square_root
 */
union cast_double{ uint64_t asLong; double asDouble; };
static inline double invSqrt( const double& x )
{ //Stolen from physicsforums
  cast_double caster;
  caster.asDouble = x;
  double xhalf = ( double )0.5 * caster.asDouble;
  caster.asLong = 0x5fe6ec85e7de30daLL - ( caster.asLong >> 1 );
  double y = caster.asDouble;
  y = y * ( ( double )1.5 - xhalf * y * y );
  y = y * ( ( double )1.5 - xhalf * y * y ); //For better accuracy

  return y;
}

/**
 * Computes the negative log likelihood of beta given X, y, and m, (minus some
 * value constant in beta).
 * Implements the expresion: (m - y)' X beta + m' log(1 + exp(-X beta))
 *
 * Assumes yi ~ Binomial(mi, wi), where wi = 1/(1 + exp(-xi' * beta)).
 *
 * Args:
 *   beta: the weight vector
 *   y: the vector of labels
 *   X: the feature matrix (each row is one data point)
 *   m: the vector of number of trials for each data point
 */
// [[Rcpp::export]]
double computeNll_cpp(
		const VectorXd& beta,
		const VectorXd& y,
		const SparseMatrix<double, RowMajor>& X,
		const VectorXd& m) {
	VectorXd Xbeta = X * beta;
	VectorXd exp_term = (1.0 + (-1.0 * Xbeta.array()).exp()).log();
	double nll = (m - y).dot(Xbeta) + m.dot(exp_term);
	return nll;
}

/**
 * Computes the negative log likelihood of beta for a single data point (minus
 * some value constant in beta).
 *
 * Assumes yi ~ Binomial(mi, wi), where wi = 1/(1 + exp(-xi' * beta)).
 *
 * Args:
 *   beta: the weight vector
 *   y: the vector of labels
 *   X: the feature matrix (each row is one data point)
 *   m: the vector of number of trials for each data point
 *   index: the row index into y, X, and m.
 */
double computeNll_single_cpp(
		const VectorXd& beta,
		const VectorXd& y,
		const SparseMatrix<double, RowMajor>& X,
		const VectorXd& m,
		const int index) {

	double single_y = y(index);
	double single_m = m(index);

	double xBeta = 0;
	for (SparseMatrix<double, RowMajor>::InnerIterator it(X, index); it; ++it) {
		xBeta += it.value() * beta(it.index());
	}

	// small xBeta (large negative) leads to overflow, ie. exp(-xBeta) -> inf
	// So I make this approximation: log(1 + exp(-xBeta)) -> -xBeta
	double right;
	if (xBeta > -20) {
		// true value
		right = single_m * std::log(1 + std::exp(-xBeta));
	} else {
		// approximation
		right = - single_m * xBeta;
	}

	double left = (single_m - single_y) * xBeta;
	double nll = left + right;

	return nll;
}

/**
 * Computes the gradient of the negative log-likelihood function.
 * Implements the expression: X' (y_hat - y)
 *
 * Args:
 *   beta: the weight vector
 *   y: the vector of labels
 *   X: the feature matrix (each row is one data point)
 *   m: the vector of number of trials for each data point
 */
SparseMatrix<double> computeGradient_cpp(
		const VectorXd& beta,
		const double single_y,
		const SparseMatrix<double, RowMajor>& X,
		const double single_m,
		const int index) {
	double exponent = 0;
	int nnz = 0;
	for (SparseMatrix<double, RowMajor>::InnerIterator it(X, index); it; ++it) {
		exponent += it.value() * beta(it.index());
		nnz++;
	}

	double w = 1 / (1 + std::exp(-exponent));
	double err = single_m * w - single_y;

	SparseMatrix<double> gradient(X.cols(), 1);
	gradient.reserve(nnz);
	for (SparseMatrix<double, RowMajor>::InnerIterator it(X, index); it; ++it) {
		gradient.insert(it.index(), 0) = it.value() * err;
	}

	return gradient;
}

/**
 * AdaGrad update for SGD.
 * Performs updats (beta and hessian_approx) in place, so no return value.
*/
void AdaGradUpdate_cpp(
		VectorXd& beta,
		const VectorXd& y,
		const SparseMatrix<double, RowMajor>& X,
		const VectorXd& m,
		VectorXd& hessian_approx,
		const double learning_rate,
		const int index) {
	// compute gradient vector
	SparseMatrix<double> gradient = computeGradient_cpp(beta, y(index), X, m(index), index);

	// update hessian in place
	// update beta in place
	for (SparseMatrix<double>::InnerIterator it(gradient, 0); it; ++it) {
		hessian_approx(it.index()) += std::pow(it.value(), 2);  // h <- h + g^2
		beta(it.index()) -= learning_rate * it.value() * invSqrt(hessian_approx(it.index()) + 1e-8);
	}
}

/**
 * Computes the inverse l2 norm of each feature.
 * There's a bug in here somewhere. Don't use yet.
 */
void computeScale(const MappedSparseMatrix<double>& X, VectorXd& invMags) {
	for (int col = 0; col < X.cols(); col++) {
		// compute magnitude (l2 norm)
		double sumOfSq = 0;
		for (SparseMatrix<double, RowMajor>::InnerIterator it(X, col); it; ++it) {
			sumOfSq += std::pow(it.value(), 2);
		}
		// std::cout << "col = " << col << " sumOfSq = " << sumOfSq << std::endl;
		invMags(col) = invSqrt(sumOfSq);
	}
}

void applyScale(SparseMatrix<double, RowMajor>& X, VectorXd& invMags) {
	for (int col = 0; col < X.cols(); col++) {
		// divide by magnitude (multiply by inverse of magnitude)
		std::cout << "inv mag = " << invMags(col);
		for (SparseMatrix<double, RowMajor>::InnerIterator it(X, col); it; ++it) {
			std::cout << "before " << it.value();
			X.coeffRef(it.index(), col) *= invMags(col);
			std::cout << "after " << it.value();
		}
	}
}

/**
 * Stochastic gradient descent with AdaGrad.
 *
 * Args:
 *   y: the vector of labels
 *   X: the feature matrix (each row is one data point)
 *   m: the vector of number of trials for each data point
 *   num_epochs: number of times the whole dataset is iterated through
 *   learning_rate: learning rate for AdaGrad algorithm
 *
 * Returns:
 *   a list with the following elements
 *   beta: regression coefficients
 *   sample_likelihoods: vector of negative log-likelihood of beta value for individual data points
 *   avg_likelihoods: vector of exponentially weighted running average of sample_likelihoods
 *   full_likelihoods: vector of the negative log-likelihood of beta given the entire dataset
 */
 // [[Rcpp::export]]
Rcpp::List SGD_AdaGrad_cpp(
		const Map<VectorXd> y,
		const MappedSparseMatrix<double> X_colmajor,
		const Map<VectorXd> m,
		int num_epochs,
		double learning_rate) {
	std::cout << "entering SGD_AdaGrad_cpp" << std::endl;

	// scale features to have unit l2 norm
	// VectorXd inv_feature_magnitudes = VectorXd::Zero(X_colmajor.cols());
	// computeScale(X_colmajor, inv_feature_magnitudes);

	// convert to RowMajor
	std::cout << "\tconverting to RowMajor" << std::endl;
	SparseMatrix<double, RowMajor> X(X_colmajor);
	std::cout << "X has " << X.rows() << " rows, " << X.cols() << " cols" << std::endl;

	// applyScale(X, inv_feature_magnitudes);

	// initialize beta to be all 0
	VectorXd beta = VectorXd::Zero(X.cols());

	// initialize estimate of diagonal approx to Hessian
	VectorXd hessian_approx(X.cols());
	for (int i = 0; i < hessian_approx.size(); ++i) {
		hessian_approx(i) = 0.001;
	}

	// create vector to keep track negative log-likelihood values
	// VectorXd sample_likelihoods = VectorXd::Zero(num_epochs * X.rows() + 1);
	// VectorXd avg_likelihoods = VectorXd::Zero(num_epochs * X.rows() + 1);
	// VectorXd full_likelihoods = VectorXd::Zero(num_epochs * X.rows() + 1);
	// int i = 0;

	while (num_epochs-- > 0) {

		std::cout << "\tbeginning epoch" << std::endl;
		std::clock_t begin = std::clock();

		for (int index = 0; index < X.rows(); index++) {
		// for (int index = 0; index < 100000; index++) {

			// update beta and hessian_approx in place
			AdaGradUpdate_cpp(beta, y, X, m, hessian_approx, learning_rate, index);

			// get new negative log-likelihood
			// sample_likelihoods(i) = computeNll_single_cpp(beta, y, X, m, index);
			// full_likelihoods(i) = computeNll_cpp(beta, y, X, m);

			// update exponentially weighted moving average
			// std::cout << "\t\tupdating running avg" << std::endl;
			// if (i == 0) {
			// 	avg_likelihoods(i) = sample_likelihoods(i);
			// } else {
			// 	avg_likelihoods(i) = 0.99 * avg_likelihoods(i-1) + 0.01 * sample_likelihoods(i);
			// }
			// if (i % 10 == 0) {
			// 	std::cout << i << " " << sample_likelihoods(i) << " " << avg_likelihoods(i) << std::endl;
			// }
			// i++;
		}

		std::clock_t end = std::clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "epoch: " << elapsed_secs << std::endl;
	}

	return Rcpp::List::create(
		Rcpp::Named("beta") = beta
		// Rcpp::Named("sample_likelihoods") = sample_likelihoods,
		// Rcpp::Named("full_likelihoods") = full_likelihoods,
		// Rcpp::Named("avg_likelihoods") = avg_likelihoods
	);
}
