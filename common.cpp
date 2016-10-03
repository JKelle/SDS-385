#include <Rcpp.h>
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::MappedSparseMatrix;

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
		const Map<MatrixXd> beta,
		const Map<VectorXd> y,
		const MappedSparseMatrix<double> X,
		const Map<VectorXd> m) {
	VectorXd Xbeta = X * beta;
	VectorXd exp_term = (1.0 + (-1.0 * Xbeta.array()).exp()).log();
	double nll = (m - y).dot(Xbeta) + m.dot(exp_term);
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
// [[Rcpp::export]]
ArrayXd computeGradient_cpp(
		const Map<VectorXd> beta,
		const Map<ArrayXd> y,
		const MappedSparseMatrix<double> X,
		const Map<ArrayXd> m) {
	ArrayXd w = 1.0 / (1.0 + (-X * beta).array().exp());
	VectorXd err = m * w - y;
	ArrayXd gradient = X.transpose() * err;
	return gradient;
}

/**
 * AdaGrad update for SGD.
 * Returns both the updated beta vector and the updated hessian_approx vector.
 */
// [[Rcpp::export]]
Rcpp::List AdaGradUpdate_cpp(
		const Map<VectorXd> beta,
		const Map<ArrayXd> single_y,
		const MappedSparseMatrix<double> single_X,
		const Map<ArrayXd> single_m,
		const Map<ArrayXd> hessian_approx,
		double learning_rate) {
	ArrayXd gradient = computeGradient_cpp(beta, single_y, single_X, single_m);
	ArrayXd updated_hessian = hessian_approx + gradient.square();
	ArrayXd updated_beta = beta.array() - learning_rate * gradient / (updated_hessian + 1e-8).sqrt();

	return Rcpp::List::create(
		Rcpp::Named("hessian_approx") = updated_hessian,
		Rcpp::Named("beta") = updated_beta);
}
