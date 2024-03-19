#include <Rcpp.h>
#include <Rmath.h>
#include <cmath>
using namespace Rcpp;

//' Compute the first four (central) moments
//' 
//' `mom()` compute the first moment, the second to fourth central moments.
//' 
//' @param samples A vector of the samples.
//' @returns A vector of the first four (central) moments. 
//'   \deqn{(E[x],E[(x-E[x])^2],E[(x-E[x])^3],E[(x-E[x])^4])}
//' @examples
//' mom(rnorm(100,0,1))
//' 
//' @export
//' 
// [[Rcpp::export]]
NumericVector mom(NumericVector samples) {
  NumericVector samples_new;
  LogicalVector lv = Rcpp::is_na(samples);
  int n;
  // a little weird is_true(any(lv))
  if (Rcpp::is_true(Rcpp::any(lv))) {
    n = Rcpp::sum(lv);
    Rprintf("Simulated %i samples contain %i NA(s)! ", samples.length(), n);
    Rprintf("Moments are computed without them!\n");
    samples_new = samples[!lv];
  } else {
    samples_new = samples;
  }
  double m1  = Rcpp::mean(samples_new);
  double cm2 = Rcpp::mean(Rcpp::pow(samples_new-m1, 2));
  double cm3 = Rcpp::mean(Rcpp::pow(samples_new-m1, 3));
  double cm4 = Rcpp::mean(Rcpp::pow(samples_new-m1, 4));
  //
  NumericVector moments = {m1, cm2, cm3, cm4};
  return moments;
}

//' Evaluate the Integrand Function
//' 
//' Compute the integrand function of the model which is described by the 
//' following equation:
//' \deqn{
//'   \beta(\alpha) = 1 - e^{\int_{-\infty}^{\alpha}q(x)dx},
//' }
//' where \eqn{\alpha \sim \mathcal{N}(\mu,\sigma^2)},
//' the integrand is
//' \deqn{
//'   q(x) \equiv \frac{g(N-1)F^{N-2}(x)f(x)x}{(w_0 + gF^{N-1}(x))(x-1)}.
//' }
//' and \eqn{f(x)} and \eqn{F(x)} are the PDF and CDF of the normal 
//' distribution \eqn{\mathcal{N}(\mu,\sigma^2)}, respectively.
//' 
//' @param x A scalar variable.
//' @param g The parameter \eqn{g}.
//' @param N The parameter \eqn{N}, it should be larger than 2.
//' @param mu The parameter \eqn{\mu}, mean of the normal distribution.
//' @param sigma The parameter \eqn{\sigma}, standard deviation of the normal
//' distribution.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A scalar value.
//' 
//' @examples
//' q(0.1, 0.2, 3.5, 0.5, 0.1)
//' 
//' @export
//' 
// [[Rcpp::export]]
double q(double x, double g, double N, double mu, double sigma, 
         double w0=0.1) {
  if (x >= 1) {
    stop("x (=%.2f) in q(x,g,N,mu,sigma) must < 1", x);
  }
  double p, d, num, den, val;
  p = R::pnorm(x, mu, sigma, true, false);
  d = R::dnorm(x, mu, sigma, false);
  num = g*(N-1)*std::pow(p, N-2)*d*x;   // to avoid Inf * 0, it requires N > 2
  den = (w0 + g*std::pow(p, N-1))*(x-1);// to avoid x-1 = 0, it requires x < 1
  val = num/den;
  // counter-intuitive of the following is_na()
  if (Rcpp::NumericVector::is_na(val)) {
    Rcpp::warning("q(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) returns NA!", 
                  x,g,N,mu,sigma,w0);
  }
  return val;
}

//' Evaluate the Integral Numerically
//' 
//' Evaluate the integral with given upper and lower bounds by using 
//' trapezoidal rule.
//' 
//' @param g The parameter \eqn{g}.
//' @param N The parameter \eqn{N}, it should be larger than 2.
//' @param mu The parameter \eqn{\mu}, mean of the normal distribution.
//' @param sigma The parameter \eqn{\sigma}, standard deviation of the normal
//' distribution.
//' @param ub Upper bound of the integral, \eqn{\int_{lb}^{ub}q(x)dx}.
//' @param lb Lower bound of the integral.
//' @param n Segments to divide when computing the integral numerically.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A scalar value.
//' @seealso [q()]
//' @examples
//' intq(0.2, 3.5, 0.5, 0.1, 0.4, 0)
//' 
//' @export
//' 
// [[Rcpp::export]]
double intq(double g, double N, double mu, double sigma, 
            double ub, double lb, int n=10, double w0=0.1) {
  double len = ub - lb, x, val;
  // trapezoidal rule to compute the integral numerically
  val = q(lb, g,N,mu,sigma,w0)/2;
  for (int k=1; k < n; k++) {
    x = lb + (k/n) * len;
    val += q(x, g, N, mu, sigma, w0);
  }
  val += q(ub, g,N,mu,sigma,w0)/2;
  //
  val = val * (len/n);
  return val;
}


//' Simulate Samples of Variable Y
//' 
//' Simulate samples of the variable
//' \deqn{
//'   Y \equiv \log(1 - \beta(\alpha)), Y = \int_{-\infty}^X q(x) dx.
//' }
//' where \eqn{\quad X \equiv \alpha}.
//' 
//' @param g The parameter \eqn{g}.
//' @param N The parameter \eqn{N}, it should be larger than 2.
//' @param mu The parameter \eqn{\mu}, mean of the normal distribution.
//' @param sigma The parameter \eqn{\sigma}, standard deviation of the normal
//' distribution.
//' @param L Length of the samples to generate.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A vector of simulated \eqn{Y} samples.
//' @examples
//' sim_Y(0.2, 3.5, 0.5, 0.1, 100)
//' 
//' @export
//' 
// [[Rcpp::export]]
NumericVector sim_Y(double g, double N, double mu, double sigma, 
                    int L, double w0=0.1) {
  NumericVector Y (L);
  LogicalVector lv1 (L), lv2 (L);
  int n1, n2;
  //
  NumericVector X = Rcpp::rnorm(L, mu, sigma);
  X = X.sort();
  lv1 = X >= 1;
  lv2 = X <= 0;
  n1 = Rcpp::sum(lv1);
  n2 = Rcpp::sum(lv2);
  //
  if (n1 >= 1) {
    warning("There are %i variable(s) >= 1 within the simulated %i X!", n1, L);
    X[lv1] = 0.999;// may error! Change them to 0.999!
  }
  //
  if (n2 >= 1) {
    warning("There are %i variable(s) <= 0 within the simulated %i X!", n2, L);
    X[lv2] = 0.001;// may error! Change them to 0.001!
  }
  //
  double lb = X[0];                      // smallest X
  Y[0] = intq(g, N, mu, sigma, lb, lb-1);// int_{lb-1}^{lb}q(x)dx
  //
  for (int i=1; i < L; i++) {
    Y[i] = Y[i-1] + intq(g, N, mu, sigma, X[i], X[i-1]);
  }
  //
  return Y;
}

//' Evaluate the Objective Function
//' 
//' Evaluate the objective function, i.e., the error function. First, generate
//' a sequence of samples by simulation under the given parameters. Then,
//' compute the first four (central) moments of the simulated samples. Last,
//' compute the differences between the true sample moments and the simulated
//' sample moments.
//' 
//' @param g The parameter \eqn{g}.
//' @param N The parameter \eqn{N}, it should be larger than 2.
//' @param mu The parameter \eqn{\mu}, mean of the normal distribution.
//' @param sigma The parameter \eqn{\sigma}, standard deviation of the normal
//' distribution.
//' @param L Length of the samples to generate.
//' @param moments A vector of the true sample moments, 1st moment,
//' 2nd - 4th central moments.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A scalar value.
//' @examples
//' Y = sim_Y(0.2, 3.5, 0.5, 0.1, 1000)
//' moments = mom(Y)
//' f_obj(0.3, 3, 0.4, 0.05, 1000, moments)
//' 
//' @export
//' 
// [[Rcpp::export]]
double f_obj(double g, double N, double mu, double sigma, int L, 
             NumericVector moments, double w0=0.1) {
  NumericVector Y = sim_Y(g, N, mu, sigma, L, w0);
  NumericVector smoments = mom(Y);
  double error = Rcpp::sum(Rcpp::pow(smoments-moments, 2));
  return error;
}

//' Evaluate Gradient of the Objective Function
//' 
//' Evaluate gradient of the objective function by Finite Difference (FD).
//' 
//' @param g The parameter \eqn{g}.
//' @param N The parameter \eqn{N}, it should be larger than 2.
//' @param mu The parameter \eqn{\mu}, mean of the normal distribution.
//' @param sigma The parameter \eqn{\sigma}, standard deviation of the normal
//' distribution.
//' @param L Length of the samples to generate.
//' @param moments A vector of the true sample moments, 1st moment,
//' 2nd - 4th central moments.
//' @param stepsize A vector of four scalars, step size for each of the 
//' four parameters.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A vector of four partial derivatives.
//' @examples
//' Y = sim_Y(0.2, 3.5, 0.5, 0.1, 1000)
//' moments = mom(Y)
//' stepsize = rep(0.01, 4)
//' df_obj(0.3, 3, 0.4, 0.05, 1000, moments, stepsize)
//' 
//' @export
//' 
// [[Rcpp::export]]
NumericVector df_obj(double g, double N, double mu, double sigma, int L,
                     NumericVector moments, NumericVector stepsize,
                     double w0=0.1) {
  NumericVector df (4);
  double f1, f2, h;
  //
  h = stepsize[0];
  f1 = f_obj(g-h, N, mu, sigma, L, moments, w0);
  f2 = f_obj(g+h, N, mu, sigma, L, moments, w0);
  df[0] = (f2-f1)/(2*h);
  //
  h = stepsize[1];
  f1 = f_obj(g, N-h, mu, sigma, L, moments, w0);
  f2 = f_obj(g, N+h, mu, sigma, L, moments, w0);
  df[1] = (f2 - f1) / (2*h);
  //
  h = stepsize[2];
  f1 = f_obj(g, N, mu-h, sigma, L, moments, w0);
  f2 = f_obj(g, N, mu+h, sigma, L, moments, w0);
  df[2] = (f2 - f1) / (2*h);
  //
  h = stepsize[3];
  f1 = f_obj(g, N, mu, sigma-h, L, moments, w0);
  f2 = f_obj(g, N, mu, sigma+h, L, moments, w0);
  df[3] = (f2 - f1) / (2*h);
  //
  return df;
}

//' Evaluate Single Partial Derivative
//' 
//' Evaluate single partial derivative of the objective function by
//' Finite Difference (FD).
//' 
//' @param i An integer, 1,2,3,4 corresponding to \eqn{g, N, \mu, \sigma}
//' respectively.
//' @param theta A vector of the parameters \eqn{g, N, \mu, \sigma}.
//' @param L Length of the samples to generate.
//' @param moments A vector of the true sample moments, 1st moment,
//' 2nd - 4th central moments.
//' @param h A scalar, step size for the parameter change.
//' @param G A integer, default to 10, number of replications.
//' @param w0 The parameter \eqn{w_0}, default to 0.1.
//' 
//' @returns A scalar.
//' @examples
//' theta = c(0.3, 3, 0.4, 0.05)
//' Y = sim_Y(0.2, 3.5, 0.5, 0.1, 1000)
//' moments = mom(Y)
//' df_obj2(2, theta, 1000, moments)
//' 
// [[Rcpp::export]]
double df_obj2(int i, NumericVector theta, int L, NumericVector moments,
               double h=0.01, int G=10, double w0=0.1) {
  NumericVector f_va1 (G);
  NumericVector f_va2 (G);
  NumericVector delta (4);
  delta[i-1] = h;
  NumericVector thetn (4);
  thetn = theta + delta;
  //
  for (int n=0; n < G; n++) {
    f_va1[n] = f_obj(theta[0], theta[1], theta[2], theta[3], L, moments);
    f_va2[n] = f_obj(thetn[0], thetn[1], thetn[2], thetn[3], L, moments);
  }
  double df = Rcpp::mean(f_va2 - f_va1) / h;
  return df;
}

//' Project Parameters back to the Feasible Space
//' 
//' Project parameters back to the feasible space.
//' 
//' @param theta A vector of the parameters.
//' @param Theta A matrix each row of which have two scalars, 
//' first: lower bound, second: upper bound.
//' 
//' @returns A vector.
//' @examples
//' theta = c(-0.3, 3, 0.4, 0.05)
//' Theta = matrix(c(0,10, 2,10, 0.1,0.9, 0.01,0.4), 
//'                  nrow=4, ncol=2, byrow=TRUE)
//' project(theta, Theta)
//' 
//' @export
//' 
// [[Rcpp::export]]
NumericVector project(NumericVector theta, NumericMatrix Theta) {
  if (Rcpp::any(Rcpp::is_na(theta))) {
    Rcpp::warning("theta contains NA!");
  }
  for (int i=0; i < theta.length(); i++) {
    if (theta[i] < Theta(i,0)) {
      theta[i] = Theta(i,0);
    } else if (theta[i] > Theta(i,1)) {
      theta[i] = Theta(i,1);
    }
  }
  return theta;
}

//' More Fine Projection
//' 
//' Fine tune the mean and standard deviation of the normal distribution,
//' considering that it should not produce random variable smaller than 0,
//' or larger than 1.
//' 
//' @param theta A vector of the parameters.
//' @param Theta A matrix each row of which have two scalars, 
//' first: lower bound, second: upper bound.
//' 
//' @returns A vector.
//' @examples
//' theta = c(-0.3, 3, 0.4, 1)
//' Theta = matrix(c(0,10, 2,10, 0.1,0.9, 0.01,0.4), 
//'                  nrow=4, ncol=2, byrow=TRUE)
//' project2(theta, Theta)
//' 
//' @export
//' 
// [[Rcpp::export]]
NumericVector project2(NumericVector theta, NumericMatrix Theta) {
  theta = project(theta, Theta);
  double mu = theta[2];
  double sd = theta[3];
  //
  if (R::pnorm(1, mu, sd, true, false) < 0.9999) {
    Rcpp::warning("too close to 1! decrease sigma");
    theta[3] = (1-mu)/4;
  }
  if (R::pnorm(0, mu, sd, true, false) > 0.0001) {
    Rcpp::warning("too close to 0! decrease sigma");
    theta[3] = (mu-0)/4;
  }
  return theta;
}

//' Stochastic Approximation
//' 
//' Search the optimal parameters by Stochastic Approximation (SA). That is
//' to find the parameters that minimize the objective (error) function.
//' 
//' @param theta0 A vector of four scalars, i.e., the initial guess of the
//' parameters' value.
//' @param Theta A matrix of \eqn{4\times 2}, feasible space of the parameters.
//' Each row have two scalars, first: lower bound, second: upper bound.
//' @param stepsize A vector of four scalars, step size for each parameter.
//' @param L An integer, sample length used for each simulation embedded in the
//' SA searching.
//' @param moments A vector of the true sample moments, 1st moment,
//' 2nd - 4th central moments.
//' @param G An integer, number of searching iterations.
//' 
//' @returns A matrix of the search trajectory, the first four columns
//' corresponding to the four parameters, and the last column corresponding
//' to the objective function value.
//' @examples
//' Y_sim = sim_Y(0.2, 3.5, 0.5, 0.1, 1000)    # 仿真生成1,000个模型样本
//' L = length(Y_sim)  
//' moments = mom(Y_sim)                       # 统计出一阶矩，2-4阶中心矩
//' lbub = c(0.1,5, 2.1,10, 0.1,0.9, 0.01,0.4) # 参数搜索范围
//' # 表示：0.1 <g< 5, 2.1 <N< 10, 0.1 <mu< 0.9, 0.01 <sigma< 0.4
//' Theta = matrix(lbub, nrow=4, ncol=2, byrow=TRUE) # 从数列转换成矩阵
//' # SA 搜索，限定搜索20步，步长固定为0.01（与1/k序列不一样）
//' theta0 = c(0.25, 3.0, 0.4, 0.05)           # 初始参数值 (g, N, mu, sigma)
//' stepsize = rep(0.01, 4)                    # 步长固定为0.01
//' theta_f_record = sa(theta0, Theta, stepsize, L, moments, G=20)
//' 
// [[Rcpp::export]]
NumericMatrix sa(NumericVector theta0, NumericMatrix Theta, 
                 NumericVector stepsize,
                 int L, NumericVector moments, int G=200) {
  NumericMatrix theta_f_record(G+1, 5);
  NumericVector df (4);
  double f;
  //
  f = f_obj(theta0[0], theta0[1], theta0[2], theta0[3], L, moments);
  theta_f_record(0,0) = theta0[0];
  theta_f_record(0,1) = theta0[1];
  theta_f_record(0,2) = theta0[2];
  theta_f_record(0,3) = theta0[3];
  theta_f_record(0,4) = f;
  //
  Rprintf("%i-th iter: %.2f, %.2f, %.2f, %.2f\t", 0, 
          theta0[0], theta0[1], theta0[2], theta0[3]);
  Rprintf("f_obj: %.6f\n\n", f);
  //
  for (int k=1; k <= G; k++) {
    df = df_obj(theta0[0],theta0[1],theta0[2],theta0[3],L,moments,stepsize);
    for (int n=0; n < theta0.length(); n++) {
      if (df[n] > 0.0001) {        // 0.0001 <-> 0
        theta0[n] -= stepsize[n];// theta0[n] -= 0.01;
      } else if (df[n] < 0.0001) { // 0.0001 <-> 0
        theta0[n] += stepsize[n];// theta0[n] += 0.01;
      }
    }
    //
    theta0 = project2(theta0, Theta);
    f = f_obj(theta0[0], theta0[1], theta0[2], theta0[3], L, moments);
    //
    theta_f_record(k,0) = theta0[0];
    theta_f_record(k,1) = theta0[1];
    theta_f_record(k,2) = theta0[2];
    theta_f_record(k,3) = theta0[3];
    theta_f_record(k,4) = f;
    // 
    if (k % G == 0) {
      Rprintf("%i-th iter: %.2f, %.2f, %.2f, %.2f\t", k, 
              theta0[0], theta0[1], theta0[2], theta0[3]);
      Rprintf("f_obj: %.6f\n\n", f);
    }
  }
  return theta_f_record;
}
