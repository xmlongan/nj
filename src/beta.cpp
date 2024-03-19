#include <Rcpp.h>
#include <Rmath.h>
#include <cmath>

#include "core.h"

using namespace Rcpp;

//' Evalute the Integrand Function
//' 
//' Analog to [q()], but here the model is set with beta distribution.
//' 
//' @seealso [q()]
//' 
// [[Rcpp::export]]
double q_beta(double x, double g, double N, double a, double b, 
              double w0=0.1) {
  // to avoid Inf * 0, it requires 0 < x < 1
  if (x > 0.9999) {
    x = 0.9999;
  } else if (x < 0.0001) {
    x = 0.0001;
  }
  //
  double p = R::pbeta(x, a, b, true, false);
  double d = R::dbeta(x, a, b, false);
  // to avoid Inf * 0, it requires N > 2
  double num = g*(N-1)*std::pow(p, N-2)*d*x;
  // to avoid x - 1 = 0, it requires x < 1
  double den = (w0 + g*std::pow(p, N-1))*(x-1);
  double val = num/den;
  // counter-intuitive of the following is_na()
  if (Rcpp::NumericVector::is_na(val)) {
    Rcpp::warning("q_beta(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) returns NA!", 
                  x,g,N,a,b,w0);
  }
  return val;
}

//' Evaluate the Integral Numerically
//' 
//' Analog to [intq()], but here the model is set with beta distribution.
//' 
//' @seealso [intq()]
//' 
// [[Rcpp::export]]
double intq_beta(double g, double N, double a, double b, 
                 double ub, double lb, int n=10, double w0=0.1) {
  double len = ub - lb, x, val;
  val = q_beta(lb,g,N,a,b,w0)/2;
  // trapezoidal rule to compute the integral numerically
  for (int k=1; k < n; k++) {
    x = lb + (k/n) * len;
    val += q_beta(x, g, N, a, b, w0);
  }
  val += q_beta(ub,g,N,a,b,w0)/2;
  val = val * (len/n);
  return val;
}

//' Simulate Samples of Variable Y
//' 
//' Analog to [sim_Y()], but here the model is set with beta distribution.
//' 
//' @seealso [sim_Y()]
//' 
// [[Rcpp::export]]
NumericVector sim_Y_beta(double g, double N, double a, double b, int L,
                         double w0=0.1) {
  NumericVector Y (L);
  NumericVector X (L);
  LogicalVector lv (L);
  int n;
  //
  X = Rcpp::rbeta(L, a, b);
  X = X.sort();
  // It is for sure that every x in X is in (0,1)
  lv = X >= 0.9999;
  n = Rcpp::sum(lv);
  if (n >= 1) {
    warning("There are %i variable(s) >= 0.9999 within the simulated %i X!", 
            n, L);
    X[lv] = 0.9999;// may error, keep it away from 1 by changing to 0.9999!
  }
  //
  lv = X <= 0.0001;
  n = Rcpp::sum(lv);
  if (n >= 1) {
    X[lv] = 0.0001;// may error, keep it away from 0 by changing to 0.0001! 
  }
  //
  double lb = X[0];         // smallest X
  Y[0] = intq_beta(g, N, a, b, lb, 0.0001); // int_0^lb q(x)dx  0->0.0001
  //
  for (int i=1; i < L; i++) {
    Y[i] = Y[i-1] + intq_beta(g, N, a, b, X[i], X[i-1]);
  }
  return Y;
}

//' Evaluate the Objective Function
//' 
//' Analog to [f_obj()], but here the model is set with beta distribution.
//' 
//' @seealso [f_obj()]
//' 
// [[Rcpp::export]]
double f_obj_beta(double g, double N, double a, double b, int L,
                  NumericVector moments, double w0=0.1) {
  NumericVector Y = sim_Y_beta(g, N, a, b, L, w0);
  NumericVector smoments = mom(Y);
  double error = Rcpp::sum(Rcpp::pow(smoments-moments, 2));
  return error;
}

//' Evaluate Gradient of the Objective Function
//' 
//' Analog to [df_obj()], but here the model is set with beta distribution.
//' 
//' @seealso [df_obj()]
//' 
// [[Rcpp::export]]
NumericVector df_obj_beta(double g, double N, double a, double b, int L,
                          NumericVector moments, NumericVector stepsize, 
                          double w0=0.1) {
  NumericVector df (4);
  double f1, f2, h;
  //
  h = stepsize[0];
  f1 = f_obj_beta(g-h, N, a, b, L, moments, w0);
  f2 = f_obj_beta(g+h, N, a, b, L, moments, w0);
  df[0] = (f2-f1)/(2*h);
  //
  h = stepsize[1];
  f1 = f_obj_beta(g, N-h, a, b, L, moments, w0);
  f2 = f_obj_beta(g, N+h, a, b, L, moments, w0);
  df[1] = (f2 - f1) / (2*h);
  //
  h = stepsize[2];
  f1 = f_obj_beta(g, N, a-h, b, L, moments, w0);
  f2 = f_obj_beta(g, N, a+h, b, L, moments, w0);
  df[2] = (f2 - f1) / (2*h);
  //
  h = stepsize[3];
  f1 = f_obj_beta(g, N, a, b-h, L, moments, w0);
  f2 = f_obj_beta(g, N, a, b+h, L, moments, w0);
  df[3] = (f2 - f1) / (2*h);
  //
  return df;
}

//' Stochastic Approximation
//' 
//' Analog to [sa()], but here the model is set with beta distribution.
//' 
//' @seealso [sa()]
//' 
// [[Rcpp::export]]
NumericMatrix sa_beta(NumericVector theta0, NumericMatrix Theta, 
                      NumericVector stepsize,
                      int L, NumericVector moments, int G=200) {
  NumericMatrix theta_f_record(G+1, 5);
  NumericVector df (4);
  double f;
  //
  f = f_obj_beta(theta0[0], theta0[1], theta0[2], theta0[3], L, moments);
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
    df = df_obj_beta(theta0[0],theta0[1],theta0[2],theta0[3],L,moments,stepsize);
    for (int n=0; n < theta0.length(); n++) {
      if (df[n] > 0.0001) {        // 0.0001 <-> 0
        theta0[n] -= stepsize[n]; //theta0[n] -= 0.01;
      } else if (df[n] < 0.0001) { // 0.0001 <-> 0
        theta0[n] += stepsize[n]; // theta0[n] += 0.01;
      }
    }
    //
    theta0 = project(theta0, Theta);
    f = f_obj_beta(theta0[0], theta0[1], theta0[2], theta0[3], L, moments);
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
