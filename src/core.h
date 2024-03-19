#include <Rcpp.h>
using namespace Rcpp;

NumericVector mom(NumericVector samples);
NumericVector project(NumericVector theta, NumericMatrix Theta);