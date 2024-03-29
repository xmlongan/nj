% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{f_obj}
\alias{f_obj}
\title{Evaluate the Objective Function}
\usage{
f_obj(g, N, mu, sigma, L, moments, w0 = 0.1)
}
\arguments{
\item{g}{The parameter \eqn{g}.}

\item{N}{The parameter \eqn{N}, it should be larger than 2.}

\item{mu}{The parameter \eqn{\mu}, mean of the normal distribution.}

\item{sigma}{The parameter \eqn{\sigma}, standard deviation of the normal
distribution.}

\item{L}{Length of the samples to generate.}

\item{moments}{A vector of the true sample moments, 1st moment,
2nd - 4th central moments.}

\item{w0}{The parameter \eqn{w_0}, default to 0.1.}
}
\value{
A scalar value.
}
\description{
Evaluate the objective function, i.e., the error function. First, generate
a sequence of samples by simulation under the given parameters. Then,
compute the first four (central) moments of the simulated samples. Last,
compute the differences between the true sample moments and the simulated
sample moments.
}
\examples{
Y = sim_Y(0.2, 3.5, 0.5, 0.1, 1000)
moments = mom(Y)
f_obj(0.3, 3, 0.4, 0.05, 1000, moments)

}
