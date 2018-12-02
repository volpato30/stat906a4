//
// Created by rui on 01/12/18.
//

#ifndef STAT906A4_UTILS_H
#define STAT906A4_UTILS_H
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "Eigen/Dense"

inline double mean(const std::vector<double> &x) {
    double m = 0;
    auto length = double(x.size());
    for (const double &d : x) {
        m += d;
    }
    m = m / length;
    return m;
}

inline double computeVariance(const std::vector<double> &x_array) {
    double mean = 0, variance = 0;
    unsigned long n = x_array.size();
    for (const double &x : x_array) {
        mean += x;
    }
    mean = mean / n;
    for (const double &x : x_array) {
        variance += pow(x - mean, 2);
    }
    variance = variance / (n - 1);
    return variance;
}

double get_expectation(const std::vector<double>& prices) {
    double result = 0.;
    unsigned long m = 12;
    double final_price = prices.at(m-1);
    for (const double & s : prices) {
        result += std::max(s - final_price, 0.0);
    }
    result /= m;
    return result;
}


class KorobovLattice {
public:
    KorobovLattice(unsigned long num, unsigned long a, unsigned ndim) : _num(num), _a(a), _ndim(ndim) {
        _counter = 0;
    }

    void next_point(std::vector<double> &result) {
        if (_counter >= _num) {
            throw std::runtime_error("Korobov lattice called too many time, out of bound");
        }
        double coef, intpart, fractpart;
        unsigned long pp=1;
        for (unsigned i = 0; i < _ndim; i++) {
            // for numerical reason add epsilon
            coef = (_counter + 0.5) / _num;
            fractpart = modf(coef * (pp % _num), &intpart);
            result.at(i) = fractpart;
            pp *= _a;
        }
        _counter += 1;
    }

    void reset() {
        _counter = 0;
    }

private:
    unsigned long _num;
    unsigned long _a;
    unsigned _ndim;
    unsigned long _counter;
};


//sources:
//https://stackoverflow.com/questions/16361226/error-while-creating-object-from-templated-class/16364899?noredirect=1#comment23457843_16364899
namespace Eigen {
    namespace internal {
        template<typename Scalar>
        struct scalar_normal_dist_op
        {
            static std::mt19937 rng;                        // The uniform pseudo-random algorithm
            mutable std::normal_distribution<Scalar> norm;  // The gaussian combinator

            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
        };

        template<typename Scalar>
        std::mt19937 scalar_normal_dist_op<Scalar>::rng;

        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> >
        { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };

    } // end namespace internal
/**
    Find the eigen-decomposition of the covariance matrix
    and then store it for sampling from a multi-variate normal
*/
    template<typename Scalar, int Size>
    class EigenMultivariateNormal
    {
        Matrix<Scalar,Size,Size> _covar;
        Matrix<Scalar,Size,Size> _transform;
        Matrix< Scalar, Size, 1> _mean;
        internal::scalar_normal_dist_op<Scalar> randN; // Gaussian functor


    public:
        EigenMultivariateNormal(const Matrix<Scalar,Size,1>& mean,const Matrix<Scalar,Size,Size>& covar)
        {
            setMean(mean);
            setCovar(covar);
        }

        void setMean(const Matrix<Scalar,Size,1>& mean) { _mean = mean; }
        Matrix<Scalar,Size,1> getMean() {return _mean;}
        void setCovar(const Matrix<Scalar,Size,Size>& covar)
        {
            _covar = covar;

            // Assuming that we'll be using this repeatedly,
            // compute the transformation matrix that will
            // be applied to unit-variance independent normals

            /*
            Eigen::LDLT<Eigen::Matrix<Scalar,Size,Size> > cholSolver(_covar);
            // We can only use the cholesky decomposition if
            // the covariance matrix is symmetric, pos-definite.
            // But a covariance matrix might be pos-semi-definite.
            // In that case, we'll go to an EigenSolver
            if (cholSolver.info()==Eigen::Success) {
              // Use cholesky solver
              _transform = cholSolver.matrixL();
            } else {*/
            SelfAdjointEigenSolver<Matrix<Scalar,Size,Size> > eigenSolver(_covar);
            _transform = eigenSolver.eigenvectors()*eigenSolver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
            /*}*/

        }

        /// Draw nn samples from the gaussian and return them
        /// as columns in a Size by nn matrix
        Matrix<Scalar,Size,-1> samples(int nn)
        {
            return (_transform * Matrix<Scalar,Size,-1>::NullaryExpr(Size,nn,randN)).colwise() + _mean;
        }

        Matrix<Scalar,Size,-1> transform(Matrix<Scalar,Size, 1> standard_gaussian)
        {
            return _transform * standard_gaussian + _mean;
        }
    }; // end class EigenMultivariateNormal
} // end namespace Eigen


#endif //STAT906A4_UTILS_H
