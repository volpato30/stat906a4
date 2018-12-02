#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>
#include "normalInverse.h"
#include "utils.h"


std::random_device rd;
std::mt19937 rgen(rd());
using std::normal_distribution;
using std::uniform_real_distribution;

double s0 = 1.0, r=0.03, sigma=0.4;



void method1(std::vector<double> &prices) {
    double delta_t = 1.0 / 12, previous_price=s0;
    normal_distribution<double> normal_gen(0, sigma*sqrt(delta_t));
    prices.clear();
    for (unsigned long i = 0; i<12; i++) {
        previous_price = previous_price * exp( (r - sigma*sigma / 2.0) * delta_t + normal_gen(rgen) );
        prices.push_back(previous_price);
    }
}

void method2(std::vector<double> &prices, KorobovLattice &kl, Eigen::EigenMultivariateNormal<double, 11> &normX,
        Eigen::Matrix<double, 11, 1> &Sigma_12) {
    prices.clear();
    std::vector<double> low_discrepancy_point(1);
    kl.next_point(low_discrepancy_point);
    //first generate log_s_t_12
    double mu_12 = (r - sigma * sigma / 2.);
    double log_s_t_12 = mu_12 + sigma * NormalCDFInverse(low_discrepancy_point.at(0));

    Eigen::Matrix<double, 11, 1> _original_mean;
    for (int i =0; i < 11; i++) {
        _original_mean(i, 0) = (r - sigma * sigma / 2.) * (i + 1) / 12.;
    }
    Eigen::Matrix<double, 11, 1> _conditional_mean;
    _conditional_mean = _original_mean + (Sigma_12 / (sigma * sigma) * (log_s_t_12 - mu_12));
    normX.setMean(_conditional_mean);
    auto sampled_log_ss = normX.samples(1);
    for (unsigned long j=0; j < 12; j++) {
        if (j < 11) {
            prices.push_back(exp(sampled_log_ss(j, 0)));
        }
        else {
            prices.push_back(exp(log_s_t_12));
        }
    }
}

Eigen::Matrix<double, 2, 1> get_log_s_6_and_log_s_12(std::vector<double> &low_discrepancy_point,
        Eigen::EigenMultivariateNormal<double, 2> &normX0) {
    Eigen::Matrix<double, 2, 1> standard_gaussian;
    standard_gaussian(0, 0) = NormalCDFInverse(low_discrepancy_point.at(0));
    standard_gaussian(1, 0) = NormalCDFInverse(low_discrepancy_point.at(1));
    return normX0.transform(standard_gaussian);
}

void method3(std::vector<double> &prices, KorobovLattice &kl, Eigen::EigenMultivariateNormal<double, 2> &normX0,
        Eigen::EigenMultivariateNormal<double, 10> &conditionalNormX,
        Eigen::Matrix<double, 10, 2> &Sigma12,
        Eigen::Matrix<double, 2, 2> &Sigma22){
    prices.clear();
    std::vector<double> low_discrepancy_point(2);
    kl.next_point(low_discrepancy_point);
    auto x0 = get_log_s_6_and_log_s_12(low_discrepancy_point, normX0);

    Eigen::Matrix<double, 10, 1> _original_mean;
    for (int i =0; i < 10; i++) {
        if (i < 5) {
            _original_mean(i, 0) = (r - sigma * sigma / 2.) * (i + 1) / 12.;
        } else if (i == 5) {
            continue;
        } else {
            _original_mean(i-1, 0) = (r - sigma * sigma / 2.) * (i + 1) / 12.;
        }
    }
    Eigen::Matrix<double, 10, 1> _conditional_mean;
    _conditional_mean = _original_mean + (Sigma12 * Sigma22.inverse() * (x0 - normX0.getMean()));
    conditionalNormX.setMean(_conditional_mean);
    auto sampled_log_ss = conditionalNormX.samples(1);
    for (unsigned long j=0; j < 12; j++) {
        if (j < 5) {
            prices.push_back(exp(sampled_log_ss(j, 0)));
        } else if (j==5) {
            prices.push_back(exp(x0(0,0)));
        } else if (j < 11) {
            prices.push_back(exp(sampled_log_ss(j-1, 0)));
        }
        else {
            prices.push_back(exp(x0(1,0)));
        }
    }

}

int main() {
    int i=0, j=0;
    //method 1
    std::vector<double> prices;
    unsigned long num_exp = 1000;
    unsigned long num_path = 100;
    std::vector<double> results(num_exp);
    std::vector<double> mc_paths(num_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned long index = 0; index < num_exp; index++) {
        for (unsigned long path_index = 0; path_index < num_path; path_index++){
            method1(prices);
            mc_paths.at(path_index) = get_expectation(prices);
        }
        results.at(index) = mean(mc_paths);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "mean:\t" << mean(results) << "\tvariance:\t" << computeVariance(results) << "\ttakes\t"
        <<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;


    //method 2
    KorobovLattice kl1(num_path, 3709, 1);
    // initialize cov matrix
    Eigen::Matrix<double, 11, 11> Sigma11;
    Eigen::Matrix<double, 11, 1> Sigma12;
    Eigen::Matrix<double, 1, 11> Sigma21;
    for (i=0; i<11; i++) {
        for (j=0; j<11; j++) {
            Sigma11(i, j) = double(std::min(i+1, j+1)) / 12. * sigma * sigma;
        }
        Sigma12(i, 0) = double(i+1) / 12 * sigma * sigma;
        Sigma21(0, i) = double(i+1) / 12 * sigma * sigma;
    }

    Eigen::Matrix<double, 11, 11> covMatrix = Sigma11 - Sigma12 * Sigma21 / (sigma * sigma);
    // initial mean of normX does not matter
    Eigen::EigenMultivariateNormal<double, 11> normX(Sigma12, covMatrix);

    t1 = std::chrono::high_resolution_clock::now();
    for (unsigned long index = 0; index < num_exp; index++) {
        kl1.reset();
        for (unsigned long path_index = 0; path_index < num_path; path_index++){
            method2(prices, kl1, normX, Sigma12);
            mc_paths.at(path_index) = get_expectation(prices);
        }
        results.at(index) = mean(mc_paths);
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "mean:\t" << mean(results) << "\tvariance:\t" << computeVariance(results) << "\ttakes\t"
              <<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;

    //method 3
    KorobovLattice kl2(num_path, 3709, 2);
    // initialize cov matrix
    Eigen::Matrix<double, 10, 10> Sigma11_;
    Eigen::Matrix<double, 10, 2> Sigma12_;
    Eigen::Matrix<double, 2, 10> Sigma21_;
    int row, col;
    for (i=0; i<10; i++) {
        if (i < 5) {
            row = i;
        } else{
            row = i+1;
        }
        for (j=0; j<10; j++) {
            if (j < 5) {
                col = j;
            } else{
                col = j+1;
            }
            Sigma11_(i, j) = double(std::min(row+1, col+1)) / 12. * sigma * sigma;
        }
        Sigma12_(i, 0) = double(std::min(row+1, 6)) / 12 * sigma * sigma;
        Sigma12_(i, 1) = double(row+1) / 12 * sigma * sigma;
        Sigma21_(0, i) = double(std::min(row+1, 6)) / 12 * sigma * sigma;
        Sigma21_(1, i) = double(row+1) / 12 * sigma * sigma;
    }
    Eigen::Matrix<double, 2, 2> Sigma22_;
    Sigma22_ << 0.5 * sigma * sigma, 0.5 * sigma * sigma,
                0.5 * sigma * sigma, sigma * sigma;
    Eigen::Matrix<double, 2, 1> mu0;
    mu0 << (r - sigma * sigma / 2.) * 0.5,
            (r - sigma * sigma / 2.);

    Eigen::Matrix<double, 10, 10> covMatrix_ = Sigma11_ - Sigma12_ * Sigma22_.inverse() * Sigma21_;

    Eigen::EigenMultivariateNormal<double, 2> normX0(mu0, Sigma22_);
    // initial mean of conditionalNormX does not matter
    Eigen::Matrix<double, 10, 1> temp;
    Eigen::EigenMultivariateNormal<double, 10> conditionalNormX(temp, covMatrix_);

    t1 = std::chrono::high_resolution_clock::now();
    for (unsigned long index = 0; index < num_exp; index++) {
        kl2.reset();
        for (unsigned long path_index = 0; path_index < num_path; path_index++){
            method3(prices, kl2, normX0, conditionalNormX, Sigma12_, Sigma22_);
            mc_paths.at(path_index) = get_expectation(prices);
        }
        results.at(index) = mean(mc_paths);
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "mean:\t" << mean(results) << "\tvariance:\t" << computeVariance(results) << "\ttakes\t"
              <<std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds" << std::endl;

    return 0;
}