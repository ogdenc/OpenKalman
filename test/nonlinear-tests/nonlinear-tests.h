/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef NONLINEAR_TESTS_H
#define NONLINEAR_TESTS_H

#include <functional>
#include "../tests.h"
#include "../transformations.h"
#include "distributions/GaussianDistribution.h"
//#include "transformations/LinearizedTransformation.h"

#include <gtest/gtest.h>

using namespace OpenKalman;

using C2 = Coefficients<Axis, Axis>;

class nonlinear_tests : public ::testing::Test {
public:
    nonlinear_tests() {
    }

    void SetUp() override {
        // code here will execute just before the test ensues
    }

    void TearDown() override {
        // code here will be called just after the test completes
        // ok to throw exceptions from here if need be
    }

    ~nonlinear_tests() override {
        // cleanup any pending stuff, but no exceptions allowed
    }

    // put in any custom members that you need


    template<
        template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...> typename Trans,
        template<typename, typename> typename Dist,
        typename Scalar,
        int n,
        typename ... Args>
    void doReduction(
        const Trans<Dist, Scalar, Axes<n>, Coefficients<Axis>, NoiseType::none, Args ...>& t,
        const Eigen::Matrix<Scalar, n, 1>& mu,
        const Eigen::Matrix<Scalar, n, n>& P,
        const Scalar expected_mu,
        const Scalar expected_P,
        const Scalar err_mu,
        const Scalar err_P)
    {
        const GaussianDistribution<Scalar, Axes<n>> in {mu, P};
        const GaussianDistribution<Scalar, Coefficients<Axis>> out {std::get<0>(t(in))};
        EXPECT_NEAR(mean(out)(0), expected_mu, err_mu);
        EXPECT_NEAR(covariance(out).base_matrix()(0), expected_P, err_P);
    }


    template<
        template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...> typename Trans,
        template<typename, typename> typename Dist,
        typename Scalar,
        typename ... Args>
    void do2x2Transform(
        const Trans<Dist, Scalar, C2, C2, NoiseType::none, Args ...>& t,
        const Eigen::Matrix<Scalar, 2, 1>& mu_x,
        const Eigen::Matrix<Scalar, 2, 2>& P_xx,
        const std::array<Scalar, 4>& expected_mu_z,
        const std::array<Scalar, 8>& expected_P_zz)
    {
        const auto res = std::get<0>(t(GaussianDistribution<Scalar, C2> {mu_x, P_xx}));
        const Eigen::Matrix<Scalar, 2, 1> mu_z = mean(res);
        const Eigen::Matrix<Scalar, 2, 2> P_zz = covariance(res).base_matrix();
        EXPECT_NEAR(mu_z(0), expected_mu_z[0], expected_mu_z[1]);
        EXPECT_NEAR(mu_z(1), expected_mu_z[2], expected_mu_z[3]);
        EXPECT_NEAR(P_zz(0,0), expected_P_zz[0], expected_P_zz[1]);
        EXPECT_NEAR(P_zz(0,1), expected_P_zz[2], expected_P_zz[3]);
        EXPECT_NEAR(P_zz(1,0), expected_P_zz[4], expected_P_zz[5]);
        EXPECT_NEAR(P_zz(1,1), expected_P_zz[6], expected_P_zz[7]);
    }


};


#endif //NONLINEAR_TESTS_H
