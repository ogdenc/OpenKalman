/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformation-tests.h"

using M1 = Mean<Axis, Eigen::Matrix<double, 1, 1>>;
using M2 = Mean<Axes<2>, Eigen::Matrix<double, 2, 1>>;
using M2P = Mean<Polar<>, Eigen::Matrix<double, 2, 1>>;
using M3 = Mean<Axes<3>, Eigen::Matrix<double, 3, 1>>;
using A22 = TypedMatrix<Axes<2>, Axes<2>, Eigen::Matrix<double, 2, 2>>;
using A22P = TypedMatrix<Polar<>, Axes<2>, Eigen::Matrix<double, 2, 2>>;
using A22PP = TypedMatrix<Polar<>, Polar<>, Eigen::Matrix<double, 2, 2>>;
using A32 = TypedMatrix<Axes<3>, Axes<2>, Eigen::Matrix<double, 3, 2>>;
using A33 = TypedMatrix<Axes<3>, Axes<3>, Eigen::Matrix<double, 3, 3>>;

TEST_F(transformation_tests, finite_diff_linear_2by2)
{
  A22 a {1, 2, 3, 4};
  auto f = [&] (const auto& x, const auto&...n) { return ((a * x) + ... + (A22::identity() * n)); };
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(3, -4))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2), M2(0.1, 0.2))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M2(0.1, 0.2))), A22::identity()));
  EXPECT_TRUE(is_near(std::get<2>(t.jacobian(M2(1, 2), M2(0.1, 0.2), M2(0.3, 0.4))), A22::identity()));
}

TEST_F(transformation_tests, finite_diff_linear_2by3)
{
  A32 a {1, 2, 3, 4, 5, 6};
  auto f = [&] (const auto& x, const auto&...n) { return ((a * x) + ... + (A33::identity() * n)); };
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M3 {1e-4, 1e-4, 1e-4}, M3 {1e-4, 1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t(M2(1, 2)), M3(5, 11, 17)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(3, 4))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3))), A33::identity()));
  EXPECT_TRUE(is_near(std::get<2>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3), M3(0.4, 0.5, 0.6))), A33::identity()));
}

TEST_F(transformation_tests, finite_diff_sum_of_squares_2D)
{
  auto f = sum_of_squares<2>;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M2(2, 3)), M1(13)));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2)), f.jacobian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, -2), M1(0.1)), f.jacobian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2), M1(0.1), M1(0.2)), f.jacobian(M2(1, 2), M1(0.1), M1(0.2))));
}

TEST_F(transformation_tests, finite_diff_sum_of_squares_3D)
{
  auto f = sum_of_squares<3>;
  auto t = FiniteDifferenceLinearization {f, M3 {1e-4, 1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M3(1, 2, 3)), M1(14)));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, 2, 3)), f.jacobian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, -2, 3), M1(0.1)), f.jacobian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
}

TEST_F(transformation_tests, finite_diff_time_of_arrival_2D)
{
  auto f = time_of_arrival<2>;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M2(3, 4)), M1(5)));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2)), f.jacobian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, -2), M1(0.1)), f.jacobian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-1, 2), M1(0.1), M1(0.2)), f.jacobian(M2(-1, 2), M1(0.1), M1(0.2))));
}

TEST_F(transformation_tests, finite_diff_time_of_arrival_3D)
{
  auto f = time_of_arrival<3>;
  auto t = FiniteDifferenceLinearization {f, M3 {1e-4, 1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M3(1, 2, 3)), M1(std::sqrt(14))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, 2, 3)), f.jacobian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, -2, 3), M1(0.1)), f.jacobian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
}

TEST_F(transformation_tests, finite_diff_radar)
{
  auto f = radar;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-5}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t.jacobian(M2(2, M_PI/3)), f.jacobian(M2(2, M_PI/3))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-2, M_PI/3), M2(0.2, 0.3)), f.jacobian(M2(-2, M_PI/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.jacobian(M2(2, -M_PI/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.jacobian(M2(2, -M_PI/3), M2(0.2, 0.3), M2(0.2, 0.3))));
}

TEST_F(transformation_tests, finite_diff_radarP)
{
  auto f = radarP;
  auto t = FiniteDifferenceLinearization {f, M2P {1e-4, 1e-5}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t.jacobian(M2P(2, M_PI/3)), f.jacobian(M2P(2, M_PI/3))));
  EXPECT_TRUE(is_near(t.jacobian(M2P(2, -M_PI/3), M2(0.2, 0.3)), f.jacobian(M2P(2, -M_PI/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.jacobian(M2P(2, -M_PI/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.jacobian(M2P(2, -M_PI/3), M2(0.2, 0.3), M2(0.2, 0.3))));
}

TEST_F(transformation_tests, finite_diff_Cartesian2polar)
{
  auto f = Cartesian2polar;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M2P {1e-4, 1e-5}, M2P {1e-4, 1e-5}};
  EXPECT_TRUE(is_near(t.jacobian(M2(2, 1)), f.jacobian(M2(2, 1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(2, -1), M2P(0.2, M_PI/30)), f.jacobian(M2(2, -1), M2P(0.2, M_PI/30))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-2, 1), M2P(0.2, M_PI/30), M2P(0.2, M_PI/30)), f.jacobian(M2(-2, 1), M2P(0.2, M_PI/30), M2P(0.2, M_PI/30))));
}

