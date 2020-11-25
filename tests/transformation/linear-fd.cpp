/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformations.hpp"

using M1 = Mean<Axis, native_matrix_t<double, 1, 1>>;
using M2 = Mean<Axes<2>, native_matrix_t<double, 2, 1>>;
using MP = Mean<Polar<>, native_matrix_t<double, 2, 1>>;
using M3 = Mean<Axes<3>, native_matrix_t<double, 3, 1>>;
using MC = Mean<Coefficients<Polar<>, Axis>, native_matrix_t<double, 3, 1>>;
using MS = Mean<Spherical<>, native_matrix_t<double, 3, 1>>;
using A22 = Matrix<Axes<2>, Axes<2>, native_matrix_t<double, 2, 2>>;
using AP2 = Matrix<Polar<>, Axes<2>, native_matrix_t<double, 2, 2>>;
using APP = Matrix<Polar<>, Polar<>, native_matrix_t<double, 2, 2>>;
using A32 = Matrix<Axes<3>, Axes<2>, native_matrix_t<double, 3, 2>>;
using A33 = Matrix<Axes<3>, Axes<3>, native_matrix_t<double, 3, 3>>;

TEST_F(transformations, finite_diff_linear_2by2)
{
  A22 a {1, 2, 3, 4};
  auto f = [&] (const auto& x, const auto&...n) { return make_self_contained(((a * x) + ... + (A22::identity() * n))); };
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(3, -4))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2), M2(0.1, 0.2))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M2(0.1, 0.2))), A22::identity()));
  EXPECT_TRUE(is_near(std::get<2>(t.jacobian(M2(1, 2), M2(0.1, 0.2), M2(0.3, 0.4))), A22::identity()));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2), M2(0.1, 0.2), M2(0.3, 0.4)), zero_hessian<Axes<2>, M2, M2, M2>()));
}

TEST_F(transformations, finite_diff_linear_2by3)
{
  A32 a {1, 2, 3, 4, 5, 6};
  auto f = [&] (const auto& x, const auto&...n) { return make_self_contained(((a * x) + ... + (A33::identity() * n))); };
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M3 {1e-4, 1e-4, 1e-4}, M3 {1e-4, 1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t(M2(1, 2)), M3(5, 11, 17)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(3, 4))), a));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2)), zero_hessian<Axes<3>, M2>()));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3))), A33::identity()));
  EXPECT_TRUE(is_near(std::get<2>(t.jacobian(M2(1, 2), M3(0.1, 0.2, 0.3), M3(0.4, 0.5, 0.6))), A33::identity()));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2), M3(0.1, 0.2, 0.3), M3(0.4, 0.5, 0.6)), zero_hessian<Axes<3>, M2, M3, M3>()));
}

TEST_F(transformations, finite_diff_sum_of_squares_2D)
{
  auto f = sum_of_squares<2>;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M2(2, 3)), M1(13)));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2)), f.jacobian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2)), f.hessian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, -2), M1(0.1)), f.jacobian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.hessian(M2(1, -2), M1(0.1)), f.hessian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2), M1(0.1), M1(0.2)), f.jacobian(M2(1, 2), M1(0.1), M1(0.2))));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2), M1(0.1), M1(0.2)), f.hessian(M2(1, 2), M1(0.1), M1(0.2))));
}

TEST_F(transformations, finite_diff_sum_of_squares_3D)
{
  auto f = sum_of_squares<3>;
  auto t = FiniteDifferenceLinearization {f, M3 {1e-4, 1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M3(1, 2, 3)), M1(14)));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, 2, 3)), f.jacobian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.hessian(M3(1, 2, 3)), f.hessian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, -2, 3), M1(0.1)), f.jacobian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.hessian(M3(1, -2, 3), M1(0.1)), f.hessian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
  EXPECT_TRUE(is_near(t.hessian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.hessian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
}

TEST_F(transformations, finite_diff_time_of_arrival_2D)
{
  auto f = time_of_arrival<2>;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M2(3, 4)), M1(5)));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, 2)), f.jacobian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.hessian(M2(1, 2)), f.hessian(M2(1, 2))));
  EXPECT_TRUE(is_near(t.jacobian(M2(1, -2), M1(0.1)), f.jacobian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.hessian(M2(1, -2), M1(0.1)), f.hessian(M2(1, -2), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-1, 2), M1(0.1), M1(0.2)), f.jacobian(M2(-1, 2), M1(0.1), M1(0.2))));
  EXPECT_TRUE(is_near(t.hessian(M2(-1, 2), M1(0.1), M1(0.2)), f.hessian(M2(-1, 2), M1(0.1), M1(0.2))));
}

TEST_F(transformations, finite_diff_time_of_arrival_3D)
{
  auto f = time_of_arrival<3>;
  auto t = FiniteDifferenceLinearization {f, M3 {1e-4, 1e-4, 1e-4}, M1 {1e-4}, M1 {1e-4}};
  EXPECT_TRUE(is_near(t(M3(1, 2, 3)), M1(std::sqrt(14))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, 2, 3)), f.jacobian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.hessian(M3(1, 2, 3)), f.hessian(M3(1, 2, 3))));
  EXPECT_TRUE(is_near(t.jacobian(M3(1, -2, 3), M1(0.1)), f.jacobian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.hessian(M3(1, -2, 3), M1(0.1)), f.hessian(M3(1, -2, 3), M1(0.1))));
  EXPECT_TRUE(is_near(t.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.jacobian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
  EXPECT_TRUE(is_near(t.hessian(M3(-1, 2, 3), M1(0.1), M1(0.2)), f.hessian(M3(-1, 2, 3), M1(0.1), M1(0.2))));
}

TEST_F(transformations, finite_diff_radar)
{
  auto f = radar;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-5}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t.jacobian(M2(2, pi/3)), f.jacobian(M2(2, pi/3))));
  EXPECT_TRUE(is_near(t.hessian(M2(2, pi/3)), f.hessian(M2(2, pi/3))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-2, pi/3), M2(0.2, 0.3)), f.jacobian(M2(-2, pi/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.hessian(M2(-2, pi/3), M2(0.2, 0.3)), f.hessian(M2(-2, pi/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.jacobian(M2(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.jacobian(M2(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.hessian(M2(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.hessian(M2(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3))));
}

TEST_F(transformations, finite_diff_radarP)
{
  auto f = radarP;
  auto t = FiniteDifferenceLinearization {f, MP {1e-4, 1e-5}, M2 {1e-4, 1e-4}, M2 {1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t.jacobian(MP(0, pi)), f.jacobian(MP(0, pi))));
  EXPECT_TRUE(is_near(t.hessian(MP(0, pi)), f.hessian(MP(0, pi))));
  EXPECT_TRUE(is_near(t.jacobian(MP(2, pi)), f.jacobian(MP(2, pi))));
  EXPECT_TRUE(is_near(t.hessian(MP(2, pi)), f.hessian(MP(2, pi))));
  EXPECT_TRUE(is_near(t.jacobian(MP(2, pi/3)), f.jacobian(MP(2, pi/3))));
  EXPECT_TRUE(is_near(t.hessian(MP(2, pi/3)), f.hessian(MP(2, pi/3))));
  EXPECT_TRUE(is_near(t.jacobian(MP(2, -pi/3), M2(0.2, 0.3)), f.jacobian(MP(2, -pi/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.hessian(MP(2, -pi/3), M2(0.2, 0.3)), f.hessian(MP(2, -pi/3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.jacobian(MP(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.jacobian(MP(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3))));
  EXPECT_TRUE(is_near(t.hessian(MP(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3)), f.hessian(MP(2, -pi/3), M2(0.2, 0.3), M2(0.2, 0.3))));
}

TEST_F(transformations, finite_diff_Cartesian2polar)
{
  auto f = Cartesian2polar;
  auto t = FiniteDifferenceLinearization {f, M2 {1e-4, 1e-4}, MP {1e-4, 1e-5}, MP {1e-4, 1e-5}};
  EXPECT_TRUE(is_near(t.jacobian(M2(-2, 0)), f.jacobian(M2(-2, 0))));
  EXPECT_TRUE(is_near(t.hessian(M2(-2, 0)), f.hessian(M2(-2, 0))));
  EXPECT_TRUE(is_near(t.jacobian(M2(2, 1)), f.jacobian(M2(2, 1))));
  EXPECT_TRUE(is_near(t.hessian(M2(2, 1)), f.hessian(M2(2, 1))));
  EXPECT_TRUE(is_near(t.jacobian(M2(2, -1), MP(0.2, pi/30)), f.jacobian(M2(2, -1), MP(0.2, pi/30))));
  EXPECT_TRUE(is_near(t.hessian(M2(2, -1), MP(0.2, pi/30)), f.hessian(M2(2, -1), MP(0.2, pi/30))));
  EXPECT_TRUE(is_near(t.jacobian(M2(-2, 1), MP(0.2, pi/30), MP(0.2, pi/30)), f.jacobian(M2(-2, 1), MP(0.2, pi/30), MP(0.2, pi/30))));
  EXPECT_TRUE(is_near(t.hessian(M2(-2, 1), MP(0.2, pi/30), MP(0.2, pi/30)), f.hessian(M2(-2, 1), MP(0.2, pi/30), MP(0.2, pi/30))));
}

TEST_F(transformations, finite_diff_Cylindrical2spherical)
{
  auto f = Cylindrical2spherical;
  auto t = FiniteDifferenceLinearization {f, MC {1e-4, 1e-4, 1e-4}, MS {1e-4, 1e-4, 1e-4}, MS {1e-4, 1e-4, 1e-4}};
  EXPECT_TRUE(is_near(t.jacobian(MC(2, pi, 1)), f.jacobian(MC(2, pi, 1))));
  EXPECT_TRUE(is_near(t.hessian(MC(2, pi, 1)), f.hessian(MC(2, pi, 1))));
  EXPECT_TRUE(is_near(t.jacobian(MC(2, pi/3, 1)), f.jacobian(MC(2, pi/3, 1))));
  EXPECT_TRUE(is_near(t.hessian(MC(2, pi/3, 1)), f.hessian(MC(2, pi/3, 1))));
  EXPECT_TRUE(is_near(t.jacobian(MC(2, pi/3, -1), MS(0.2, pi/30, pi/30)), f.jacobian(MC(2, pi/3, -1), MS(0.2, pi/30, pi/30))));
  EXPECT_TRUE(is_near(t.hessian(MC(2, pi/3, -1), MS(0.2, pi/30, pi/30)), f.hessian(MC(2, pi/3, -1), MS(0.2, pi/30, pi/30))));
  EXPECT_TRUE(is_near(t.jacobian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30), MS(0.2, pi/30, pi/30)), f.jacobian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30), MS(0.2, pi/30, pi/30))));
  EXPECT_TRUE(is_near(t.hessian(MC(-2, pi/3, 1)), f.hessian(MC(-2, pi/3, 1))));
  EXPECT_TRUE(is_near(t.hessian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30)), f.hessian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30))));
  EXPECT_TRUE(is_near(t.hessian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30), MS(0.2, pi/30, pi/30)), f.hessian(MC(-2, pi/3, 1), MS(0.2, pi/30, pi/30), MS(0.2, pi/30, pi/30))));
}

