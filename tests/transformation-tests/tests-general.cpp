/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformation-tests.hpp"

using namespace OpenKalman;

using M_int2 = Mean<Axes<2>, Eigen::Matrix<int, 2, 1>>;
using A_int2 = TypedMatrix<Axes<2>, Axes<2>, Eigen::Matrix<int, 2, 2>>;

template<auto t, auto tn = t>
struct Scale
{
  template<typename Arg, typename ... Noise>
  auto operator()(const Arg& in, const Noise& ... n) const
  {
    return strict(((t * in) + ... + (tn * n)));
  }
};

TEST_F(transformation_tests, Scale_no_noise_1D)
{
  using M_int1 = Mean<Coefficients<Axis>, Eigen::Matrix<int, 1, 1>>;
  Transformation<Scale<3>> t;
  EXPECT_EQ(t(M_int1 {2}), M_int1 {6});
}

TEST_F(transformation_tests, Scale_no_noise_2D)
{
  Transformation<Scale<7>> t;
  EXPECT_EQ(t(M_int2 {2, 3}), M_int2(14, 21));
}

TEST_F(transformation_tests, Scale_additive)
{
  Transformation<Scale<7>> t;
  EXPECT_EQ(t(M_int2 {2, 3}), (M_int2 {14, 21}));
  EXPECT_EQ(t(M_int2(2, 3)) + M_int2(1, 1), M_int2(15, 22));
  static_assert(std::is_same_v<decltype(t(M_int2(2, 3)) + M_int2(1, 1)), decltype(M_int2(15, 22))>);
  EXPECT_EQ(t(M_int2(2, 3)) + M_int2(3, 3), M_int2(17, 24));
}

TEST_F(transformation_tests, Scale_augmented)
{
  Transformation<Scale<7, 3>> t;
  EXPECT_EQ(t(M_int2 {2, 3}, M_int2 {1, 1}), M_int2(17, 24));
}

TEST_F(transformation_tests, Mult_additive_axis)
{
  using M = Mean<Axes<2>>;
  using A = TypedMatrix<Axes<2>, Axes<2>>;
  const auto f = [](const M& x) -> M { return A {1, 2, 3, 4} * x; };
  auto t = make_Transformation(f);
  EXPECT_TRUE(is_near(t(M(1, 0.5)) + M(0.1, 0.1), M(2.1, 5.1)));
}

TEST_F(transformation_tests, Mult_additive_angle)
{
  using C = Coefficients<Axis, Angle>;
  using M = Mean<C>;
  using A = TypedMatrix<C, C>;
  const auto f = [](const M& x) -> M { return A {1, 2, 3, 4} * x; };
  auto t = make_Transformation(f);
  EXPECT_TRUE(is_near(t(M(1, 0.5)), M(2, 5 - M_PI*2)));
  EXPECT_TRUE(is_near(t(M(1, 0.5)) + M(0.1, 0.1), M(2.1, 5.1 - M_PI*2)));
}

TEST_F(transformation_tests, Mult_augmented_axis)
{
  A_int2 a, an;
  a << 1, 2,
    4, 3;
  an << 3, 4,
    2, 1;
  const auto f = [&](const auto& in, const auto& ... n) { return strict(((a * in) + ... + (an * n))); };
  auto t = make_Transformation(f);
  EXPECT_EQ(t(M_int2(2, 3), M_int2(1, 1)), M_int2(15, 20));
  EXPECT_EQ(t(M_int2(2, 3), M_int2(3, 3)), M_int2(29, 26));
}

TEST_F(transformation_tests, Mult_augmented_angle)
{
  using C = Coefficients<Axis, Angle>;
  using M = Mean<C>;
  using A = TypedMatrix<C, C>;
  A a, an;
  a << 1, 2,
    4, 3;
  an << 3, 4,
    2, 1;
  const auto f = [&](const auto& in, const auto& ... n) { return strict(((a * in) + ... + (an * n))); };
  auto t = make_Transformation(f);
  EXPECT_TRUE(is_near(M(t(M(1, 0.5), M(0.1, 0.1))), M(2.7, 5.8 - M_PI*2)));
}

TEST_F(transformation_tests, Identity)
{
  IdentityTransformation t;
  EXPECT_EQ(t(M_int2 {2, 3}), M_int2(2, 3));
  EXPECT_EQ(t(M_int2 {2, 3}, M_int2 {1, 1}), M_int2(3, 4));
  EXPECT_EQ(std::get<0>(t.jacobian(M_int2 {2, 3})), A_int2::identity());
  EXPECT_EQ(std::get<0>(t.jacobian(M_int2 {2, 3}, M_int2 {1, 1})), A_int2::identity());
  EXPECT_EQ(std::get<1>(t.jacobian(M_int2 {2, 3}, M_int2 {1, 1})), A_int2::zero());
  EXPECT_EQ(std::get<0>(t.hessian(M_int2 {2, 3}, M_int2 {1, 1}))[0], A_int2::zero());
}

