/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformations.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::test;

using numbers::pi;

using M_int2 = Mean<Dimensions<2>, eigen_matrix_t<int, 2, 1>>;
using A_int2 = Matrix<Dimensions<2>, Dimensions<2>, eigen_matrix_t<int, 2, 2>>;

template<auto t, auto tn = t>
struct Scale
{
  template<typename Arg, typename ... Noise>
  auto operator()(const Arg& in, const Noise& ... n) const
  {
    return make_self_contained(((t * in) + ... + (tn * n)));
  }
};

TEST(transformations, Scale_no_noise_1D)
{
  using M_int1 = Mean<StaticDescriptor<Axis>, eigen_matrix_t<int, 1, 1>>;
  Transformation<Scale<3>> t;
  EXPECT_EQ(t(M_int1 {2}), M_int1 {6});
}

TEST(transformations, Scale_no_noise_2D)
{
  Transformation<Scale<7>> t;
  EXPECT_EQ(t(M_int2 {2, 3}), M_int2(14, 21));
}

TEST(transformations, Scale_additive)
{
  Transformation<Scale<7>> t;
  EXPECT_EQ(t(M_int2 {2, 3}), (M_int2 {14, 21}));
  EXPECT_EQ(t(M_int2(2, 3)) + M_int2(1, 1), M_int2(15, 22));
  static_assert(std::is_same_v<decltype(t(M_int2(2, 3)) + M_int2(1, 1)), decltype(M_int2(15, 22))>);
  EXPECT_EQ(t(M_int2(2, 3)) + M_int2(3, 3), M_int2(17, 24));
}

TEST(transformations, Scale_augmented)
{
  Transformation<Scale<7, 3>> t;
  EXPECT_EQ(t(M_int2 {2, 3}, M_int2 {1, 1}), M_int2(17, 24));
}

TEST(transformations, Mult_additive_axis)
{
  using M = Mean<Dimensions<2>>;
  using A = Matrix<Dimensions<2>, Dimensions<2>>;
  const auto f = [](const M& x) -> M { return A {1, 2, 3, 4} * x; };
  auto t = Transformation {f};
  EXPECT_TRUE(is_near(t(M(1, 0.5)) + M(0.1, 0.1), M(2.1, 5.1)));
}

TEST(transformations, Mult_additive_angle)
{
  using C = StaticDescriptor<Axis, angle::Radians>;
  using M = Mean<C>;
  using A = Matrix<C, C>;
  const auto f = [](const M& x) -> M { return A {1, 2, 3, 4} * x; };
  auto t = Transformation {f};
  EXPECT_TRUE(is_near(t(M(1, 0.5)), M(2, 5 - pi*2)));
  EXPECT_TRUE(is_near(t(M(1, 0.5)) + M(0.1, 0.1), M(2.1, 5.1 - pi*2)));
}

TEST(transformations, Mult_augmented_axis)
{
  A_int2 a, an;
  a << 1, 2,
    4, 3;
  an << 3, 4,
    2, 1;
  const auto f = [&](const auto& in, const auto& ... n) { return make_self_contained(((a * in) + ... + (an * n))); };
  auto t = Transformation {f};
  EXPECT_EQ(t(M_int2(2, 3), M_int2(1, 1)), M_int2(15, 20));
  EXPECT_EQ(t(M_int2(2, 3), M_int2(3, 3)), M_int2(29, 26));
}

TEST(transformations, Mult_augmented_angle)
{
  using C = StaticDescriptor<Axis, angle::Radians>;
  using M = Mean<C>;
  using A = Matrix<C, C>;
  A a, an;
  a << 1, 2,
    4, 3;
  an << 3, 4,
    2, 1;
  const auto f = [&](const auto& in, const auto& ... n) { return make_self_contained(((a * in) + ... + (an * n))); };
  auto t = Transformation {f};
  EXPECT_TRUE(is_near(M(t(M(1, 0.5), M(0.1, 0.1))), M(2.7, 5.8 - pi*2)));
}

TEST(transformations, Identity)
{
  IdentityTransformation t;
  EXPECT_EQ(t(M_int2 {2, 3}), M_int2(2, 3));
  EXPECT_EQ(t(M_int2 {2, 3}, M_int2 {1, 1}), M_int2(3, 4));
  EXPECT_EQ(std::get<0>(t.jacobian(M_int2 {2, 3})), make_identity_matrix_like<A_int2>());
  EXPECT_EQ(std::get<0>(t.jacobian(M_int2 {2, 3}, M_int2 {1, 1})), make_identity_matrix_like<A_int2>());
  EXPECT_EQ(std::get<1>(t.jacobian(M_int2 {2, 3}, M_int2 {1, 1})), make_zero<A_int2>());
}

