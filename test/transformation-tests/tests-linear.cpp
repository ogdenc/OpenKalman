/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformation-tests.h"

using namespace OpenKalman;

using M2 = Mean<Axes<2>, Eigen::Matrix<int, 2, 1>>;
using A_int = TypedMatrix<Axes<2>, Axes<2>, Eigen::Matrix<int, 2, 2>>;

template<typename A, typename X>
struct Trans
{
  const A a;

  Trans(const A& a_): a(a_) {}

  auto operator()(const X& x) const
  {
    return a * x;
  }

  auto jacobian(const X& x) const
  {
    return std::tuple(a);
  }

  auto hessian(const X& x) const
  {
    using H = Eigen::Matrix<typename A::Scalar, A::ColsAtCompileTime, A::ColsAtCompileTime>;
    return std::tuple(std::array<H, A::RowsAtCompileTime>().fill(H::Zero()));
  }

};



TEST_F(transformation_tests, linear)
{
  A_int a;
  a << 1, 2, 3, 4;
  LinearTransformation t {a};
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(t(M2(1, 2), M2(1, 1)), M2(6, 12)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M2(1, 1))), M2::identity()));
}

TEST_F(transformation_tests, linearized)
{
  A_int a;
  a << 1, 2, 3, 4;
  Transformation<Axes<2>, Axes<2>, Trans<A_int, M2>> t(Trans<A_int, M2> {a});
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.function.jacobian(M2(1, 2))), a));
}


TEST_F(transformation_tests, linear_additive)
{
  A_int a;
  a << 1, 2,
       3, 4;
  LinearTransformation t {a};
  EXPECT_TRUE(is_near(t(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(t(M2(2, 3)), M2(8, 18)));
  Transformation<Axes<2>, Axes<2>, decltype(t)> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(tn(M2(2, 3)), M2(8, 18)));
}


TEST_F(transformation_tests, linear_augmented)
{
  A_int a, an;
  a << 1, 2, 4, 3;
  an << 3, 4, 2, 1;
  LinearTransformation t1 {a, an};
  EXPECT_TRUE(is_near(t1(M2(2, 3)), M2(8, 17)));
  EXPECT_TRUE(is_near(t1(M2(2, 3), M2(3, 3)), M2(29, 26)));
  EXPECT_TRUE(is_near(t1(M2(2, 3), M2(3, 3), M2(1, 1)), M2(30, 27)));
  Transformation<Axes<2>, Axes<2>, decltype(t1)> t2 {t1};
  EXPECT_TRUE(is_near(t2(M2(2, 3), M2(3, 3)), M2(29, 26)));
  EXPECT_TRUE(is_near(std::get<0>(t1.jacobian(M2(2, 3), M2(3, 3))), a));
  EXPECT_TRUE(is_near(std::get<1>(t1.jacobian(M2(2, 3), M2(3, 3))), an));
}


TEST_F(transformation_tests, linearized_additive)
{
  using A = TypedMatrix<Coefficients<Axis, Axis>, Coefficients<Axis, Axis>, Eigen::Matrix<int, 2, 2>>;
  A a;
  a << 1, 2,
      3, 4;
  using T = Trans<A, M2>;
  T t(a);
  Transformation<Axes<2>, Axes<2>, T> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(3, 3), M2(11, 21)));
  Transformation<Axes<2>, Axes<2>, T> tn2 {t};
  EXPECT_TRUE(is_near(tn2(M2(2, 3)) + M2(3, 3), M2(11, 21)));
}
