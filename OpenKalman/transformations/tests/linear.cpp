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

using M2 = Mean<Axes<2>, eigen_matrix_t<int, 2, 1>>;
using A2 = Matrix<Axes<2>, Axes<2>, eigen_matrix_t<int, 2, 2>>;

template<typename A, typename X>
struct Trans1
{
  const A a;

  Trans1(const A& a_): a(a_) {}

  auto operator()(const X& x) const
  {
    return a * x;
  }

  auto jacobian(const X& x) const
  {
    return std::tuple(a);
  }
};

template<typename A, typename X>
struct Trans2
{
  const A a;

  Trans2(const A& a_): a(a_) {}

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
    using H = eigen_matrix_t<scalar_type_of_t<A>, column_dimension_of_v<A>, column_dimension_of_v<A>>;
    using C = typename MatrixTraits<X>::RowCoefficients;
    using MH = Matrix<C, C, H>;
    auto Arr = std::array<MH, A::RowsAtCompileTime>();
    Arr.fill(make_zero_matrix_like<MH>());
    return std::tuple {Arr};
  }

};


TEST(transformations, linear)
{
  A2 a;
  a << 1, 2, 3, 4;
  LinearTransformation t {a};
  static_assert(linearized_function<decltype(t), 0>);
  static_assert(linearized_function<decltype(t), 1>);
  static_assert(not linearized_function<decltype(t), 2>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(t(M2(1, 2), M2(1, 1)), M2(6, 12)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M2(1, 1))), make_identity_matrix_like<A2>()));
}

TEST(transformations, linearized1)
{
  A2 a;
  a << 1, 2, 3, 4;
  using F = Trans1<A2, M2>;
  auto f = F {a};
  auto t = Transformation {f};
  using T = decltype(t);
  static_assert(linearized_function<F, 0>);
  static_assert(linearized_function<F, 1>);
  static_assert(not linearized_function<F, 2>);
  static_assert(not linearized_function<F, 3>);
  static_assert(linearized_function<T, 0>);
  static_assert(linearized_function<T, 1>);
  static_assert(not linearized_function<T, 2>);
  static_assert(not linearized_function<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST(transformations, linearized2)
{
  A2 a;
  a << 1, 2, 3, 4;
  using F = Trans2<A2, M2>;
  auto f = F {a};
  auto t = Transformation(f);
  using T = decltype(t);
  static_assert(linearized_function<F, 0>);
  static_assert(linearized_function<F, 1>);
  static_assert(linearized_function<F, 2>);
  static_assert(not linearized_function<F, 3>);
  static_assert(linearized_function<T, 0>);
  static_assert(linearized_function<T, 1>);
  static_assert(linearized_function<T, 2>);
  static_assert(not linearized_function<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST(transformations, linearized_lambdas)
{
  A2 a;
  a << 1, 2, 3, 4;
  auto f = [&a] (const M2& x) { return a * x; };
  auto j = [&a] (const M2& x) { return std::tuple(a); };
  auto h = [] (const M2& x) {
    using H = eigen_matrix_t<scalar_type_of_t<A2>, column_dimension_of_v<A2>, column_dimension_of_v<A2>>;
    using C = typename MatrixTraits<M2>::RowCoefficients;
    using MH = Matrix<C, C, H>;
    auto Arr = std::array<MH, A2::RowsAtCompileTime>();
    Arr.fill(make_zero_matrix_like<MH>());
    return std::tuple {Arr};
  };
  auto t = Transformation(f, j, h);
  using F = decltype(f);
  using T = decltype(t);
  static_assert(linearized_function<F, 0>);
  static_assert(not linearized_function<F, 1>);
  static_assert(not linearized_function<F, 2>);
  static_assert(linearized_function<T, 0>);
  static_assert(linearized_function<T, 1>);
  static_assert(linearized_function<T, 2>);
  static_assert(not linearized_function<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST(transformations, linear_additive)
{
  A2 a;
  a << 1, 2,
       3, 4;
  LinearTransformation t {a};
  EXPECT_TRUE(is_near(t(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(t(M2(2, 3)), M2(8, 18)));
  Transformation<decltype(t)> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(tn(M2(2, 3)), M2(8, 18)));
}


TEST(transformations, linear_augmented)
{
  A2 a, an;
  a << 1, 2, 4, 3;
  an << 3, 4, 2, 1;
  LinearTransformation t1 {a, an};
  EXPECT_TRUE(is_near(t1(M2(2, 3)), M2(8, 17)));
  EXPECT_TRUE(is_near(t1(M2(2, 3), M2(3, 3)), M2(29, 26)));
  EXPECT_TRUE(is_near(t1(M2(2, 3), M2(3, 3), M2(1, 1)), M2(30, 27)));
  Transformation<decltype(t1)> t2 {t1};
  EXPECT_TRUE(is_near(t2(M2(2, 3), M2(3, 3)), M2(29, 26)));
  EXPECT_TRUE(is_near(std::get<0>(t1.jacobian(M2(2, 3), M2(3, 3))), a));
  EXPECT_TRUE(is_near(std::get<1>(t1.jacobian(M2(2, 3), M2(3, 3))), an));
}


TEST(transformations, linearized_additive)
{
  A2 a;
  a << 1, 2,
       3, 4;
  using T = Trans2<A2, M2>;
  T t(a);
  Transformation<T> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(3, 3), M2(11, 21)));
  Transformation<T> tn2 {t};
  EXPECT_TRUE(is_near(tn2(M2(2, 3)) + M2(3, 3), M2(11, 21)));
}
