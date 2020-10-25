/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transformations.hpp"

using M2 = Mean<Axes<2>, Eigen::Matrix<int, 2, 1>>;
using A_int = Matrix<Axes<2>, Axes<2>, Eigen::Matrix<int, 2, 2>>;

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
    using H = Eigen::Matrix<typename A::Scalar, A::ColsAtCompileTime, A::ColsAtCompileTime>;
    using C = typename MatrixTraits<X>::RowCoefficients;
    using MH = Matrix<C, C, H>;
    auto Arr = std::array<MH, A::RowsAtCompileTime>();
    Arr.fill(MH::zero());
    return std::tuple {Arr};
  }

};



TEST_F(transformations, linear)
{
  A_int a;
  a << 1, 2, 3, 4;
  LinearTransformation t {a};
  static_assert(is_linearized_function_v<decltype(t), 0>);
  static_assert(is_linearized_function_v<decltype(t), 1>);
  static_assert(is_linearized_function_v<decltype(t), 2>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(t(M2(1, 2), M2(1, 1)), M2(6, 12)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
  EXPECT_TRUE(is_near(std::get<1>(t.jacobian(M2(1, 2), M2(1, 1))), M2::identity()));
}

TEST_F(transformations, linearized1)
{
  A_int a;
  a << 1, 2, 3, 4;
  using F = Trans1<A_int, M2>;
  auto f = F {a};
  auto t = make_Transformation(f);
  using T = decltype(t);
  static_assert(is_linearized_function_v<F, 0>);
  static_assert(is_linearized_function_v<F, 1>);
  static_assert(not is_linearized_function_v<F, 2>);
  static_assert(not is_linearized_function_v<F, 3>);
  static_assert(is_linearized_function_v<T, 0>);
  static_assert(is_linearized_function_v<T, 1>);
  static_assert(is_linearized_function_v<T, 2>);
  static_assert(not is_linearized_function_v<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST_F(transformations, linearized2)
{
  A_int a;
  a << 1, 2, 3, 4;
  using F = Trans2<A_int, M2>;
  auto f = F {a};
  auto t = Transformation(f);
  using T = decltype(t);
  static_assert(is_linearized_function_v<F, 0>);
  static_assert(is_linearized_function_v<F, 1>);
  static_assert(is_linearized_function_v<F, 2>);
  static_assert(not is_linearized_function_v<F, 3>);
  static_assert(is_linearized_function_v<T, 0>);
  static_assert(is_linearized_function_v<T, 1>);
  static_assert(is_linearized_function_v<T, 2>);
  static_assert(not is_linearized_function_v<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST_F(transformations, linearized_lambdas)
{
  A_int a;
  a << 1, 2, 3, 4;
  auto f = [&a] (const M2& x) { return a * x; };
  auto j = [&a] (const M2& x) { return std::tuple(a); };
  auto h = [] (const M2& x)
    {
      using H = Eigen::Matrix<typename A_int::Scalar, A_int::ColsAtCompileTime, A_int::ColsAtCompileTime>;
      using C = typename MatrixTraits<M2>::RowCoefficients;
      using MH = Matrix<C, C, H>;
      auto Arr = std::array<MH, A_int::RowsAtCompileTime>();
      Arr.fill(MH::zero());
      return std::tuple {Arr};
    };
  auto t = Transformation(f, j, h);
  using F = decltype(f);
  using T = decltype(t);
  static_assert(is_linearized_function_v<F, 0>);
  static_assert(not is_linearized_function_v<F, 1>);
  static_assert(not is_linearized_function_v<F, 2>);
  static_assert(is_linearized_function_v<T, 0>);
  static_assert(is_linearized_function_v<T, 1>);
  static_assert(is_linearized_function_v<T, 2>);
  static_assert(not is_linearized_function_v<T, 3>);
  EXPECT_TRUE(is_near(t(M2(1, 2)), M2(5, 11)));
  EXPECT_TRUE(is_near(std::get<0>(t.jacobian(M2(1, 2))), a));
}


TEST_F(transformations, linear_additive)
{
  A_int a;
  a << 1, 2,
       3, 4;
  LinearTransformation t {a};
  EXPECT_TRUE(is_near(t(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(t(M2(2, 3)), M2(8, 18)));
  Transformation<decltype(t)> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(2, 4), M2(10, 22)));
  EXPECT_TRUE(is_near(tn(M2(2, 3)), M2(8, 18)));
}


TEST_F(transformations, linear_augmented)
{
  A_int a, an;
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


TEST_F(transformations, linearized_additive)
{
  using A = Matrix<Coefficients<Axis, Axis>, Coefficients<Axis, Axis>, Eigen::Matrix<int, 2, 2>>;
  A a;
  a << 1, 2,
      3, 4;
  using T = Trans2<A, M2>;
  T t(a);
  Transformation<T> tn {t};
  EXPECT_TRUE(is_near(tn(M2(2, 3)) + M2(3, 3), M2(11, 21)));
  Transformation<T> tn2 {t};
  EXPECT_TRUE(is_near(tn2(M2(2, 3)) + M2(3, 3), M2(11, 21)));
}
