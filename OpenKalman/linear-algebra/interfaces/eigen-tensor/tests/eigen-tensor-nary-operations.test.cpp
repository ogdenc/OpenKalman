/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen-tensor.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;

namespace
{
  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 4>;
}


TEST(eigen_tensor, nullary_operations)
{
  auto a232 = make_dense_object<M22>(N2{}, N3{}, N2{});
  using A232 = decltype(a232);
  EXPECT_TRUE(is_near(a232.nullaryExpr([](auto x){ return 7; }), a232.constant(7)));

  static_assert(self_contained<decltype(a232.constant(7))>);
  using T2 = Eigen::TensorFixedSize<double, Eigen::Sizes<2,2>>;
  using I2 = Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, T2>;
  static_assert(constant_diagonal_coefficient_v<I2> == 1);
  using T2d = Eigen::Tensor<double, 2>;
  using I2d = Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, T2d>;
  static_assert(constant_diagonal_coefficient_v<I2d> == 1);
  EXPECT_EQ(constant_coefficient {a232.constant(7)}(), 7);
  EXPECT_EQ(constant_coefficient {a232.nullaryExpr([]{ return 7.; })}(), 7.);

  auto b3 = make_dense_object<A232>(3);
  using B3 = decltype(b3);
  b3.setValues({0, 1, 2});
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions<3>{}}, [](std::size_t i){return 3.5 + i;}), b3.constant(3.5) + b3));

  auto b23 = make_dense_object<B3>(2, 3);
  b23.setValues({{0, 1, 2},
                 {1, 2, 3}});
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, [](std::size_t i, std::size_t j){return 4.5 + i + j;}), b23.constant(4.5) + b23));

  auto b232 = make_dense_object<M22>(2, 3, 2);
  b232.setValues({{{0, 1}, {1, 2}, {2, 3}},
                  {{1, 2}, {2, 3}, {3, 4}}});
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions<2>{}, Dimensions<3>{}, Dimensions<2>{}}, [](std::size_t i, std::size_t j, std::size_t k){return 5.5 + i + j + k;}), a232.constant(5.5) + b232));
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions{2}, Dimensions<3>{}, Dimensions<2>{}}, [](std::size_t i, std::size_t j, std::size_t k){return 5.5 + i + j + k;}), a232.constant(5.5) + b232));
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions<2>{}, Dimensions{3}, Dimensions<2>{}}, [](std::size_t i, std::size_t j, std::size_t k){return 5.5 + i + j + k;}), a232.constant(5.5) + b232));
  EXPECT_TRUE(is_near(n_ary_operation<B3>(std::tuple {Dimensions{2}, Dimensions{3}, Dimensions<2>{}}, [](std::size_t i, std::size_t j, std::size_t k){return 5.5 + i + j + k;}), a232.constant(5.5) + b232));
  EXPECT_TRUE(is_near(n_ary_operation<M22>(std::tuple {Dimensions{2}, Dimensions{3}, Dimensions<2>{}}, [](std::size_t i, std::size_t j, std::size_t k){return 5.5 + i + j + k;}), a232.constant(5.5) + b232));


  /*
  // One operation for the entire matrix
  auto m23 = make_dense_object_from<M23>(5.5, 5.5, 5.5, 5.5, 5.5, 5.5);
  auto m23p = make_dense_object_from<M23>(0, 1, 2, 1, 2, 3);

  // One operation for each element
  m23 = make_dense_object_from<M23>(5.4, 5.5, 5.6, 5.7, 5.8, 5.9);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0, 1>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.4;}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}, []{return 5.8;}, []{return 5.9;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0, 1>([]{return 5.4;}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}, []{return 5.8;}, []{return 5.9;}), m23));

  m23p = make_dense_object_from<M23>(0, 0, 0, 1, 2, 3);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1, 0>(std::tuple {Dimensions<2>{}, Dimensions<3>{}}, []{return 5.4;}, [](std::size_t r, std::size_t c){return 5.7 + r + c;}, []{return 5.5;}, [](std::size_t r, std::size_t c){return 5.8 + r + c;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.9 + r + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1, 0>([]{return 5.4;}, [](std::size_t r, std::size_t c){return 5.7 + r + c;}, []{return 5.5;}, [](std::size_t r, std::size_t c){return 5.8 + r + c;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.9 + r + c;}), m23 + m23p));

  // One operation for each row
  m23 = make_dense_object_from<M23>(5.5, 5.5, 5.5, 5.6, 5.6, 5.6);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0>(std::tuple {Dimensions<2>{}, Dimensions{3}}, []{return 5.5;}, []{return 5.6;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0>([]{return 5.5;}, []{return 5.6;}), m23));

  m23p = make_dense_object_from<M23>(0, 1, 2, 0, 1, 2);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 0>(std::tuple {Dimensions<2>{}, Dimensions{3}}, [](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 0>([](std::size_t r, std::size_t c){return 5.5 + c;}, [](std::size_t r, std::size_t c){return 5.6 + c;}), m23 + m23p));

  // One operation for each column
  m23 = make_dense_object_from<M23>(5.5, 5.6, 5.7, 5.5, 5.6, 5.7);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1>(std::tuple {Dimensions{2}, Dimensions<3>{}}, []{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1>([]{return 5.5;}, []{return 5.6;}, []{return 5.7;}), m23));

  m23p = make_dense_object_from<M23>(0, 0, 0, 1, 0, 1);
  EXPECT_TRUE(is_near(n_ary_operation<Mxx, 1>(std::tuple {Dimensions{2}, Dimensions<3>{}}, [](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
  EXPECT_TRUE(is_near(n_ary_operation<M23, 1>([](std::size_t r, std::size_t c){return 5.5 + r;}, []{return 5.6;}, [](std::size_t r, std::size_t c){return 5.7 + r;}), m23 + m23p));
  */
}


/*TEST(eigen_tensor, unary_operations)
{
  auto a232 = make_dense_object<M22>(N2{}, N3{}, N2{});
  a232.setValues({{{1, 2}, {3, 4}, {5, 6}},
                  {{7, 8}, {9, 10}, {11, 12}}});
  auto b232 = make_dense_object<M22>(2, 3, 2);
  b232.setValues({{{2, 3}, {4, 5}, {6, 7}},
                  {{8, 9}, {10, 11}, {12, 13}}});

  EXPECT_TRUE(is_near(a232.unaryExpr([](auto x){ return x + 1; }), b232));
}*/
