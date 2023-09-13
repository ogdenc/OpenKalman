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


TEST(eigen_tensor, Tensor)
{
  using T4 = Eigen::Tensor<double, 4>;
  static_assert(std::is_same_v<scalar_type_of_t<T4>, double>);
  static_assert(element_gettable<T4, 4>);
  static_assert(element_settable<T4, 4>);
  static_assert(writable<T4>);
  static_assert(dynamic_dimension<T4, 0>);
  static_assert(dynamic_dimension<T4, 1>);
  static_assert(dynamic_dimension<T4, 2>);
  static_assert(dynamic_dimension<T4, 3>);

  T4 t2222 {2, 2, 2, 2};
  t2222.setValues({{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}});

  EXPECT_EQ(get_element(t2222, 0, 0, 0, 0), 1);
  EXPECT_EQ(get_element(t2222, 1, 1, 1, 1), 16);
  EXPECT_EQ(get_element(t2222, 0, 1, 0, 1), 6);
  EXPECT_EQ(get_element(t2222, 1, 0, 1, 0), 11);
  set_element(t2222, 11.5, 1, 0, 1, 0);
  EXPECT_EQ(get_element(t2222, 1, 0, 1, 0), 11.5);

  auto t1234 = make_default_dense_writable_matrix_like<T4>(1, 2, 3, 4);
  static_assert(std::is_same_v<std::decay_t<decltype(t1234)>, T4>);

  t1234.setValues({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  EXPECT_EQ(get_element(t1234, 0, 0, 0, 0), 1);
  EXPECT_EQ(get_element(t1234, 0, 1, 2, 3), 24);
  EXPECT_EQ(get_element(t1234, 0, 0, 2, 3), 12);
  EXPECT_EQ(get_element(t1234, 0, 1, 1, 2), 19);
  set_element(t1234, 19.5, 0, 1, 1, 2);
  EXPECT_EQ(get_element(t1234, 0, 1, 1, 2), 19.5);

  static_assert(layout_of_v<T4> == Layout::left);
  static_assert(layout_of_v<Eigen::Tensor<double, 4, Eigen::RowMajor>> == Layout::right);
  static_assert(directly_accessible<T4>);
  EXPECT_EQ(*internal::raw_data(t2222), 1);
  EXPECT_EQ(internal::raw_data(t2222)[1], 9);
  EXPECT_EQ(internal::raw_data(t1234)[1], 13);
}


TEST(eigen_tensor, TensorFixedSize)
{
  using T2222 = Eigen::TensorFixedSize<double, Eigen::Sizes<2,2,2,2>>;

  static_assert(std::is_same_v<scalar_type_of_t<T2222>, double>);
  static_assert(element_gettable<T2222, 4>);
  static_assert(writable<T2222>);

  T2222 t2222;
  t2222.setValues({{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}}});

  EXPECT_EQ(get_element(t2222, 0, 0, 0, 0), 1);
  EXPECT_EQ(get_element(t2222, 1, 1, 1, 1), 16);
  EXPECT_EQ(get_element(t2222, 0, 1, 0, 1), 6);
  EXPECT_EQ(get_element(t2222, 1, 0, 1, 0), 11);
  set_element(t2222, 11.5, 1, 0, 1, 0);
  EXPECT_EQ(get_element(t2222, 1, 0, 1, 0), 11.5);

  auto t1234 = make_default_dense_writable_matrix_like<T2222>(N1{}, N2{}, N3{}, N4{});
  using T1234 = decltype(t1234);

  static_assert(index_dimension_of_v<T1234, 0> == 1);
  static_assert(index_dimension_of_v<T1234, 1> == 2);
  static_assert(index_dimension_of_v<T1234, 2> == 3);
  static_assert(index_dimension_of_v<T1234, 3> == 4);

  static_assert(element_gettable<T1234, 4>);
  static_assert(element_settable<T1234, 4>);
  static_assert(writable<T1234>);

  t1234.setValues({{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});

  EXPECT_EQ(get_element(t1234, 0, 0, 0, 0), 1);
  EXPECT_EQ(get_element(t1234, 0, 1, 2, 3), 24);
  EXPECT_EQ(get_element(t1234, 0, 0, 2, 3), 12);
  EXPECT_EQ(get_element(t1234, 0, 1, 1, 2), 19);
  set_element(t1234, 19.5, 0, 1, 1, 2);
  EXPECT_EQ(get_element(t1234, 0, 1, 1, 2), 19.5);

  static_assert(layout_of_v<T2222> == Layout::left);
  static_assert(layout_of_v<Eigen::TensorFixedSize<double, Eigen::Sizes<2,2,2,2>, Eigen::RowMajor>> == Layout::right);
  static_assert(directly_accessible<T2222>);
  EXPECT_EQ(*internal::raw_data(t2222), 1);
  EXPECT_EQ(internal::raw_data(t2222)[1], 9);
  EXPECT_EQ(internal::raw_data(t1234)[1], 13);
}


TEST(eigen_tensor, TensorMap)
{
  double s64[64] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  using S64_4d = Eigen::TensorMap<Eigen::Tensor<double, 4>>;
  S64_4d s64_4d {s64, 2, 4, 2, 4};

  static_assert(std::is_same_v<scalar_type_of_t<S64_4d>, double>);
  static_assert(element_gettable<S64_4d, 4>);
  static_assert(element_settable<S64_4d, 4>);
  static_assert(not writable<S64_4d>);
  static_assert(dynamic_dimension<S64_4d, 0>);
  static_assert(dynamic_dimension<S64_4d, 1>);
  static_assert(dynamic_dimension<S64_4d, 2>);
  static_assert(dynamic_dimension<S64_4d, 3>);
  static_assert(layout_of_v<S64_4d> == Layout::left);
  static_assert(directly_accessible<S64_4d>);

  EXPECT_EQ(get_element(s64_4d, 0, 0, 0, 0), 0);
  EXPECT_EQ(get_element(s64_4d, 1, 2, 1, 1), 29);
  set_element(s64_4d, 0.5, 0, 0, 0, 0);
  set_element(s64_4d, 29.5, 1, 2, 1, 1);
  EXPECT_EQ(get_element(s64_4d, 1, 2, 1, 1), 29.5);
  EXPECT_EQ(*internal::raw_data(s64_4d), 0.5);
  EXPECT_EQ(internal::raw_data(s64_4d)[29], 29.5);

  using S64_2d = Eigen::TensorMap<Eigen::TensorFixedSize<double, Eigen::Sizes<4, 16>, Eigen::RowMajor>>;
  S64_2d s64_2d {s64, 4, 16};

  static_assert(std::is_same_v<scalar_type_of_t<S64_2d>, double>);
  static_assert(element_gettable<S64_2d, 2>);
  static_assert(element_settable<S64_2d, 2>);
  static_assert(not writable<S64_2d>);
  static_assert(dynamic_dimension<S64_2d, 0>);
  static_assert(dynamic_dimension<S64_2d, 1>);
  static_assert(layout_of_v<S64_2d> == Layout::right);
  static_assert(directly_accessible<S64_2d>);

  EXPECT_EQ(get_element(s64_2d, 0, 0), 0.5);
  EXPECT_EQ(get_element(s64_2d, 1, 13), 29.5);
  set_element(s64_2d, 0.7, 0, 0);
  set_element(s64_2d, 29.7, 1, 13);
  EXPECT_EQ(get_element(s64_2d, 1, 13), 29.7);
  EXPECT_EQ(*internal::raw_data(s64_2d), 0.7);
  EXPECT_EQ(internal::raw_data(s64_2d)[29], 29.7);
}


TEST(eigen_tensor, tensor_to_matrix)
{
  const auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);

  using T23 = Eigen::TensorFixedSize<double, Eigen::Sizes<2,3>>;
  static_assert(layout_of_v<T23> == Layout::left);
  static_assert(Eigen3::eigen_dense_general<Eigen3::EigenWrapper<T23>, true>);
  T23 t23;
  t23.setValues({{1, 2, 3}, {4, 5, 6}});
  EXPECT_TRUE(is_near(Eigen3::EigenWrapper {t23}, m23));
  EXPECT_TRUE(is_near(Eigen3::EigenWrapper {t23} + m23.reverse(), M23::Constant(7)));
  EXPECT_TRUE(is_near(Eigen3::EigenWrapper {t23} * m23.transpose(), make_dense_writable_matrix_from<M22>(14, 32, 32, 77)));

  static_assert(not eigen_matrix_general<T23>);
  static_assert(not eigen_array_general<T23>);
  static_assert(directly_accessible<T23>);
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23), m23));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23) + m23.reverse(), M23::Constant(7)));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23) * m23.transpose(), make_dense_writable_matrix_from<M22>(14, 32, 32, 77)));

  using T23r = Eigen::TensorFixedSize<double, Eigen::Sizes<2,3>, Eigen::RowMajor>;
  static_assert(layout_of_v<T23r> == Layout::right);
  T23r t23r;
  t23r.setValues({{1, 2, 3}, {4, 5, 6}});
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r), m23));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r) + m23.reverse(), M23::Constant(7)));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r) * m23.transpose(), make_dense_writable_matrix_from<M22>(14, 32, 32, 77)));
}


TEST(eigen_tensor, matrix_to_tensor)
{
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<M22>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<I22>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<M22&>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<I22&>>()))>);

  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>&>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenTensorWrapper<Eigen::DiagonalWrapper<M31>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenTensorWrapper<Eigen::DiagonalWrapper<M31>&>>()))>);

  static_assert(std::is_same_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>>>())), Eigen::DiagonalMatrix<double, 3>&&>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);

  static_assert(Eigen3::eigen_tensor_general<Eigen3::EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>>>);

  static_assert(Eigen3::eigen_general<decltype(make_default_dense_writable_matrix_like<M11>(N1{}, N2{}))>);
  static_assert(not Eigen3::eigen_tensor_general<decltype(make_default_dense_writable_matrix_like<M11>(N1{}, N2{}))>);

  // Because there are four inputs, the result cannot be an Eigen matrix so this function returns an Eigen tensor:
  using T1234 = decltype(make_default_dense_writable_matrix_like<M11>(N1{}, N2{}, N3{}, N4{}));
  static_assert(Eigen3::eigen_tensor_general<T1234>);
  static_assert(index_dimension_of_v<T1234, 0> == 1);
  static_assert(index_dimension_of_v<T1234, 1> == 2);
  static_assert(index_dimension_of_v<T1234, 2> == 3);
  static_assert(index_dimension_of_v<T1234, 3> == 4);

  using T4mat = decltype(make_default_dense_writable_matrix_like<M11>(1, 2, 3, 4));
  static_assert(Eigen3::eigen_tensor_general<T4mat>);
  static_assert(dynamic_dimension<T4mat, 0>);
  static_assert(dynamic_dimension<T4mat, 1>);
  static_assert(dynamic_dimension<T4mat, 2>);
  static_assert(dynamic_dimension<T4mat, 3>);

  auto m23 = make_dense_writable_matrix_from<M23>(1, 2, 3, 4, 5, 6);
  using T23 = Eigen::TensorFixedSize<double, Eigen::Sizes<2,3>>;
  T23 t23;
  t23.setValues({{1, 2, 3}, {4, 5, 6}});
  EXPECT_TRUE(is_near(Eigen3::EigenTensorWrapper {m23}, t23));
  EXPECT_TRUE(is_near(Eigen3::EigenTensorWrapper {m23} + t23.reverse(std::array{true, true}), t23.constant(7)));

  using T32 = Eigen::TensorFixedSize<double, Eigen::Sizes<3,2>>;
  T32 t32;
  t32.setValues({{1, 4}, {2, 5}, {3, 6}});
  Eigen::TensorFixedSize<double, Eigen::Sizes<2,2>> t22;  t22.setValues({{14, 32}, {32, 77}});
  EXPECT_TRUE(is_near(Eigen3::EigenTensorWrapper {m23}.contract(t32, std::array{std::pair{1, 0}}), t22));
}

