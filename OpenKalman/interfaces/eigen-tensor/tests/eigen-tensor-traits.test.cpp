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


TEST(eigen_tensor, EigenTensorWrapper)
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
  static_assert(Eigen3::has_eigen_tensor_evaluator<Eigen3::EigenTensorWrapper<Eigen::DiagonalMatrix<double, 3>>, Eigen::DefaultDevice>);
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

  using N1 = std::integral_constant<std::size_t, 1>;
  using N2 = std::integral_constant<std::size_t, 2>;
  using N3 = std::integral_constant<std::size_t, 3>;
  using N4 = std::integral_constant<std::size_t, 4>;

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
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r), m23));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r) + m23.reverse(), M23::Constant(7)));
  EXPECT_TRUE(is_near(to_native_matrix<M23>(t23r) * m23.transpose(), make_dense_writable_matrix_from<M22>(14, 32, 32, 77)));


  //t23 = Eigen3::EigenTensorWrapper {m23};
  //t23 = make_dense_writable_matrix_from<T23>(1, 2, 3, 4, 5, 6);

}

