/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using numbers::pi;

namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_size, dynamic_size>;
  using M10 = eigen_matrix_t<double, 1, dynamic_size>;
  using M01 = eigen_matrix_t<double, dynamic_size, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_size>;
  using M02 = eigen_matrix_t<double, dynamic_size, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_size>;
  using M03 = eigen_matrix_t<double, dynamic_size, 3>;
  using M04 = eigen_matrix_t<double, dynamic_size, 4>;
  using M40 = eigen_matrix_t<double, 4, dynamic_size>;
  using M50 = eigen_matrix_t<double, 5, dynamic_size>;
  using M05 = eigen_matrix_t<double, dynamic_size, 5>;

  using cdouble = std::complex<double>;

  using CM11 = eigen_matrix_t<cdouble, 1, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM33 = eigen_matrix_t<cdouble, 3, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;

  using A11 = Eigen::Array<double, 1, 1>; static_assert(one_by_one_matrix<A11>);
  using A10 = Eigen::Array<double, 1, Eigen::Dynamic>;
  using A01 = Eigen::Array<double, Eigen::Dynamic, 1>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using A22 = Eigen::Array<double, 2, 2>;
  using A20 = Eigen::Array<double, 2, Eigen::Dynamic>;
  using A02 = Eigen::Array<double, Eigen::Dynamic, 2>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using CA22 = Eigen::Array<cdouble, 2, 2>;

  using I11 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A11>; static_assert(constant_coefficient_v<I11> == 1);
  static_assert(one_by_one_matrix<I11, Likelihood::maybe>);
  using I10 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A10>; static_assert(constant_coefficient_v<I10> == 1);
  using I01 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A01>; static_assert(constant_coefficient_v<I01> == 1);
  using I00 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A00>; static_assert(constant_coefficient_v<I00> == 1);
  using I22 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A22>;
  using I20 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A20>;

  using Z11 = decltype(std::declval<I11>() - std::declval<I11>());
  using Z22 = decltype(std::declval<I22>() - std::declval<I22>());
  using Z21 = Eigen::Replicate<Z11, 2, 1>;
  using Z23 = Eigen::Replicate<Z11, 2, 3>;
  using Z12 = Eigen::Replicate<Z11, 1, 2>;
  using Z20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic>;
  using Z02 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>;
  using Z00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>;
  using Z10 = Eigen::Replicate<Z11, 1, Eigen::Dynamic>;
  using Z01 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>;

  using C11_1 = I11;
  using C22_1 = Eigen::Replicate<C11_1, 2, 2>;
  using C21_1 = Eigen::Replicate<C11_1, 2, 1>;
  using C12_1 = Eigen::Replicate<C11_1, 1, 2>;
  using C20_1 = Eigen::Replicate<C11_1, 2, Eigen::Dynamic>;
  using C10_1 = Eigen::Replicate<C11_1, 1, Eigen::Dynamic>;
  using C01_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 1>;
  using C02_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 2>;
  using C00_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_2 = decltype(std::declval<I11>() + std::declval<I11>());
  using C22_2 = Eigen::Replicate<C11_2, 2, 2>;
  using C21_2 = Eigen::Replicate<C11_2, 2, 1>;
  using C12_2 = Eigen::Replicate<C11_2, 1, 2>;
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C10_2 = Eigen::Replicate<C11_2, 1, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 1>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_3 = decltype(std::declval<I11>() + std::declval<I11>() + std::declval<I11>());
  using C21_3 = Eigen::Replicate<C11_3, 2, 1>;
  using C20_3 = Eigen::Replicate<C11_3, 2, Eigen::Dynamic>;
  using C01_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, 1>;
  using C00_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m1 = decltype(-std::declval<I11>());
  using C21_m1 = Eigen::Replicate<C11_m1, 2, 1>;
  using C20_m1 = Eigen::Replicate<C11_m1, 2, Eigen::Dynamic>;
  using C01_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, 1>;
  using C00_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m2 = decltype(-(std::declval<I11>() + std::declval<I11>()));
  using C22_m2 = Eigen::Replicate<C11_m2, 2, 2>;
  using C21_m2 = Eigen::Replicate<C11_m2, 2, 1>;
  using C20_m2 = Eigen::Replicate<C11_m2, 2, Eigen::Dynamic>;
  using C01_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, 1>;
  using C00_m2 = Eigen::Replicate<C11_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_1_complex = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<cdouble>, Eigen::Array<cdouble, 1, 1>>;
  using C11_2_complex = decltype(std::declval<C11_1_complex>() + std::declval<C11_1_complex>());

  using B11 = Eigen::Array<bool, 1, 1>;
  using B22 = Eigen::Array<bool, 2, 2>;

  using B11_true = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<bool>, B11>;
  using B11_false = decltype(not std::declval<B11_true>());
  using B22_true = decltype(std::declval<B11_true>().replicate<2,2>());
  using B22_false = decltype(std::declval<B11_false>().replicate<2,2>());
  using BI22 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<bool>, B22>;

  using Cd22_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using Cd20_2 = Eigen::Replicate<Cd22_2, 1, Eigen::Dynamic>;
  using Cd02_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, 1>;
  using Cd00_2 = Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_3 = decltype(std::declval<I22>() + std::declval<I22>() + std::declval<I22>());
  using Cd20_3 = Eigen::Replicate<Cd22_3, 1, Eigen::Dynamic>;
  using Cd02_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, 1>;
  using Cd00_3 = Eigen::Replicate<Cd22_3, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m1 = decltype(-std::declval<I22>());
  using Cd20_m1 = Eigen::Replicate<Cd22_m1, 1, Eigen::Dynamic>;
  using Cd02_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, 1>;
  using Cd00_m1 = Eigen::Replicate<Cd22_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using Cd22_m2 = decltype(-std::declval<Cd22_2>());
  using Cd20_m2 = Eigen::Replicate<Cd22_m2, 1, Eigen::Dynamic>;
  using Cd02_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, 1>;
  using Cd00_m2 = Eigen::Replicate<Cd22_m2, Eigen::Dynamic, Eigen::Dynamic>;

  using DW2_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using DW0_2 = decltype(std::declval<I22>() + std::declval<I22>());
  using DW2_m2 = decltype(-std::declval<DW2_2>());

  using DM2 = EigenWrapper<Eigen::DiagonalMatrix<double, 2>>;
  using DM0 = EigenWrapper<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>;

  using Md21 = EigenWrapper<Eigen::DiagonalWrapper<M21>>;
  using Md20_1 = EigenWrapper<Eigen::DiagonalWrapper<M20>>;
  using Md01_2 = EigenWrapper<Eigen::DiagonalWrapper<M01>>;
  using Md00_21 = EigenWrapper<Eigen::DiagonalWrapper<M00>>;

  using Salv22 = EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Lower>>;
  using Salv20 = EigenWrapper<Eigen::SelfAdjointView<M20, Eigen::Lower>>;
  using Salv02 = EigenWrapper<Eigen::SelfAdjointView<M02, Eigen::Lower>>;
  using Salv00 = EigenWrapper<Eigen::SelfAdjointView<M00, Eigen::Lower>>;

  using Sauv22 = EigenWrapper<Eigen::SelfAdjointView<M22, Eigen::Upper>>;
  using Sauv20 = EigenWrapper<Eigen::SelfAdjointView<M20, Eigen::Upper>>;
  using Sauv02 = EigenWrapper<Eigen::SelfAdjointView<M02, Eigen::Upper>>;
  using Sauv00 = EigenWrapper<Eigen::SelfAdjointView<M00, Eigen::Upper>>;

  using Sadv22 = EigenWrapper<Eigen::SelfAdjointView<M22::IdentityReturnType, Eigen::Lower>>;
  using Sadv20 = EigenWrapper<Eigen::SelfAdjointView<M20::IdentityReturnType, Eigen::Lower>>;
  using Sadv02 = EigenWrapper<Eigen::SelfAdjointView<M02::IdentityReturnType, Eigen::Lower>>;
  using Sadv00 = EigenWrapper<Eigen::SelfAdjointView<M00::IdentityReturnType, Eigen::Lower>>;

  using Tlv22 = EigenWrapper<Eigen::TriangularView<M22, Eigen::Lower>>;
  using Tlv20 = EigenWrapper<Eigen::TriangularView<M20, Eigen::Lower>>;
  using Tlv02 = EigenWrapper<Eigen::TriangularView<M02, Eigen::Lower>>;
  using Tlv00 = EigenWrapper<Eigen::TriangularView<M00, Eigen::Lower>>;

  using Tuv22 = EigenWrapper<Eigen::TriangularView<M22, Eigen::Upper>>;
  using Tuv20 = EigenWrapper<Eigen::TriangularView<M20, Eigen::Upper>>;
  using Tuv02 = EigenWrapper<Eigen::TriangularView<M02, Eigen::Upper>>;
  using Tuv00 = EigenWrapper<Eigen::TriangularView<M00, Eigen::Upper>>;
}


TEST(eigen3, EigenWrapper)
{
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<M22>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<I22>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<M22&>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<I22&>>()))>);

  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenWrapper<Eigen::DiagonalMatrix<double, 3>&>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>>>()))>);
  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<EigenWrapper<Eigen::DiagonalWrapper<M31>&>>()))>);

  static_assert(std::is_same_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>())), Eigen::DiagonalMatrix<double, 3>&&>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen3::EigenWrapper<Eigen::DiagonalMatrix<double, 3>>>()))>);
}


TEST(eigen3, Eigen_Matrix)
{
  static_assert(Eigen3::native_eigen_matrix<M11>);
  static_assert(Eigen3::native_eigen_matrix<M00>);
  static_assert(Eigen3::native_eigen_general<M11>);
  static_assert(Eigen3::native_eigen_general<M00>);

  static_assert(max_indices_of_v<M11> == 2);
  static_assert(max_indices_of_v<M00> == 2);
  static_assert(max_indices_of_v<M21> == 2);
  static_assert(max_indices_of_v<Eigen::Matrix<double, 0, 0>> == 2);
  static_assert(max_indices_of_v<Eigen::Matrix<double, 2, 0>> == 2);
  static_assert(max_indices_of_v<Eigen::Matrix<double, 0, 2>> == 2);

  static_assert(max_tensor_order_of_v<M23> == 2);
  static_assert(max_tensor_order_of_v<M21> == 1);
  static_assert(max_tensor_order_of_v<M11> == 0);
  static_assert(max_tensor_order_of_v<M20> == 2);
  static_assert(max_tensor_order_of_v<M02> == 2);
  static_assert(max_tensor_order_of_v<M10> == 1);
  static_assert(max_tensor_order_of_v<M01> == 1);
  static_assert(max_tensor_order_of_v<M00> == 2);

  static_assert(index_dimension_of_v<M11, 0> == 1);
  static_assert(index_dimension_of_v<M21, 0> == 2);
  static_assert(index_dimension_of_v<M00, 0> == dynamic_size);
  static_assert(index_dimension_of_v<M11, 1> == 1);
  static_assert(index_dimension_of_v<M21, 1> == 1);
  static_assert(index_dimension_of_v<M00, 1> == dynamic_size);
  EXPECT_EQ(get_index_descriptor<0>(M11{}), 1);
  EXPECT_EQ(get_index_descriptor<0>(M21{}), 2);
  EXPECT_EQ((get_index_descriptor<0>(M00{2, 1})), 2);
  EXPECT_EQ((get_index_descriptor<1>(M11{})), 1);
  EXPECT_EQ((get_index_descriptor<1>(M21{})), 1);
  EXPECT_EQ((get_index_descriptor<1>(M00{2, 1})), 1);

  static_assert(std::is_same_v<typename interface::Elements<M00>::scalar_type, double>);

  static_assert(dynamic_rows<eigen_matrix_t<double, dynamic_size, dynamic_size>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, dynamic_size, dynamic_size>>);
  static_assert(dynamic_rows<eigen_matrix_t<double, dynamic_size, 1>>);
  static_assert(not dynamic_columns<eigen_matrix_t<double, dynamic_size, 1>>);
  static_assert(not dynamic_rows<eigen_matrix_t<double, 1, dynamic_size>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 1, dynamic_size>>);

  static_assert(number_of_dynamic_indices_v<M22> == 0);
  static_assert(number_of_dynamic_indices_v<M20> == 1);
  static_assert(number_of_dynamic_indices_v<M02> == 1);
  static_assert(number_of_dynamic_indices_v<M00> == 2);

  static_assert(std::is_same_v<dense_writable_matrix_t<M33>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<M30>, M30>);
  static_assert(std::is_same_v<dense_writable_matrix_t<M03>, M03>);
  static_assert(std::is_same_v<dense_writable_matrix_t<M00>, M00>);

  static_assert(writable<M22>);
  static_assert(writable<M20>);
  static_assert(writable<M02>);
  static_assert(writable<M00>);
  static_assert(writable<M22&>);
  static_assert(not writable<const M22>);
  static_assert(not writable<const M22&>);
  static_assert(writable<dense_writable_matrix_t<M22>>);

  static_assert(element_gettable<M32, 2>);
  static_assert(element_gettable<const M32, 2>);
  static_assert(element_gettable<M31, 2>);
  static_assert(element_gettable<M13, 2>);
  static_assert(element_gettable<M30, 2>);
  static_assert(element_gettable<M02, 2>);
  static_assert(element_gettable<M01, 2>);
  static_assert(element_gettable<M10, 2>);
  static_assert(element_gettable<M00, 2>);

  static_assert(element_gettable<M32, 1>);
  static_assert(element_gettable<M31, 1>);
  static_assert(element_gettable<M13, 1>);
  static_assert(element_gettable<M02, 1>);
  static_assert(element_gettable<M01, 1>);
  static_assert(element_gettable<M10, 1>);
  static_assert(element_gettable<M00, 1>);

  static_assert(element_settable<M32, 2>);
  static_assert(element_settable<M32&&, 2>);
  static_assert(not element_settable<const M32&, 2>);
  static_assert(element_settable<M31&, 2>);
  static_assert(element_settable<M13&, 2>);
  static_assert(element_settable<M30&, 2>);
  static_assert(element_settable<M02&, 2>);
  static_assert(element_settable<M01&, 2>);
  static_assert(element_settable<M10&, 2>);
  static_assert(element_settable<M00&, 2>);

  static_assert(element_settable<M32&, 1>);
  static_assert(element_settable<M31&, 1>);
  static_assert(element_settable<M13&, 1>);
  static_assert(not element_settable<const M31&, 1>);
  static_assert(element_settable<M02&, 1>);
  static_assert(element_settable<M01&, 1>);
  static_assert(element_settable<M10&, 1>);
  static_assert(element_settable<M00&, 1>);

  M22 m22; m22 << 1, 2, 3, 4;
  M23 m23; m23 << 1, 2, 3, 4, 5, 6;
  M03 m03_2 {2,3}; m03_2 << 1, 2, 3, 4, 5, 6;
  M32 m32; m32 << 1, 2, 3, 4, 5, 6;
  CM22 cm22; cm22 << cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1};

  EXPECT_TRUE(is_near(MatrixTraits<M22>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(m23), m23));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(m32), m32));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(m22), m22));
  EXPECT_TRUE(is_near(MatrixTraits<CM22>::make(cm22), cm22));
  static_assert(column_dimension_of_v<decltype(MatrixTraits<M20>::make(m23))> == 3);
  static_assert(row_dimension_of_v<decltype(MatrixTraits<M03>::make(m03_2))> == dynamic_size);

  EXPECT_TRUE(is_near(MatrixTraits<M22>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2, 3, 4, 5, 6), m23));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M20>::make(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2, 3, 4, 5, 6), m32));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2, 3, 4), m22));
  EXPECT_TRUE(is_near(MatrixTraits<M02>::make(1, 2), M12 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<M00>::make(1, 2), M21 {1, 2}));
  EXPECT_TRUE(is_near(MatrixTraits<CM22>::make(cdouble {1,4}, cdouble {2,3}, cdouble {3,2}, cdouble {4,1}), cm22));
  static_assert(column_dimension_of_v<decltype(MatrixTraits<M20>::make(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(MatrixTraits<M02>::make(1, 2))> == 1);
  static_assert(row_dimension_of_v<decltype(MatrixTraits<M00>::make(1, 2))> == 2);

  static_assert(std::is_same_v<index_descriptor_of<M11, 0>::type, Dimensions<1>>);
  static_assert(std::is_same_v<index_descriptor_of<M11, 1>::type, Dimensions<1>>);
  static_assert(equivalent_to<index_descriptor_of_t<M11, 0>, Axis>);
  static_assert(equivalent_to<index_descriptor_of_t<M11, 1>, Axis>);
  static_assert(std::is_same_v<index_descriptor_of<M22, 0>::type, Dimensions<2>>);
  static_assert(std::is_same_v<index_descriptor_of<M22, 1>::type, Dimensions<2>>);
  static_assert(equivalent_to<index_descriptor_of_t<M22, 0>, TypedIndex<Axis, Axis>>);
  static_assert(equivalent_to<index_descriptor_of_t<M22, 1>, TypedIndex<Axis, Axis>>);

  static_assert(maybe_index_descriptors_match<M22, M20, M02, M00>);
  static_assert(index_descriptors_match<M22, CM22, M22>);
  EXPECT_TRUE(get_index_descriptors_match(m22, cm22, M20{m22}, M02{m22}, M00{m22}));

  static_assert(compatible_with_index_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 3>>);
  static_assert(not compatible_with_index_descriptors<M23, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_index_descriptors<M20, std::integral_constant<int, 2>, std::integral_constant<int, 2>>);
  static_assert(compatible_with_index_descriptors<M20, std::integral_constant<int, 2>, int>);
  static_assert(not compatible_with_index_descriptors<M20, std::integral_constant<int, 3>, int>);

  static_assert(square_matrix<M11, Likelihood::maybe>);
  static_assert(square_matrix<M22, Likelihood::maybe>);
  static_assert(not square_matrix<M32, Likelihood::maybe>);
  static_assert(square_matrix<M20, Likelihood::maybe>);
  static_assert(square_matrix<M02, Likelihood::maybe>);
  static_assert(square_matrix<M00, Likelihood::maybe>);
  static_assert(square_matrix<CM22, Likelihood::maybe>);
  static_assert(not square_matrix<CM32, Likelihood::maybe>);
  static_assert(square_matrix<CM20, Likelihood::maybe>);
  static_assert(square_matrix<CM02, Likelihood::maybe>);
  static_assert(square_matrix<CM00, Likelihood::maybe>);

  static_assert(square_matrix<M11>);
  static_assert(square_matrix<M22>);
  static_assert(not square_matrix<M20>);
  static_assert(not square_matrix<M02>);
  static_assert(not square_matrix<M00>);
  static_assert(square_matrix<CM22>);
  static_assert(not square_matrix<CM20>);
  static_assert(not square_matrix<CM02>);
  static_assert(not square_matrix<CM00>);

  M11 m11_1{1};
  Eigen::Matrix<double, 0, 0> m0;

  static_assert(get_is_square(m22));
  static_assert(*get_is_square(m22) == Dimensions<2>{});
  EXPECT_TRUE(get_is_square(M20{m22}));
  EXPECT_TRUE(get_is_square(M02{m22}));
  EXPECT_TRUE(get_is_square(M00{m22}));
  EXPECT_TRUE(*get_is_square(M00{m22}) == 2);
  static_assert(get_is_square(m11_1));
  static_assert(*get_is_square(m11_1) == Dimensions<1>{});
  EXPECT_TRUE(get_is_square(M10{m11_1}));
  EXPECT_TRUE(get_is_square(M01{m11_1}));
  EXPECT_TRUE(get_is_square(M00{m11_1}));
  EXPECT_TRUE(*get_is_square(M00{m11_1}) == 1);
  static_assert(not get_is_square(m0));

  static_assert(one_by_one_matrix<M11>);
  static_assert(not one_by_one_matrix<M10>);
  static_assert(one_by_one_matrix<M10, Likelihood::maybe>);

  static_assert(get_is_one_by_one(m11_1));
  EXPECT_TRUE(get_is_one_by_one(M10{m11_1}));
  EXPECT_TRUE(get_is_one_by_one(M01{m11_1}));
  EXPECT_TRUE(get_is_one_by_one(M00{m11_1}));
  static_assert(not get_is_one_by_one(m0));

  static_assert(dimension_size_of_index_is<M31, 1, 1>);
  static_assert(dimension_size_of_index_is<M01, 1, 1>);
  static_assert(dimension_size_of_index_is<M30, 1, 1, Likelihood::maybe>);
  static_assert(dimension_size_of_index_is<M00, 1, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M32, 1, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M02, 1, 1, Likelihood::maybe>);

  static_assert(dimension_size_of_index_is<M13, 0, 1>);
  static_assert(dimension_size_of_index_is<M10, 0, 1>);
  static_assert(dimension_size_of_index_is<M03, 0, 1, Likelihood::maybe>);
  static_assert(dimension_size_of_index_is<M00, 0, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M23, 0, 1, Likelihood::maybe>);
  static_assert(not dimension_size_of_index_is<M20, 0, 1, Likelihood::maybe>);

  static_assert(native_eigen_matrix<M22>);
  static_assert(native_eigen_matrix<M00>);
  static_assert(native_eigen_matrix<CM22>);
  static_assert(native_eigen_matrix<CM00>);
  static_assert(not native_eigen_matrix<double>);

  //static_assert(modifiable<M33, M33>);
  //static_assert(not modifiable<M33, M31>);
  //static_assert(not modifiable<M33, eigen_matrix_t<int, 3, 3>>);
  //static_assert(not modifiable<const M33, M33>);
  //static_assert(modifiable<M33, Eigen3::IdentityMatrix<M33>>);
}


TEST(eigen3, Eigen_check_test_classes)
{
  static_assert(not constant_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);
  static_assert(constant_diagonal_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>>);

  static_assert(constant_matrix<Z21, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::known>);
  static_assert(not scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::unknown>);

  static_assert(constant_matrix<M11, CompileTimeStatus::unknown>);
  static_assert(constant_matrix<M10, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(constant_matrix<M01, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(constant_matrix<M00, CompileTimeStatus::unknown, Likelihood::maybe>);
  static_assert(not constant_matrix<M21, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_matrix<M21, CompileTimeStatus::unknown, Likelihood::maybe>);
  EXPECT_EQ(constant_coefficient{make_dense_writable_matrix_from<M11>(5.5)}(), 5.5);
  static_assert(constant_diagonal_matrix<M00, CompileTimeStatus::unknown, Likelihood::maybe>);
  EXPECT_EQ(constant_diagonal_coefficient{make_dense_writable_matrix_from<M11>(5.5)}(), 5.5);

  static_assert(constant_coefficient_v<Z21> == 0);
  static_assert(constant_coefficient_v<Z12> == 0);
  static_assert(constant_coefficient_v<Z23> == 0);
  static_assert(constant_coefficient_v<Z20> == 0);
  static_assert(constant_coefficient_v<Z02> == 0);

  static_assert(constant_coefficient_v<Z00> == 0);
  static_assert(constant_coefficient_v<Z01> == 0);
  static_assert(constant_coefficient_v<C11_1> == 1);
  static_assert(constant_coefficient_v<C11_m1> == -1);
  static_assert(constant_coefficient_v<C11_2> == 2);
  static_assert(constant_coefficient_v<C11_m2> == -2);
  static_assert(constant_coefficient_v<C11_3> == 3);
  static_assert(constant_coefficient_v<C20_2> == 2);
  static_assert(constant_coefficient_v<C02_2> == 2);
  static_assert(constant_coefficient_v<C00_2> == 2);
  static_assert(constant_coefficient_v<B22_true> == true);
  static_assert(constant_coefficient_v<B22_false> == false);
  static_assert(not constant_matrix<Cd22_2>);
  static_assert(not constant_matrix<Cd20_2>);
  static_assert(not constant_matrix<Cd02_2>);
  static_assert(not constant_matrix<Cd22_3>);
  static_assert(not constant_matrix<Cd22_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd20_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd02_2, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd22_3, CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(constant_diagonal_matrix<Z11, CompileTimeStatus::known>);
  static_assert(constant_diagonal_matrix<Z10, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z20, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z02, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z01, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z00, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Z01>);
  static_assert(not constant_diagonal_matrix<Z00>);
  static_assert(not constant_diagonal_matrix<Z21, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Z12, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Z23, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<C11_m1> == -1);
  static_assert(constant_diagonal_coefficient_v<C11_2> == 2);
  static_assert(constant_diagonal_coefficient_v<C11_m2> == -2);
  static_assert(constant_diagonal_matrix<C11_1>);
  static_assert(not constant_diagonal_matrix<C21_1, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<C20_1, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<C10_1>);
  static_assert(not constant_diagonal_matrix<C01_1>);
  static_assert(not constant_diagonal_matrix<C00_1>);
  static_assert(constant_diagonal_matrix<C11_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C10_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C01_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C00_1, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<I22> == 1);
  static_assert(constant_diagonal_coefficient_v<I20> == 1);
  static_assert(constant_diagonal_coefficient_v<Cd22_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd20_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd02_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd00_2> == 2);

  static_assert(zero_matrix<Z21>);
  static_assert(zero_matrix<Eigen::DiagonalWrapper<Z21>>);
  static_assert(zero_matrix<Z23>);
  static_assert(zero_matrix<Z20>);
  static_assert(zero_matrix<Z02>);
  static_assert(zero_matrix<B22_false>);
  static_assert(not zero_matrix<Cd22_2>);
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z00>);

  static_assert(not identity_matrix<C21_1, Likelihood::maybe>);
  static_assert(not identity_matrix<C20_1, Likelihood::maybe>);
  static_assert(identity_matrix<I22>);
  static_assert(identity_matrix<I20, Likelihood::maybe>);
  static_assert(not identity_matrix<Cd22_2, Likelihood::maybe>);
  static_assert(not identity_matrix<Cd22_3, Likelihood::maybe>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<C10_1>);
  static_assert(not identity_matrix<C01_1>);
  static_assert(not identity_matrix<C00_1>);
  static_assert(not identity_matrix<C21_1, Likelihood::maybe>);
  static_assert(not identity_matrix<C20_1, Likelihood::maybe>);
  static_assert(identity_matrix<C10_1, Likelihood::maybe>);
  static_assert(identity_matrix<C01_1, Likelihood::maybe>);
  static_assert(identity_matrix<C00_1, Likelihood::maybe>);

  static_assert(one_by_one_matrix<Z01, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Z01>);
  static_assert(one_by_one_matrix<C11_1>);
  static_assert(one_by_one_matrix<C10_1, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<C10_1>);
  static_assert(one_by_one_matrix<C00_1, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<C00_1>);
  static_assert(one_by_one_matrix<C11_m1>);
  static_assert(one_by_one_matrix<Eigen::DiagonalWrapper<M11>>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M10>>);
  static_assert(one_by_one_matrix<Eigen::DiagonalWrapper<M10>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M01>>);
  static_assert(one_by_one_matrix<Eigen::DiagonalWrapper<M01>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M00>>);
  static_assert(one_by_one_matrix<Eigen::DiagonalWrapper<M00>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M22>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M20>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::DiagonalWrapper<M02>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Cd22_2, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Cd20_2, Likelihood::maybe>);
  static_assert(one_by_one_matrix<EigenWrapper<Eigen::DiagonalWrapper<C01_1>>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<EigenWrapper<Eigen::DiagonalWrapper<C10_1>>, Likelihood::maybe>);

  static_assert(square_matrix<M22>);
  static_assert(square_matrix<M20, Likelihood::maybe>);
  static_assert(square_matrix<M02, Likelihood::maybe>);
  static_assert(square_matrix<M00, Likelihood::maybe>);
  static_assert(square_matrix<M11>);
  static_assert(square_matrix<M10, Likelihood::maybe>);
  static_assert(square_matrix<M01, Likelihood::maybe>);

  static_assert(square_matrix<C11_m1, Likelihood::maybe>);
  static_assert(square_matrix<Z22, Likelihood::maybe>);
  static_assert(square_matrix<Z20, Likelihood::maybe>);
  static_assert(square_matrix<Z02, Likelihood::maybe>);
  static_assert(square_matrix<Z00, Likelihood::maybe>);
  static_assert(square_matrix<C22_2, Likelihood::maybe>);
  static_assert(square_matrix<C20_2, Likelihood::maybe>);
  static_assert(square_matrix<C02_2, Likelihood::maybe>);
  static_assert(square_matrix<C00_2, Likelihood::maybe>);
  static_assert(square_matrix<DM2, Likelihood::maybe>);
  static_assert(square_matrix<DM0, Likelihood::maybe>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M11>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M10>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M01>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M00>>);

  static_assert(square_matrix<C11_m1>);
  static_assert(square_matrix<Z22>);
  static_assert(not square_matrix<Z20>);
  static_assert(not square_matrix<Z02>);
  static_assert(not square_matrix<Z00>);
  static_assert(square_matrix<C22_2>);
  static_assert(not square_matrix<C20_2>);
  static_assert(not square_matrix<C02_2>);
  static_assert(not square_matrix<C00_2>);
  static_assert(square_matrix<DM2>);
  static_assert(square_matrix<DM0>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M11>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M10>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M01>>);
  static_assert(square_matrix<Eigen::DiagonalWrapper<M00>>);

  static_assert(square_matrix<Tlv22>);
  static_assert(square_matrix<Tlv20, Likelihood::maybe>);
  static_assert(square_matrix<Tlv02, Likelihood::maybe>);
  static_assert(square_matrix<Tlv00, Likelihood::maybe>);
  static_assert(not square_matrix<Tlv20>);
  static_assert(not square_matrix<Tlv02>);
  static_assert(not square_matrix<Tlv00>);

  static_assert(square_matrix<Salv22>);
  static_assert(square_matrix<Salv20, Likelihood::maybe>);
  static_assert(square_matrix<Salv02, Likelihood::maybe>);
  static_assert(square_matrix<Salv00, Likelihood::maybe>);
  static_assert(not square_matrix<Salv20>);
  static_assert(not square_matrix<Salv02>);
  static_assert(not square_matrix<Salv00>);

  static_assert(diagonal_matrix<Z22>);
  static_assert(not diagonal_matrix<Z20>);
  static_assert(not diagonal_matrix<Z02>);
  static_assert(not diagonal_matrix<Z00>);
  static_assert(diagonal_matrix<Z22, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z20, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z02, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z00, Likelihood::maybe>);
  static_assert(diagonal_matrix<C11_2>);
  static_assert(diagonal_matrix<I22>);
  static_assert(diagonal_matrix<I20, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cd22_2>);
  static_assert(diagonal_matrix<Cd20_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cd02_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Cd00_2, Likelihood::maybe>);
  static_assert(diagonal_matrix<Md21>);
  static_assert(diagonal_matrix<Md20_1>);
  static_assert(diagonal_matrix<Md01_2>);
  static_assert(diagonal_matrix<Md00_21>);
  static_assert(not diagonal_matrix<Salv22>);
  static_assert(not diagonal_matrix<Salv20>);
  static_assert(not diagonal_matrix<Salv02>);
  static_assert(not diagonal_matrix<Salv00>);
  static_assert(not diagonal_matrix<Sauv22>);
  static_assert(not diagonal_matrix<Sauv20>);
  static_assert(not diagonal_matrix<Sauv02>);
  static_assert(not diagonal_matrix<Sauv00>);
  static_assert(diagonal_matrix<Sadv22>);
  static_assert(diagonal_matrix<Sadv20, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadv02, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadv00, Likelihood::maybe>);
  static_assert(diagonal_matrix<M11>);
  static_assert(not diagonal_matrix<M10>);
  static_assert(not diagonal_matrix<M01>);
  static_assert(not diagonal_matrix<M00>);
  static_assert(diagonal_matrix<M11, Likelihood::maybe>);
  static_assert(diagonal_matrix<M10, Likelihood::maybe>);
  static_assert(diagonal_matrix<M01, Likelihood::maybe>);
  static_assert(diagonal_matrix<M00, Likelihood::maybe>);

  static_assert(not diagonal_adapter<M11>);
  static_assert(not diagonal_adapter<M10>);
  static_assert(not diagonal_adapter<M01>);
  static_assert(not diagonal_adapter<M00>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M20>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M20>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M01>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M10>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M10>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M00>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M00>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M20>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M20>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M01>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M10>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M10>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M00>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M00>, Likelihood::maybe>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M22>, Likelihood::maybe>);

  static_assert(triangular_matrix<Z22, TriangleType::lower>);
  static_assert(not triangular_matrix<Z20, TriangleType::lower>);
  static_assert(not triangular_matrix<Z02, TriangleType::lower>);
  static_assert(not triangular_matrix<Z00, TriangleType::lower>);
  static_assert(triangular_matrix<Z20, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Z02, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Z00, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<C11_2, TriangleType::lower>);
  static_assert(not triangular_matrix<C22_2, TriangleType::lower>);
  static_assert(not triangular_matrix<C22_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<I22, TriangleType::lower>);
  static_assert(triangular_matrix<I20, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::lower>);
  static_assert(triangular_matrix<Cd20_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd02_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd00_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Md21, TriangleType::lower>);
  static_assert(triangular_matrix<Md20_1, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Md01_2, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Md00_21, TriangleType::lower, Likelihood::maybe>);

  static_assert(triangular_matrix<Z22, TriangleType::upper>);
  static_assert(not triangular_matrix<Z20, TriangleType::upper>);
  static_assert(not triangular_matrix<Z02, TriangleType::upper>);
  static_assert(not triangular_matrix<Z00, TriangleType::upper>);
  static_assert(triangular_matrix<Z20, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Z02, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Z00, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<C11_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper>);
  static_assert(not triangular_matrix<C22_2, TriangleType::upper, Likelihood::maybe>);

  static_assert(triangular_matrix<I22, TriangleType::upper>);
  static_assert(triangular_matrix<I20, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd22_2, TriangleType::upper>);
  static_assert(triangular_matrix<Cd20_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd02_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Cd00_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Md21, TriangleType::upper>);
  static_assert(triangular_matrix<Md20_1, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Md01_2, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Md00_21, TriangleType::upper, Likelihood::maybe>);

  static_assert(triangular_matrix<Tlv22, TriangleType::lower>);
  static_assert(triangular_matrix<Tlv20, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlv02, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlv00, TriangleType::lower, Likelihood::maybe>);
  static_assert(not triangular_matrix<Tlv20, TriangleType::lower>);
  static_assert(not triangular_matrix<Tlv02, TriangleType::lower>);
  static_assert(not triangular_matrix<Tlv00, TriangleType::lower>);
  static_assert(not triangular_matrix<Tuv22, TriangleType::lower>);

  static_assert(triangular_matrix<Tuv22, TriangleType::upper>);
  static_assert(triangular_matrix<Tuv20, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuv02, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuv00, TriangleType::upper, Likelihood::maybe>);
  static_assert(not triangular_matrix<Tuv20, TriangleType::upper>);
  static_assert(not triangular_matrix<Tuv02, TriangleType::upper>);
  static_assert(not triangular_matrix<Tuv00, TriangleType::upper>);
  static_assert(not triangular_matrix<Tlv22, TriangleType::upper>);

  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<Z20, Likelihood::maybe>);
  static_assert(hermitian_matrix<Z02, Likelihood::maybe>);
  static_assert(hermitian_matrix<Z00, Likelihood::maybe>);
  static_assert(not hermitian_adapter<Z22>);
  static_assert(not hermitian_adapter<Z20>);
  static_assert(not hermitian_adapter<Z02>);
  static_assert(not hermitian_adapter<Z00>);
  static_assert(hermitian_matrix<C22_2>);
  static_assert(hermitian_matrix<I22>);
  static_assert(hermitian_matrix<I20, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cd22_2>);
  static_assert(hermitian_matrix<Cd20_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cd02_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Cd00_2, Likelihood::maybe>);
  static_assert(hermitian_matrix<Md21>);
  static_assert(hermitian_matrix<Md20_1>);
  static_assert(hermitian_matrix<Md01_2>);
  static_assert(hermitian_matrix<Md00_21>);
  static_assert(not hermitian_adapter<C22_2>);
  static_assert(not hermitian_adapter<I22>);
  static_assert(not hermitian_adapter<I20>);
  static_assert(not hermitian_adapter<Cd22_2>);
  static_assert(not hermitian_adapter<Cd20_2>);
  static_assert(not hermitian_adapter<Cd02_2>);
  static_assert(not hermitian_adapter<Cd00_2>);
  static_assert(not hermitian_adapter<Md21>);
  static_assert(not hermitian_adapter<Md20_1>);
  static_assert(not hermitian_adapter<Md01_2>);
  static_assert(not hermitian_adapter<Md00_21>);

  static_assert(hermitian_adapter<nested_matrix_of_t<Salv22>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv20>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv02>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv00>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv20>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv02>, HermitianAdapterType::lower>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Salv00>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<nested_matrix_of_t<Sauv22>, HermitianAdapterType::lower>);

  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv22>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv20>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv02>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv00>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv20>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv02>, HermitianAdapterType::upper>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sauv00>, HermitianAdapterType::upper>);
  static_assert(not hermitian_adapter<nested_matrix_of_t<Salv22>, HermitianAdapterType::upper>);

  static_assert(hermitian_adapter<nested_matrix_of_t<Sadv22>, HermitianAdapterType::any>);
  static_assert(hermitian_adapter<nested_matrix_of_t<Sadv22>, HermitianAdapterType::any>);

  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Salv22>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Salv22>, nested_matrix_of_t<Salv22>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Salv20>, nested_matrix_of_t<Salv02>, nested_matrix_of_t<Salv00>> == HermitianAdapterType::lower);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Sauv22>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Sauv22>, nested_matrix_of_t<Sauv22>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Sauv20>, nested_matrix_of_t<Sauv02>, nested_matrix_of_t<Sauv00>> == HermitianAdapterType::upper);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Sauv22>, nested_matrix_of_t<Salv22>> == HermitianAdapterType::any);
  static_assert(hermitian_adapter_type_of_v<nested_matrix_of_t<Salv22>, nested_matrix_of_t<Sauv22>> == HermitianAdapterType::any);

  static_assert(maybe_has_same_shape_as<>);
  static_assert(maybe_has_same_shape_as<M32>);
  static_assert(maybe_has_same_shape_as<M32, M02, M30>);
  static_assert(maybe_has_same_shape_as<M20, M23, M03>);
  static_assert(maybe_has_same_shape_as<M20, Z23>);
  static_assert(maybe_has_same_shape_as<M20, M03>);

  static_assert(has_same_shape_as<M32, M32>);
  static_assert(not has_same_shape_as<M32, M02>);
  static_assert(not has_same_shape_as<M02, M32>);
  static_assert(has_same_shape_as<M22, Salv22, M22, Z22>);
}


TEST(eigen3, Eigen_Array)
{
  static_assert(native_eigen_array<Eigen::Array<double, 3, 2>>);
  static_assert(not native_eigen_matrix<Eigen::Array<double, 3, 2>>);
  static_assert(self_contained<Eigen::Array<double, 3, 2>>);
  static_assert(row_dimension_of_v<Eigen::Array<double, 3, 2>> == 3);
  static_assert(column_dimension_of_v<Eigen::Array<double, 3, 2>> == 2);
  static_assert(not square_matrix<Eigen::Array<double, 2, 1>>);
}


TEST(eigen3, Eigen_ArrayWrapper)
{
  static_assert(native_eigen_array<Eigen::ArrayWrapper<M32>>);
  static_assert(not native_eigen_matrix<Eigen::ArrayWrapper<M32>>);
  static_assert(self_contained<Eigen::ArrayWrapper<I22>>);
  static_assert(not self_contained<Eigen::ArrayWrapper<M32>>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen::ArrayWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen::ArrayWrapper<I22>>()))>);

  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C22_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C20_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C02_2>> == 2);
  static_assert(constant_coefficient_v<Eigen::ArrayWrapper<C00_2>> == 2);

  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd22_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd20_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd02_2>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<Cd00_2>> == 2);

  static_assert(zero_matrix<Eigen::ArrayWrapper<Z22>>);
  static_assert(zero_matrix<Eigen::ArrayWrapper<Z21>>);
  static_assert(zero_matrix<Eigen::ArrayWrapper<Z23>>);

  static_assert(diagonal_matrix<Md21>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<Z22>>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<C22_2>>);
  static_assert(triangular_matrix<Md21, TriangleType::lower>);
  static_assert(triangular_matrix<Md21, TriangleType::upper>);
}


TEST(eigen3, Eigen_Block)
{
  static_assert(native_eigen_matrix<decltype(std::declval<C22_2>().matrix().block<2,1>(0, 0))>);
  static_assert(native_eigen_array<decltype(std::declval<C22_2>().block<2,1>(0, 0))>);
  static_assert(not native_eigen_matrix<decltype(std::declval<C22_2>().block<2,1>(0, 0))>);

  static_assert(self_contained<decltype(std::declval<I22>().block<2,1>(0, 0))>);
  static_assert(not self_contained<decltype(std::declval<Eigen::ArrayWrapper<M32>>().block<2,1>(0, 0))>);
  static_assert(self_contained<decltype((2 * std::declval<I22>() + std::declval<I22>()).col(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I22>() + A22 {1, 2, 3, 4}).col(0))>);
  static_assert(self_contained<decltype((2 * std::declval<I22>() + std::declval<I22>()).row(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I22>() + A22 {1, 2, 3, 4}).row(0))>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block<2, 1>(0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().block(2, 1, 0, 0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 2>(0, 0))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().block<1, 1>(0, 0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().block<2, 2>(0, 0))> == 0);

  static_assert(zero_matrix<decltype(std::declval<Z22>().block<2, 1>(0, 0))>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().block(2, 1, 0, 0))>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().block<1, 1>(0, 0))>);

  static_assert(writable<Eigen::Block<M33, 3, 1, true>>);
  static_assert(writable<Eigen::Block<M33, 3, 2, true>>);

  static_assert(modifiable<Eigen::Block<M33, 3, 1, true>, M31>);
  static_assert(modifiable<Eigen::Block<M33, 3, 1, true>, Eigen::Block<M33, 3, 1, true>>);
  static_assert(not modifiable<Eigen::Block<M33, 3, 2, true>, M31>);
  static_assert(not modifiable<Eigen::Block<M33, 3, 2, true>, Eigen::Block<M33, 3, 1, true>>);
}

// For Eigen::CwiseBinaryOp, see cwise_binary_operations

// For Eigen::CwiseNullaryOp, see cwise_nullary_operations

// For Eigen::CwiseTernaryOp, see cwise_ternary_operations

// For Eigen::CwiseUnaryOp, see cwise_unary_operations

// For Eigen::CwiseUnaryView, see cwise_unary_operations


TEST(eigen3, Eigen_Diagonal)
{
  static_assert(not self_contained<decltype(std::declval<M22>().diagonal())>);
  static_assert(self_contained<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M20::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M02::Identity().diagonal())> == 1);
  static_assert(constant_coefficient_v<decltype(M00::Identity().diagonal())> == 1);

  static_assert(constant_coefficient_v<decltype(M22::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M20::Identity().diagonal<-1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M02::Identity().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(M00::Identity().diagonal<-1>())> == 0);
  static_assert(constant_matrix<decltype(M22::Identity().diagonal<Eigen::DynamicIndex>()), CompileTimeStatus::unknown>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);

  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C10_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C01_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().diagonal())> == 2);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().matrix().diagonal<1>())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd20_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd20_2>().matrix().diagonal<1>())> == 0);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd22_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd20_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd02_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd00_2>().matrix().diagonal())>);
}


TEST(eigen3, Eigen_DiagonalMatrix)
{
  static_assert(row_dimension_of_v<DM2> == 2);
  static_assert(row_dimension_of_v<DM0> == dynamic_size);

  static_assert(column_dimension_of_v<DM2> == 2);
  static_assert(column_dimension_of_v<DM0> == dynamic_size);

  static_assert(self_contained<DM2>);
  static_assert(self_contained<DM0>);

  static_assert(square_matrix<DM0>);

  static_assert(diagonal_matrix<DM2>);
  static_assert(diagonal_matrix<DM0>);

  static_assert(triangular_matrix<DM0, TriangleType::lower>);

  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::DiagonalMatrix<double, 2>>, M21>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, 3>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>, M00>);

  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);

  static_assert(not writable<Eigen::DiagonalMatrix<double, 3>>);

  static_assert(element_gettable<Eigen::DiagonalMatrix<double, 2>, 2>);
  static_assert(element_gettable<Eigen::DiagonalMatrix<double, 2>, 1>);
  static_assert(element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2>);
  static_assert(element_gettable<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 1>);

  static_assert(element_gettable<Eigen::DiagonalWrapper<M21>, 2>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M21>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M20>, 2>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M20>, 1>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M01>, 2>);
  static_assert(element_gettable<Eigen::DiagonalWrapper<M01>, 1>);
}


TEST(eigen3, Eigen_DiagonalWrapper)
{
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M31>> == 3);
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M30>> == dynamic_size);
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M01>> == dynamic_size);
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M00>> == dynamic_size);

  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M13>> == 3);
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M10>> == dynamic_size);
  static_assert(row_dimension_of_v<Eigen::DiagonalWrapper<M03>> == dynamic_size);

  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M31>> == 3);
  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M30>> == dynamic_size);
  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M01>> == dynamic_size);
  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M00>> == dynamic_size);

  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M13>> == 3);
  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M10>> == dynamic_size);
  static_assert(column_dimension_of_v<Eigen::DiagonalWrapper<M03>> == dynamic_size);

  static_assert(square_matrix<Eigen::DiagonalWrapper<M00>>);

  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::DiagonalWrapper<M21>>, const M21&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M31>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M22>>, M44>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M30>>, M00>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M01>>, M00>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M00>>, M00>);

  static_assert(not self_contained<Eigen::DiagonalWrapper<M31>>);

  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().asDiagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z21>().matrix().asDiagonal())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().asDiagonal())> == 2);

  static_assert(not zero_matrix<decltype(std::declval<C11_1>())>);
  static_assert(not zero_matrix<decltype(std::declval<C11_2>())>);
  static_assert(not zero_matrix<decltype(std::declval<C21_2>())>);

  static_assert(zero_matrix<decltype(std::declval<Z11>().matrix().asDiagonal())>);
  static_assert(zero_matrix<decltype(std::declval<Z21>().matrix().asDiagonal())>);

  static_assert(identity_matrix<decltype(std::declval<C11_1>().matrix().asDiagonal())>);
  static_assert(identity_matrix<decltype(std::declval<C21_1>().matrix().asDiagonal())>);

  static_assert(diagonal_matrix<Eigen::DiagonalWrapper<M31>>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_2>().matrix().asDiagonal())>);
  static_assert(diagonal_matrix<decltype(std::declval<C21_2>().matrix().asDiagonal())>);

  static_assert(not writable<Eigen::DiagonalWrapper<M31>>);
}


// No current tests for Eigen::Homogeneous


#if EIGEN_VERSION_AT_LEAST(3,4,0)
// No current tests for Eigen::IndexedView
#endif


// No current tests for Eigen::Inverse


// No current tests for Eigen::Map


TEST(eigen3, Eigen_MatrixWrapper)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix())> == 2);
  static_assert(zero_matrix<decltype(std::declval<Z23>().matrix())>);
  static_assert(identity_matrix<decltype(std::declval<I22>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd22_2>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().array().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().array().matrix())>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().array().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().array().matrix()), TriangleType::upper>);

  static_assert(std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen::MatrixWrapper<M32>>()))>);
  static_assert(not std::is_lvalue_reference_v<decltype(nested_matrix(std::declval<Eigen::MatrixWrapper<I22>>()))>);
}


// No current tests for Eigen::PermutationWrapper.


TEST(eigen3, Eigen_Product)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_m2>().matrix())> == -8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().matrix() * std::declval<C22_2>().matrix())> == 8);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C02_2>().matrix())> == 8);
  static_assert(constant_matrix<decltype(std::declval<C20_2>().matrix() * std::declval<C02_2>().matrix()), CompileTimeStatus::unknown>);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<C22_2>().matrix())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<Cd22_3>().matrix())> == 6);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().matrix() * std::declval<I22>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<I22>().matrix() * std::declval<Cd22_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);

  static_assert(zero_matrix<decltype(std::declval<Z11>().matrix() * std::declval<Z11>().matrix())>);
  static_assert(zero_matrix<decltype(std::declval<Z12>().matrix() * std::declval<Z21>().matrix())>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Md21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Salv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Sauv22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().matrix() * std::declval<Cd22_2>().matrix())>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Cd22_2>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Cd22_2>().matrix()), TriangleType::upper>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().matrix() * std::declval<Tlv22>().matrix()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().matrix() * std::declval<Tuv22>().matrix()), TriangleType::upper>);
}


// No current tests for Eigen::Ref.


TEST(eigen3, Eigen_Replicate)
{
  auto z11 = M11::Identity() - M11::Identity();
  using Z11 = decltype(z11);
  auto z20_1 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 1};
  auto z01_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 1> {z11, 2, 1};
  auto z00_21 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 1};
  auto z22 = Eigen::Replicate<Z11, 2, 2> {z11};
  auto z20_2 = Eigen::Replicate<Z11, 2, Eigen::Dynamic> {z11, 2, 2};
  auto z02_2 = Eigen::Replicate<Z11, Eigen::Dynamic, 2> {z11, 2, 2};
  auto z00_22 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic> {z11, 2, 2};

  static_assert(Eigen3::native_eigen_general<Z00>);
  static_assert(max_indices_of_v<Z00> == 2);
  static_assert(index_dimension_of_v<Z00, 0> == dynamic_size);
  static_assert(index_dimension_of_v<Z00, 1> == dynamic_size);
  EXPECT_EQ(get_index_descriptor<0>(z00_21), 2);
  EXPECT_EQ(get_index_descriptor<1>(z00_21), 1);
  static_assert(std::is_same_v<typename interface::Elements<Z00>::scalar_type, double>);

  static_assert(one_by_one_matrix<Eigen::Replicate<M11, 1, 1>>);
  static_assert(one_by_one_matrix<Eigen::Replicate<M00, 1, 1>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Eigen::Replicate<M10, 1, 1>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Eigen::Replicate<M01, 1, 1>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Replicate<M20, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Replicate<M02, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Replicate<M00, 1, 1>>);
  static_assert(one_by_one_matrix<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Replicate<M11, Eigen::Dynamic, Eigen::Dynamic>>);

  static_assert(square_matrix<Eigen::Replicate<M22, 3, 3>>);
  static_assert(square_matrix<Eigen::Replicate<M22, Eigen::Dynamic, 3>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M22, Eigen::Dynamic, 3>>);
  static_assert(square_matrix<Eigen::Replicate<M22, 3, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M22, 3, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Replicate<M32, 2, 3>>);
  static_assert(square_matrix<Eigen::Replicate<M32, Eigen::Dynamic, 3>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M32, Eigen::Dynamic, 3>>);
  static_assert(square_matrix<Eigen::Replicate<M32, 2, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M32, 2, Eigen::Dynamic>>);
  static_assert(not square_matrix<Eigen::Replicate<M32, 5, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M32, Eigen::Dynamic, 2>, Likelihood::maybe>);
  static_assert(square_matrix<Eigen::Replicate<M30, 2, 3>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M30, 2, 3>>);
  static_assert(square_matrix<Eigen::Replicate<M02, 2, 3>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M02, 2, 3>>);
  static_assert(not square_matrix<Eigen::Replicate<M20, 2, 3>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Replicate<M03, 2, 3>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<Eigen::Replicate<Z11, 1, 2>> == 0);
  static_assert(constant_coefficient_v<decltype(z20_1)> == 0);
  static_assert(constant_coefficient_v<decltype(z01_2)> == 0);
  static_assert(constant_coefficient_v<Eigen::Replicate<C20_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Replicate<C02_2, 1, 2>> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().replicate<5,5>())> == 2);
  static_assert(not constant_matrix<Eigen::Replicate<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().replicate<1,1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().replicate<5,5>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(z22)> == 0);
  static_assert(not constant_diagonal_matrix<decltype(z20_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z02_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z00_22)>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().replicate<1,1>())> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Replicate<C22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::any, Likelihood::maybe>);

  static_assert(identity_matrix<Eigen::Replicate<I22, 1, 1>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().replicate<1, 1>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<2, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<2, Eigen::Dynamic>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, Eigen::Dynamic>())>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().replicate<1, 1>()), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().replicate<1, 1>()), TriangleType::upper>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().replicate<1, 1>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().replicate<1, 1>())>);
}


#if EIGEN_VERSION_AT_LEAST(3,4,0)
TEST(eigen3, Eigen_Reshaped)
{
  static_assert(native_eigen_matrix<Eigen::Reshaped<M32, 2, 3, Eigen::RowMajor>>);
  static_assert(native_eigen_array<Eigen::Reshaped<Eigen::ArrayWrapper<M32>, 2, 3, Eigen::RowMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I22, 1, 2, Eigen::RowMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I22, 1, 2, Eigen::ColMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I22, Eigen::Dynamic, 2, Eigen::RowMajor>>);
  static_assert(self_contained<Eigen::Reshaped<I00, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<M32, 2, 3, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<M32, 2, Eigen::Dynamic, Eigen::RowMajor>>);
  static_assert(not self_contained<Eigen::Reshaped<M00, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>);
  auto m32 = make_eigen_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6);
  auto m23 = make_eigen_matrix<double, 2, 3>(1, 4, 2, 5, 3, 6);
  EXPECT_TRUE(is_near(make_self_contained(m32.reshaped<Eigen::RowMajor>(2, 3)), m23));

  static_assert(index_dimension_of_v<Eigen::Reshaped<M00, 3, Eigen::Dynamic>, 0> == 3);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M00, Eigen::Dynamic, 4>, 1> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, Eigen::Dynamic, 2>, 0> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M21, 1, Eigen::Dynamic>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, Eigen::Dynamic, 1>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reshaped<M12, 2, Eigen::Dynamic>, 1> == 1);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>, 0> == 4);
  static_assert(index_dimension_of_v<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>, 1> == 4);

  static_assert(one_by_one_matrix<Eigen::Reshaped<M00, 1, 1>>);
  static_assert(one_by_one_matrix<Eigen::Reshaped<M00, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M00, 2, 2>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M20, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M02, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M20, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M02, 1, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M20, 1, 1>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Eigen::Reshaped<M02, 1, 1>, Likelihood::maybe>);

  static_assert(square_matrix<Eigen::Reshaped<M00, 2, 2>>);
  static_assert(square_matrix<Eigen::Reshaped<M00, 2, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M00, 2, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Reshaped<M00, Eigen::Dynamic, 2>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M00, Eigen::Dynamic, 2>>);
  static_assert(square_matrix<Eigen::Reshaped<M00, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M00, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Reshaped<M11, 1, 1>>);
  static_assert(square_matrix<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M22, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Reshaped<M20, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M20, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Reshaped<M02, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(square_matrix<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, 4, Eigen::Dynamic>>);
  static_assert(square_matrix<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, 4>>);
  static_assert(square_matrix<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<Eigen::Matrix<double, 2, 8>, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not square_matrix<Eigen::Reshaped<Eigen::Matrix<double, 2, 9>, Eigen::Dynamic, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M02, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(not square_matrix<Eigen::Reshaped<M50, 2, 2>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M05, 2, 2>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M20, 1, 1>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M02, 1, 1>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M20, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M02, 1, Eigen::Dynamic>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M21, Eigen::Dynamic, 1>, Likelihood::maybe>);
  static_assert(not square_matrix<Eigen::Reshaped<M21, 2, Eigen::Dynamic>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<Eigen::Reshaped<C22_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C20_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C02_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C00_2, 2, 2>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C22_2, 4, 1>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C20_2, 1, 4>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C02_2, Eigen::Dynamic, 1>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C00_2, 1, Eigen::Dynamic>> == 2);
  static_assert(constant_coefficient_v<Eigen::Reshaped<C00_2, Eigen::Dynamic, Eigen::Dynamic>> == 2);

  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd22_2, 2, 2>> == 2);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, 2, 2>, CompileTimeStatus::known>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd20_2, 2, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd20_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd20_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd20_2, Eigen::Dynamic, 2>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd02_2, Eigen::Dynamic, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd02_2, Eigen::Dynamic, 2>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd02_2, 2, Eigen::Dynamic>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd02_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reshaped<Cd00_2, 2, 2>> == 2);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd22_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, Eigen::Dynamic, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, Eigen::Dynamic, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, 2, Eigen::Dynamic>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, 2, Eigen::Dynamic>>);
  static_assert(constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, Eigen::Dynamic, 2>, CompileTimeStatus::known, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<Eigen::Reshaped<Cd00_2, Eigen::Dynamic, 2>>);

  static_assert(zero_matrix<Eigen::Reshaped<Z22, 4, 1>>);
  static_assert(zero_matrix<Eigen::Reshaped<Z21, 1, 2>>);
  static_assert(zero_matrix<Eigen::Reshaped<Z23, 3, 2>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tlv22, 2, 2>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv20, 2, Eigen::Dynamic>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv20, Eigen::Dynamic, 2>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv02, Eigen::Dynamic, 2>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv02, 2, Eigen::Dynamic>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tlv00, 2, 2>, TriangleType::lower>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tlv00, Eigen::Dynamic, 2>>);

  static_assert(triangular_matrix<Eigen::Reshaped<Tuv22, 2, 2>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv20, 2, Eigen::Dynamic>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv20, Eigen::Dynamic, 2>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv02, Eigen::Dynamic, 2>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv02, 2, Eigen::Dynamic>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reshaped<Tuv00, 2, 2>, TriangleType::upper>);
  static_assert(not triangular_matrix<Eigen::Reshaped<Tuv00, Eigen::Dynamic, 2>>);

  static_assert(not hermitian_adapter<Eigen::Reshaped<Z22, 2, 2>>);
  static_assert(not hermitian_adapter<Eigen::Reshaped<C22_2, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<Z22, 2, 2>>);
  static_assert(hermitian_matrix<Eigen::Reshaped<C22_2, 2, 2>>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 4, 1>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::Reshaped<C22_2, 1, 4>, Likelihood::maybe>);
}
#endif // EIGEN_VERSION_AT_LEAST(3,4,0)


TEST(eigen3, Eigen_Reverse)
{
  static_assert(index_dimension_of_v<Eigen::Reverse<M20, Eigen::Vertical>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<M02, Eigen::Vertical>, 1> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<M20, Eigen::Horizontal>, 0> == 2);
  static_assert(index_dimension_of_v<Eigen::Reverse<M02, Eigen::BothDirections>, 1> == 2);

  static_assert(one_by_one_matrix<Eigen::Reverse<M11, Eigen::Vertical>>);
  static_assert(one_by_one_matrix<Eigen::Reverse<M10, Eigen::Horizontal>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Eigen::Reverse<M01, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Eigen::Reverse<M00, Eigen::Vertical>, Likelihood::maybe>);

  static_assert(square_matrix<Eigen::Reverse<M22, Eigen::BothDirections>>);
  static_assert(square_matrix<Eigen::Reverse<M20, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(square_matrix<Eigen::Reverse<M02, Eigen::BothDirections>, Likelihood::maybe>);
  static_assert(square_matrix<Eigen::Reverse<M00, Eigen::BothDirections>, Likelihood::maybe>);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().reverse())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C20_2>().reverse())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<Z22, Eigen::Vertical>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<M11::IdentityReturnType, Eigen::Horizontal>> == 1);

  static_assert(zero_matrix<decltype(std::declval<Z23>().reverse())>);

  static_assert(identity_matrix<Eigen::Reverse<I22, Eigen::BothDirections>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().reverse())>);
  static_assert(diagonal_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<C00_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().reverse()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv20>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv02>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv00>().reverse()), TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().reverse()), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv20>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv02>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv00>().reverse()), TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Vertical>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::Reverse<M11, Eigen::Horizontal>, TriangleType::upper>);
}


TEST(eigen3, Eigen_Select)
{
  auto br = make_eigen_matrix<bool, 2, 2>(true, false, true, false);
  auto bsa = eigen_matrix_t<bool, 2, 2>::Identity();

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_coefficient_v<decltype(br.select(std::declval<C22_2>(), std::declval<C22_2>()))> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd22_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<Cd22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(br.select(std::declval<Cd22_2>(), std::declval<Cd22_2>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().select(std::declval<C11_2>(), std::declval<Z22>()))> == 2);

  static_assert(zero_matrix<decltype(std::declval<B22_true>().select(std::declval<Z22>(), M22::Identity()))>);
  static_assert(not zero_matrix<decltype(std::declval<B22_true>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero_matrix<decltype(std::declval<B22_false>().select(std::declval<I22>(), std::declval<Z22>()))>);
  static_assert(zero_matrix<decltype(br.select(std::declval<Z22>(), std::declval<Z22>()))>);

  static_assert(diagonal_matrix<decltype(std::declval<B22_true>().select(std::declval<Md21>(), std::declval<M22>()))>);
  static_assert(diagonal_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Md21>()))>);

  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Salv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Sauv22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Salv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Salv22>(), std::declval<Sauv22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sauv22>(), std::declval<Salv22>()))>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tlv22>(), std::declval<M22>())), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tlv22>())), TriangleType::lower>);

  static_assert(triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tuv22>(), std::declval<M22>())), TriangleType::upper>);
  static_assert(triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tuv22>())), TriangleType::upper>);
}


TEST(eigen3, Eigen_SelfAdjointView)
{
  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>, M22&>);

  static_assert(not native_eigen_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(not writable<Eigen::SelfAdjointView<M33, Eigen::Lower>>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M30, Eigen::Lower>>, M30>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M03, Eigen::Lower>>, M03>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M00, Eigen::Lower>>, M00>);

  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>>);
  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M22, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M02, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M02, Eigen::Upper>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M20, Eigen::Lower>>>);
  static_assert(has_dynamic_dimensions<dense_writable_matrix_t<Eigen::SelfAdjointView<M20, Eigen::Upper>>>);

  static_assert(constant_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C22_2>, Eigen::Upper>> == 2);

  static_assert(constant_matrix<C11_1_complex>);
  static_assert(std::real(constant_coefficient_v<C11_1_complex>) == 1);
  static_assert(std::imag(constant_coefficient_v<C11_1_complex>) == 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_1_complex>, Eigen::Lower>>);

  static_assert(constant_matrix<C11_2_complex>);
  EXPECT_EQ(std::real(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 2);
  EXPECT_EQ(std::imag(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_2_complex>, Eigen::Lower>>);

  static_assert(constant_diagonal_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>> == 2);

  static_assert(zero_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Z22>, Eigen::Upper>>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M30::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M03::Identity(3, 3)), Eigen::Upper>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M00::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);

  static_assert(diagonal_matrix<Sadv22>);
  static_assert(diagonal_matrix<Sadv20, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadv02, Likelihood::maybe>);
  static_assert(diagonal_matrix<Sadv00, Likelihood::maybe>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Lower>, HermitianAdapterType::lower>); // the diagonal must be real

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>, HermitianAdapterType::upper>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Upper>, HermitianAdapterType::upper>); // the diagonal must be real

  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M43, Eigen::Lower>, HermitianAdapterType::lower>);
  static_assert(not hermitian_adapter<Eigen::SelfAdjointView<M34, Eigen::Upper>, HermitianAdapterType::upper>);

  auto m22l = make_eigen_matrix<double, 2, 2>(9, 0, 3, 10);
  auto salv22 = m22l.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(get_element(salv22, 0, 0), 9);
  EXPECT_EQ(get_element(salv22, 0, 1), 3);
  EXPECT_EQ(get_element(salv22, 1, 0), 3);
  EXPECT_EQ(get_element(salv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_element(salv22, 0, 1))>);
  static_assert(element_settable<decltype(salv22), 2>);
  set_element(salv22, 4, 0, 1);
  EXPECT_EQ(get_element(salv22, 0, 1), 4);
  EXPECT_EQ(get_element(salv22, 1, 0), 4);

  auto m22u = make_eigen_matrix<double, 2, 2>(9, 3, 0, 10);
  auto sauv22 = m22u.template selfadjointView<Eigen::Upper>();
  EXPECT_EQ(get_element(sauv22, 0, 0), 9);
  EXPECT_EQ(get_element(sauv22, 0, 1), 3);
  EXPECT_EQ(get_element(sauv22, 1, 0), 3);
  EXPECT_EQ(get_element(sauv22, 1, 1), 10);
  static_assert(std::is_lvalue_reference_v<decltype(get_element(sauv22, 1, 0))>);
  static_assert(element_settable<decltype(sauv22), 2>);
  set_element(sauv22, 4, 1, 0);
  EXPECT_EQ(get_element(sauv22, 0, 1), 4);
  EXPECT_EQ(get_element(sauv22, 1, 0), 4);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{9, 0.9}, 0, std::complex<double>{3, 0.3}, 10);
  auto salv22c = m22lc.template selfadjointView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_element(salv22c, 0, 0)), 9);
  EXPECT_EQ(std::real(get_element(salv22c, 0, 1)), 3);
  EXPECT_EQ(std::imag(get_element(salv22c, 0, 1)), -0.3);
  EXPECT_EQ(std::real(get_element(salv22c, 1, 0)), 3);
  EXPECT_EQ(std::imag(get_element(salv22c, 1, 0)), 0.3);
  EXPECT_EQ(std::real(get_element(salv22c, 1, 1)), 10);
  static_assert(not std::is_lvalue_reference_v<decltype(get_element(salv22c, 0, 1))>);
  static_assert(element_settable<decltype(salv22c), 2>);
  set_element(salv22c, std::complex<double>{4, 0.4}, 0, 1);
  EXPECT_EQ(std::imag(get_element(salv22c, 0, 1)), 0.4);
  EXPECT_EQ(std::imag(get_element(salv22c, 1, 0)), -0.4);
}


TEST(eigen3, Eigen_Solve)
{
  static_assert(not self_contained<Eigen::Solve<Eigen::PartialPivLU<M31>, M31>>);
}


TEST(eigen3, Eigen_Transpose)
{
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().transpose())> == 2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd22_2>().transpose())> == 2);

  static_assert(zero_matrix<decltype((std::declval<Z23>()).transpose())>);

  static_assert(identity_matrix<Eigen::Transpose<I22>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sauv22>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Salv22>().transpose())>);
  static_assert(triangular_matrix<decltype(std::declval<Tuv22>().transpose()), TriangleType::lower>);
  static_assert(triangular_matrix<decltype(std::declval<Tlv22>().transpose()), TriangleType::upper>);
}


TEST(eigen3, Eigen_TriangularView)
{
  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::TriangularView<M22, Eigen::Upper>>, M22&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M33, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M30, Eigen::Upper>>, M30>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M03, Eigen::Upper>>, M03>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M00, Eigen::Upper>>, M00>);

  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<C11_2>, Eigen::Lower>> == 2);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::Lower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z00>, Eigen::Upper>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<decltype(M32::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd20_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd02_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd00_2>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z20>, Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z23>, Eigen::StrictlyLower>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z02>, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M20::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M02::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M00::Identity()), Eigen::StrictlyLower>> == 0);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::StrictlyLower>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M20::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M02::Identity()), Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M00::Identity()), Eigen::UnitLower>> == 1);
  static_assert(not constant_diagonal_matrix<Eigen::TriangularView<decltype(M23::Identity()), Eigen::UnitUpper>, CompileTimeStatus::any, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tlv22, Eigen::UnitUpper>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Tuv22, Eigen::UnitLower>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::StrictlyLower>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::UnitLower>> == 1);

  static_assert(zero_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>);

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M30::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M03::Identity(3, 3)), Eigen::Upper>, Likelihood::maybe>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M00::Identity(3, 3)), Eigen::Lower>, Likelihood::maybe>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::StrictlyUpper>>);
  static_assert(identity_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::UnitUpper>>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>, TriangleType::lower>);
  static_assert(triangular_matrix<Eigen::TriangularView<M30, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<M03, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<M00, Eigen::Lower>, TriangleType::lower, Likelihood::maybe>);
  static_assert(triangular_matrix<Tlv02, TriangleType::lower, Likelihood::maybe>);
  static_assert(not triangular_matrix<Eigen::TriangularView<M43, Eigen::Lower>, TriangleType::lower>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>, TriangleType::upper>);
  static_assert(triangular_matrix<Eigen::TriangularView<M30, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<M03, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Eigen::TriangularView<M00, Eigen::Upper>, TriangleType::upper, Likelihood::maybe>);
  static_assert(triangular_matrix<Tuv20, TriangleType::upper, Likelihood::maybe>);
  static_assert(not triangular_matrix<Eigen::TriangularView<M34, Eigen::Upper>, TriangleType::upper>);

  static_assert(triangular_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>, TriangleType::diagonal>);
  static_assert(triangular_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>, TriangleType::diagonal>);

  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv22, Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv20, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv02, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tlv00, Eigen::Upper>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv22, Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv20, Eigen::Lower>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv02, Eigen::Lower>, Likelihood::maybe>);
  static_assert(diagonal_matrix<Eigen::TriangularView<Tuv00, Eigen::Lower>, Likelihood::maybe>);

  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Cd22_2>, Eigen::Upper>>);

  auto m22l = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tlv22 = m22l.template triangularView<Eigen::Lower>();
  EXPECT_EQ(get_element(tlv22, 0, 0), 3);
  EXPECT_EQ(get_element(tlv22, 0, 1), 0);
  EXPECT_EQ(get_element(tlv22, 1, 0), 1);
  EXPECT_EQ(get_element(tlv22, 1, 1), 3);
  static_assert(element_settable<decltype(tlv22), 2>);
  set_element(tlv22, 4, 1, 0);
  EXPECT_EQ(get_element(tlv22, 1, 0), 4);
  EXPECT_EQ(get_element(tlv22, 0, 1), 0);

  auto m22u = make_eigen_matrix<double, 2, 2>(3, 1, 1, 3);
  auto tuv22 = m22u.template triangularView<Eigen::Upper>();
  EXPECT_EQ(get_element(tuv22, 0, 0), 3);
  EXPECT_EQ(get_element(tuv22, 0, 1), 1);
  EXPECT_EQ(get_element(tuv22, 1, 0), 0);
  EXPECT_EQ(get_element(tuv22, 1, 1), 3);
  static_assert(element_settable<decltype(tuv22), 2>);
  set_element(tuv22, 4, 0, 1);
  EXPECT_EQ(get_element(tuv22, 0, 1), 4);
  EXPECT_EQ(get_element(tuv22, 1, 0), 0);

  auto m22lc = make_eigen_matrix<std::complex<double>, 2, 2>(std::complex<double>{3, 0.3}, 0, std::complex<double>{1, 0.1}, 3);
  auto tlv22c = m22lc.template triangularView<Eigen::Lower>();
  EXPECT_EQ(std::real(get_element(tlv22c, 0, 0)), 3);
  EXPECT_EQ(std::real(get_element(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::imag(get_element(tlv22c, 0, 1)), 0);
  EXPECT_EQ(std::real(get_element(tlv22c, 1, 0)), 1);
  EXPECT_EQ(std::imag(get_element(tlv22c, 1, 0)), 0.1);
  EXPECT_EQ(std::real(get_element(tlv22c, 1, 1)), 3);
  static_assert(element_settable<decltype(tlv22c), 2>);
  set_element(tlv22c, std::complex<double>{4, 0.4}, 1, 0);
  EXPECT_EQ(std::imag(get_element(tlv22c, 1, 0)), 0.4);
  EXPECT_EQ(std::imag(get_element(tlv22c, 0, 1)), 0);
}


TEST(eigen3, Eigen_VectorBlock)
{
  static_assert(native_eigen_matrix<Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 1>>);
  static_assert(native_eigen_array<Eigen::VectorBlock<Eigen::Array<double, 2, 1>, 1>>);
  static_assert(std::is_same_v<double, typename Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 0>::Scalar>);

  static_assert(native_eigen_general<decltype(std::declval<C21_2>().segment<1>(0))>);

  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().segment<1>(0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_m2>().segment(1, 0))> == -2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().segment<1>(0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_m2>().segment<1>(0))> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z21>().segment<1>(0))>);
  static_assert(zero_matrix<decltype(std::declval<Z21>().segment(1, 0))>);

}
