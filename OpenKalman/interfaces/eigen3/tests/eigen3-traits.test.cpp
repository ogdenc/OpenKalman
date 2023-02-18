/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
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

  using A11 = Eigen::Array<double, 1, 1>;
  using A10 = Eigen::Array<double, 1, Eigen::Dynamic>;
  using A01 = Eigen::Array<double, Eigen::Dynamic, 1>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using A22 = Eigen::Array<double, 2, 2>;
  using A20 = Eigen::Array<double, 2, Eigen::Dynamic>;
  using A02 = Eigen::Array<double, Eigen::Dynamic, 2>;
  using A00 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

  using CA22 = Eigen::Array<cdouble, 2, 2>;

  using I11 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A11>;
  using I10 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A10>;
  using I01 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A01>;
  using I00 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A00>;

  using I21 = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<double>, A22>;
  using I20_1 = Eigen::DiagonalWrapper<Eigen::Replicate<I11, 2, Eigen::Dynamic>>;
  using I01_2 = Eigen::DiagonalWrapper<Eigen::Replicate<I11, Eigen::Dynamic, 1>>;
  using I00_21 = Eigen::DiagonalWrapper<Eigen::Replicate<I11, Eigen::Dynamic, Eigen::Dynamic>>;

  using Z11 = decltype(std::declval<I11>() - std::declval<I11>());
  using Z22 = decltype(std::declval<I21>() - std::declval<I21>());
  using Z21 = Eigen::Replicate<Z11, 2, 1>;;
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

  using D21_2 = Eigen::DiagonalWrapper<C21_2>;
  using D20_1_2 = Eigen::DiagonalWrapper<C20_2>;
  using D01_2_2 = Eigen::DiagonalWrapper<C01_2>;
  using D00_21_2 = Eigen::DiagonalWrapper<C00_2>;

  using D21_3 = Eigen::DiagonalWrapper<decltype(C21_3 {std::declval<I11>() + std::declval<I11>() + std::declval<I11>(), 2, 1})>;
  using D20_1_3 = Eigen::DiagonalWrapper<decltype(C20_3 {std::declval<I11>() + std::declval<I11>() + std::declval<I11>(), 2, 1})>;
  using D01_2_3 = Eigen::DiagonalWrapper<decltype(C01_3 {std::declval<I11>() + std::declval<I11>() + std::declval<I11>(), 2, 1})>;
  using D00_21_3 = Eigen::DiagonalWrapper<decltype(C00_3 {std::declval<I11>() + std::declval<I11>() + std::declval<I11>(), 2, 1})>;

  using DM2 = Eigen::DiagonalMatrix<double, 2>;
  using DM0 = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

  using Cd21_1 = Eigen::ArrayWrapper<DiagonalMatrix<C21_1>>;

  using Cd21_2 = Eigen::ArrayWrapper<DiagonalMatrix<C21_2>>;
  using Cd20_1_2 = Eigen::ArrayWrapper<DiagonalMatrix<C20_2>>;
  using Cd01_2_2 = Eigen::ArrayWrapper<DiagonalMatrix<C01_2>>;
  using Cd00_21_2 = Eigen::ArrayWrapper<DiagonalMatrix<C00_2>>;

  using Cd21_3 = Eigen::ArrayWrapper<DiagonalMatrix<C21_3>>;
  using Cd20_1_3 = Eigen::ArrayWrapper<DiagonalMatrix<C20_3>>;
  using Cd01_2_3 = Eigen::ArrayWrapper<DiagonalMatrix<C01_3>>;
  using Cd00_21_3 = Eigen::ArrayWrapper<DiagonalMatrix<C00_3>>;

  using Cd21_m1 = Eigen::ArrayWrapper<DiagonalMatrix<C21_m1>>;
  using Cd20_1_m1 = Eigen::ArrayWrapper<DiagonalMatrix<C20_m1>>;
  using Cd01_2_m1 = Eigen::ArrayWrapper<DiagonalMatrix<C01_m1>>;
  using Cd00_21_m1 = Eigen::ArrayWrapper<DiagonalMatrix<C00_m1>>;

  using Cd21_m2 = Eigen::ArrayWrapper<DiagonalMatrix<C21_m2>>;
  using Cd20_1_m2 = Eigen::ArrayWrapper<DiagonalMatrix<C20_m2>>;
  using Cd01_2_m2 = Eigen::ArrayWrapper<DiagonalMatrix<C01_m2>>;
  using Cd00_21_m2 = Eigen::ArrayWrapper<DiagonalMatrix<C00_m2>>;

  using Md21 = Eigen::ArrayWrapper<DiagonalMatrix<M21>>;
  using Md20_1 = Eigen::ArrayWrapper<DiagonalMatrix<M20>>;
  using Md01_2 = Eigen::ArrayWrapper<DiagonalMatrix<M01>>;
  using Md00_21 = Eigen::ArrayWrapper<DiagonalMatrix<M00>>;

  using Salv22 = Eigen::SelfAdjointView<M22, Eigen::Lower>;
  using Salv20_2 = Eigen::SelfAdjointView<M20, Eigen::Lower>;
  using Salv02_2 = Eigen::SelfAdjointView<M02, Eigen::Lower>;
  using Salv00_22 = Eigen::SelfAdjointView<M00, Eigen::Lower>;

  using Sauv22 = Eigen::SelfAdjointView<M22, Eigen::Upper>;
  using Sauv20_2 = Eigen::SelfAdjointView<M20, Eigen::Upper>;
  using Sauv02_2 = Eigen::SelfAdjointView<M02, Eigen::Upper>;
  using Sauv00_22 = Eigen::SelfAdjointView<M00, Eigen::Upper>;

  using Sadv22 = Eigen::SelfAdjointView<M22::IdentityReturnType, Eigen::Lower>;

  using Sal22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M22, TriangleType::lower>>;
  using Sal20_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M20, TriangleType::lower>>;
  using Sal02_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M02, TriangleType::lower>>;
  using Sal00_22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M00, TriangleType::lower>>;

  using Sau22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M22, TriangleType::upper>>;
  using Sau20_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M20, TriangleType::upper>>;
  using Sau02_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M02, TriangleType::upper>>;
  using Sau00_22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M00, TriangleType::upper>>;

  using Sad22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M22, TriangleType::diagonal>>;
  using Sad20_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M20, TriangleType::diagonal>>;
  using Sad02_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M02, TriangleType::diagonal>>;
  using Sad00_22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M00, TriangleType::diagonal>>;

  using Tl22 = Eigen::ArrayWrapper<TriangularMatrix<M22, TriangleType::lower>>;
  using Tl20_2 = Eigen::ArrayWrapper<TriangularMatrix<M20, TriangleType::lower>>;
  using Tl02_2 = Eigen::ArrayWrapper<TriangularMatrix<M02, TriangleType::lower>>;
  using Tl00_22 = Eigen::ArrayWrapper<TriangularMatrix<M00, TriangleType::lower>>;

  using Tu22 = Eigen::ArrayWrapper<TriangularMatrix<M22, TriangleType::upper>>;
  using Tu20_2 = Eigen::ArrayWrapper<TriangularMatrix<M20, TriangleType::upper>>;
  using Tu02_2 = Eigen::ArrayWrapper<TriangularMatrix<M02, TriangleType::upper>>;
  using Tu00_22 = Eigen::ArrayWrapper<TriangularMatrix<M00, TriangleType::upper>>;
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

  static_assert(max_tensor_order_of_v<M23> == 2);
  static_assert(max_tensor_order_of_v<M21> == 1);
  static_assert(max_tensor_order_of_v<M11> == 0);
  static_assert(max_tensor_order_of_v<M20> == 2);
  static_assert(max_tensor_order_of_v<M02> == 2);
  static_assert(max_tensor_order_of_v<M10> == 1);
  static_assert(max_tensor_order_of_v<M01> == 1);
  static_assert(max_tensor_order_of_v<M00> == 2);

  static_assert(interface::IndexTraits<M11, 0>::dimension == 1);
  static_assert(interface::IndexTraits<M21, 0>::dimension == 2);
  static_assert(interface::IndexTraits<M00, 0>::dimension == dynamic_size);
  static_assert(interface::IndexTraits<M11, 1>::dimension == 1);
  static_assert(interface::IndexTraits<M21, 1>::dimension == 1);
  static_assert(interface::IndexTraits<M00, 1>::dimension == dynamic_size);
  EXPECT_EQ((interface::IndexTraits<M11, 0>::dimension_at_runtime(M11{})), 1);
  EXPECT_EQ((interface::IndexTraits<M21, 0>::dimension_at_runtime(M21{})), 2);
  EXPECT_EQ((interface::IndexTraits<M00, 0>::dimension_at_runtime(M00{2, 1})), 2);
  EXPECT_EQ((interface::IndexTraits<M11, 1>::dimension_at_runtime(M11{})), 1);
  EXPECT_EQ((interface::IndexTraits<M21, 1>::dimension_at_runtime(M21{})), 1);
  EXPECT_EQ((interface::IndexTraits<M00, 1>::dimension_at_runtime(M00{2, 1})), 1);
  static_assert(std::is_same_v<typename interface::IndexibleObjectTraits<M00>::scalar_type, double>);

  static_assert(dynamic_rows<eigen_matrix_t<double, dynamic_size, dynamic_size>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, dynamic_size, dynamic_size>>);
  static_assert(dynamic_rows<eigen_matrix_t<double, dynamic_size, 1>>);
  static_assert(not dynamic_columns<eigen_matrix_t<double, dynamic_size, 1>>);
  static_assert(not dynamic_rows<eigen_matrix_t<double, 1, dynamic_size>>);
  static_assert(dynamic_columns<eigen_matrix_t<double, 1, dynamic_size>>);

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

  static_assert(element_gettable<M32, std::size_t, std::size_t>);
  static_assert(element_gettable<const M32, std::size_t, std::size_t>);
  static_assert(element_gettable<M31, std::size_t, std::size_t>);
  static_assert(element_gettable<M13, std::size_t, std::size_t>);
  static_assert(element_gettable<M30, std::size_t, std::size_t>);
  static_assert(element_gettable<M02, std::size_t, std::size_t>);
  static_assert(element_gettable<M01, std::size_t, std::size_t>);
  static_assert(element_gettable<M10, std::size_t, std::size_t>);
  static_assert(element_gettable<M00, std::size_t, std::size_t>);

  static_assert(element_gettable<M32, std::size_t>);
  static_assert(element_gettable<M31, std::size_t>);
  static_assert(element_gettable<M13, std::size_t>);
  static_assert(element_gettable<M02, std::size_t>);
  static_assert(element_gettable<M01, std::size_t>);
  static_assert(element_gettable<M10, std::size_t>);
  static_assert(element_gettable<M00, std::size_t>);

  static_assert(element_settable<M32, std::size_t, std::size_t>);
  static_assert(not element_settable<const M32&, std::size_t, std::size_t>);
  static_assert(element_settable<M31&, std::size_t, std::size_t>);
  static_assert(element_settable<M13&, std::size_t, std::size_t>);
  static_assert(element_settable<M30&, std::size_t, std::size_t>);
  static_assert(element_settable<M02&, std::size_t, std::size_t>);
  static_assert(element_settable<M01&, std::size_t, std::size_t>);
  static_assert(element_settable<M10&, std::size_t, std::size_t>);
  static_assert(element_settable<M00&, std::size_t, std::size_t>);

  static_assert(element_settable<M32&, std::size_t>);
  static_assert(element_settable<M31&, std::size_t>);
  static_assert(element_settable<M13&, std::size_t>);
  static_assert(not element_settable<const M31&, std::size_t>);
  static_assert(element_settable<M02&, std::size_t>);
  static_assert(element_settable<M01&, std::size_t>);
  static_assert(element_settable<M10&, std::size_t>);
  static_assert(element_settable<M00&, std::size_t>);

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

  static_assert(std::is_same_v<coefficient_types_of<M11, 0>::type, Dimensions<1>>);
  static_assert(std::is_same_v<coefficient_types_of<M11, 1>::type, Dimensions<1>>);
  static_assert(equivalent_to<coefficient_types_of_t<M11, 0>, Axis>);
  static_assert(equivalent_to<coefficient_types_of_t<M11, 1>, Axis>);
  static_assert(std::is_same_v<coefficient_types_of<M22, 0>::type, Dimensions<2>>);
  static_assert(std::is_same_v<coefficient_types_of<M22, 1>::type, Dimensions<2>>);
  static_assert(equivalent_to<coefficient_types_of_t<M22, 0>, TypedIndex<Axis, Axis>>);
  static_assert(equivalent_to<coefficient_types_of_t<M22, 1>, TypedIndex<Axis, Axis>>);

  static_assert(maybe_index_descriptors_match<M22, M20, M02, M00>);
  static_assert(index_descriptors_match<M22, CM22, M22>);
  EXPECT_TRUE(get_index_descriptors_match(m22, cm22, M20{m22}, M02{m22}, M00{m22}));

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

  static_assert(one_by_one_matrix<M11>);
  static_assert(not one_by_one_matrix<M10>);
  static_assert(one_by_one_matrix<M10, Likelihood::maybe>);

  static_assert(column_vector<M31>);
  static_assert(column_vector<M01>);
  static_assert(column_vector<M30, Likelihood::maybe>);
  static_assert(column_vector<M00, Likelihood::maybe>);
  static_assert(not column_vector<M32, Likelihood::maybe>);
  static_assert(not column_vector<M02, Likelihood::maybe>);

  static_assert(row_vector<M13>);
  static_assert(row_vector<M10>);
  static_assert(row_vector<M03, Likelihood::maybe>);
  static_assert(row_vector<M00, Likelihood::maybe>);
  static_assert(not row_vector<M23, Likelihood::maybe>);
  static_assert(not row_vector<M20, Likelihood::maybe>);

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

  static_assert(constant_matrix<Z21, Likelihood::maybe, CompileTimeStatus::known>);
  static_assert(scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::known>);
  static_assert(not scalar_constant<constant_coefficient<Z21>, CompileTimeStatus::unknown>);
  static_assert(scalar_constant<int, CompileTimeStatus::unknown>);
  struct RuntimeScalarConstant { constexpr double operator()() const noexcept { return 1; } };
  static_assert(scalar_constant<RuntimeScalarConstant, CompileTimeStatus::unknown>);

  static_assert(are_within_tolerance(constant_coefficient_v<Z21>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z12>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z23>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z20>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z02>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z00>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z01>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<C11_1>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<C11_m1>, -1));
  static_assert(are_within_tolerance(constant_coefficient_v<C11_2>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<C11_m2>, -2));
  static_assert(are_within_tolerance(constant_coefficient_v<C11_3>, 3));
  static_assert(are_within_tolerance(constant_coefficient_v<C20_2>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<C02_2>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<C00_2>, 2));
  static_assert(constant_coefficient_v<B22_true> == true);
  static_assert(constant_coefficient_v<B22_false> == false);
  static_assert(not constant_matrix<Cd21_2>);
  static_assert(not constant_matrix<Cd20_1_2>);
  static_assert(not constant_matrix<Cd01_2_2>);
  static_assert(not constant_matrix<Cd21_3>);
  static_assert(not constant_matrix<Cd21_2, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd20_1_2, Likelihood::maybe>);
  static_assert(constant_matrix<Cd01_2_2, Likelihood::maybe>);
  static_assert(not constant_matrix<Cd21_3, Likelihood::maybe>);

  static_assert(constant_diagonal_matrix<Z11>);
  static_assert(not constant_diagonal_matrix<Z21>);
  static_assert(not constant_diagonal_matrix<Z12>);
  static_assert(not constant_diagonal_matrix<Z23>);
  static_assert(not constant_diagonal_matrix<Z20>);
  static_assert(not constant_diagonal_matrix<Z02>);
  static_assert(not constant_diagonal_matrix<Z00>);
  static_assert(not constant_diagonal_matrix<Z01>);
  static_assert(not constant_diagonal_matrix<Z21, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z10, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z01, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<Z00, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<C11_m1> == -1);
  static_assert(constant_diagonal_coefficient_v<C11_2> == 2);
  static_assert(constant_diagonal_coefficient_v<C11_m2> == -2);
  static_assert(constant_diagonal_matrix<C11_1>);
  static_assert(not constant_diagonal_matrix<C21_1>);
  static_assert(not constant_diagonal_matrix<C20_1>);
  static_assert(not constant_diagonal_matrix<C01_1>);
  static_assert(not constant_diagonal_matrix<C00_1>);
  static_assert(constant_diagonal_matrix<C11_1, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C10_1, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C01_1, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<C00_1, Likelihood::maybe>);
  static_assert(constant_diagonal_coefficient_v<I21> == 1);
  static_assert(constant_diagonal_coefficient_v<I20_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I01_2> == 1);
  static_assert(constant_diagonal_coefficient_v<I00_21> == 1);
  static_assert(constant_diagonal_coefficient_v<Cd21_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd20_1_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd01_2_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd00_21_2> == 2);

  static_assert(zero_matrix<Z21>);
  static_assert(zero_matrix<Eigen::DiagonalWrapper<Z21>>);
  static_assert(zero_matrix<Z23>);
  static_assert(zero_matrix<Z20>);
  static_assert(zero_matrix<Z02>);
  static_assert(zero_matrix<B22_false>);
  static_assert(not zero_matrix<Cd21_2>);
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z00>);

  static_assert(not identity_matrix<C21_1>);
  static_assert(identity_matrix<I21>);
  static_assert(identity_matrix<I20_1>);
  static_assert(identity_matrix<I01_2>);
  static_assert(identity_matrix<I00_21>);
  static_assert(not identity_matrix<Cd21_2>);
  static_assert(not identity_matrix<Cd21_3>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<C10_1>);
  static_assert(not identity_matrix<C01_1>);
  static_assert(not identity_matrix<C00_1>);
  static_assert(not identity_matrix<C21_1, Likelihood::maybe>);
  static_assert(not identity_matrix<C20_1, Likelihood::maybe>);
  static_assert(identity_matrix<C10_1, Likelihood::maybe>);
  static_assert(identity_matrix<C01_1, Likelihood::maybe>);
  static_assert(identity_matrix<C00_1, Likelihood::maybe>);

  static_assert(diagonal_matrix<Z22>);
  static_assert(not diagonal_matrix<Z20>);
  static_assert(not diagonal_matrix<Z02>);
  static_assert(not diagonal_matrix<Z00>);
  static_assert(diagonal_matrix<Z22, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z20, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z02, Likelihood::maybe>);
  static_assert(diagonal_matrix<Z00, Likelihood::maybe>);
  static_assert(diagonal_matrix<C11_2>);
  static_assert(diagonal_matrix<I21>);
  static_assert(diagonal_matrix<I20_1>);
  static_assert(diagonal_matrix<I01_2>);
  static_assert(diagonal_matrix<I00_21>);
  static_assert(diagonal_matrix<Cd21_2>);
  static_assert(diagonal_matrix<Cd20_1_2>);
  static_assert(diagonal_matrix<Cd01_2_2>);
  static_assert(diagonal_matrix<Cd00_21_2>);
  static_assert(diagonal_matrix<Md21>);
  static_assert(diagonal_matrix<Md20_1>);
  static_assert(diagonal_matrix<Md01_2>);
  static_assert(diagonal_matrix<Md00_21>);
  static_assert(not diagonal_matrix<Sal22>);
  static_assert(not diagonal_matrix<Sau22>);
  static_assert(diagonal_matrix<Sad22>);
  static_assert(diagonal_matrix<Sad20_2>);
  static_assert(diagonal_matrix<Sad02_2>);
  static_assert(diagonal_matrix<Sad00_22>);
  static_assert(diagonal_matrix<M11>);
  static_assert(not diagonal_matrix<M10>);
  static_assert(not diagonal_matrix<M01>);
  static_assert(not diagonal_matrix<M00>);
  static_assert(diagonal_matrix<M11, Likelihood::maybe>);
  static_assert(diagonal_matrix<M10, Likelihood::maybe>);
  static_assert(diagonal_matrix<M01, Likelihood::maybe>);
  static_assert(diagonal_matrix<M00, Likelihood::maybe>);

  static_assert(not diagonal_adapter<M11, Likelihood::maybe>);
  static_assert(not diagonal_adapter<M10, Likelihood::maybe>);
  static_assert(not diagonal_adapter<M01, Likelihood::maybe>);
  static_assert(not diagonal_adapter<M00, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M20>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M01>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M10>>);
  static_assert(not diagonal_adapter<Eigen::DiagonalWrapper<M00>>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M21>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M20>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M01>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M10>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M00>, Likelihood::maybe>);
  static_assert(diagonal_adapter<Eigen::DiagonalWrapper<M22>, Likelihood::maybe>);

  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<Z20, Likelihood::maybe>);
  static_assert(hermitian_matrix<Z02, Likelihood::maybe>);
  static_assert(hermitian_matrix<Z00, Likelihood::maybe>);
  static_assert(not hermitian_adapter<Z22>);
  static_assert(not hermitian_adapter<Z20>);
  static_assert(not hermitian_adapter<Z02>);
  static_assert(not hermitian_adapter<Z00>);
  static_assert(hermitian_matrix<C22_2>);
  static_assert(hermitian_matrix<I21>);
  static_assert(hermitian_matrix<I20_1>);
  static_assert(hermitian_matrix<I01_2>);
  static_assert(hermitian_matrix<I00_21>);
  static_assert(hermitian_matrix<Cd21_2>);
  static_assert(hermitian_matrix<Cd20_1_2>);
  static_assert(hermitian_matrix<Cd01_2_2>);
  static_assert(hermitian_matrix<Cd00_21_2>);
  static_assert(hermitian_matrix<Md21>);
  static_assert(hermitian_matrix<Md20_1>);
  static_assert(hermitian_matrix<Md01_2>);
  static_assert(hermitian_matrix<Md00_21>);
  static_assert(not hermitian_adapter<C22_2>);
  static_assert(not hermitian_adapter<I21>);
  static_assert(not hermitian_adapter<I20_1>);
  static_assert(not hermitian_adapter<I01_2>);
  static_assert(not hermitian_adapter<I00_21>);
  static_assert(not hermitian_adapter<Cd21_2>);
  static_assert(not hermitian_adapter<Cd20_1_2>);
  static_assert(not hermitian_adapter<Cd01_2_2>);
  static_assert(not hermitian_adapter<Cd00_21_2>);
  static_assert(not hermitian_adapter<Md21>);
  static_assert(not hermitian_adapter<Md20_1>);
  static_assert(not hermitian_adapter<Md01_2>);
  static_assert(not hermitian_adapter<Md00_21>);
  static_assert(hermitian_matrix<Sal22>);
  static_assert(hermitian_matrix<Sal20_2>);
  static_assert(hermitian_matrix<Sal02_2>);
  static_assert(hermitian_matrix<Sal00_22>);

  static_assert(lower_hermitian_adapter<Salv22>);
  static_assert(lower_hermitian_adapter<Salv20_2>);
  static_assert(lower_hermitian_adapter<Salv02_2>);
  static_assert(lower_hermitian_adapter<Salv00_22>);
  static_assert(not upper_hermitian_adapter<Salv22>);

  static_assert(upper_hermitian_adapter<Sauv22>);
  static_assert(upper_hermitian_adapter<Sauv20_2>);
  static_assert(upper_hermitian_adapter<Sauv02_2>);
  static_assert(upper_hermitian_adapter<Sauv00_22>);
  static_assert(not lower_hermitian_adapter<Sauv22>);

  static_assert(diagonal_hermitian_adapter<Sadv22>);
  static_assert(not upper_hermitian_adapter<Sadv22>);
  static_assert(not lower_hermitian_adapter<Sadv22>);

  static_assert(hermitian_adapter_type_of_v<Salv22> == TriangleType::lower);
  static_assert(hermitian_adapter_type_of_v<Salv22, Salv22> == TriangleType::lower);
  static_assert(hermitian_adapter_type_of_v<Sauv22> == TriangleType::upper);
  static_assert(hermitian_adapter_type_of_v<Sauv22, Sauv22> == TriangleType::upper);
  static_assert(hermitian_adapter_type_of_v<Sauv22, Salv22> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<Salv22, Sauv22> == TriangleType::none);

  static_assert(lower_triangular_matrix<Z22>);
  static_assert(not lower_triangular_matrix<Z20>);
  static_assert(not lower_triangular_matrix<Z02>);
  static_assert(not lower_triangular_matrix<Z00>);
  static_assert(lower_triangular_matrix<Z20, Likelihood::maybe>);
  static_assert(lower_triangular_matrix<Z02, Likelihood::maybe>);
  static_assert(lower_triangular_matrix<Z00, Likelihood::maybe>);
  static_assert(lower_triangular_matrix<C11_2>);
  static_assert(not lower_triangular_matrix<C22_2>);
  static_assert(not lower_triangular_matrix<C22_2, Likelihood::maybe>);
  static_assert(lower_triangular_matrix<I21>);
  static_assert(lower_triangular_matrix<I20_1>);
  static_assert(lower_triangular_matrix<I01_2>);
  static_assert(lower_triangular_matrix<I00_21>);
  static_assert(lower_triangular_matrix<Cd21_2>);
  static_assert(lower_triangular_matrix<Cd20_1_2>);
  static_assert(lower_triangular_matrix<Cd01_2_2>);
  static_assert(lower_triangular_matrix<Cd00_21_2>);
  static_assert(lower_triangular_matrix<Md21>);
  static_assert(lower_triangular_matrix<Md20_1>);
  static_assert(lower_triangular_matrix<Md01_2>);
  static_assert(lower_triangular_matrix<Md00_21>);
  static_assert(lower_triangular_matrix<Tl22>);
  static_assert(lower_triangular_matrix<Tl20_2>);
  static_assert(lower_triangular_matrix<Tl02_2>);
  static_assert(lower_triangular_matrix<Tl00_22>);
  static_assert(not lower_triangular_matrix<Tu22>);
  static_assert(not lower_triangular_matrix<Tu22, Likelihood::maybe>);

  static_assert(upper_triangular_matrix<Z22>);
  static_assert(not upper_triangular_matrix<Z20>);
  static_assert(not upper_triangular_matrix<Z02>);
  static_assert(not upper_triangular_matrix<Z00>);
  static_assert(upper_triangular_matrix<Z20, Likelihood::maybe>);
  static_assert(upper_triangular_matrix<Z02, Likelihood::maybe>);
  static_assert(upper_triangular_matrix<Z00, Likelihood::maybe>);
  static_assert(upper_triangular_matrix<C11_2>);
  static_assert(not upper_triangular_matrix<C22_2>);
  static_assert(not upper_triangular_matrix<C22_2, Likelihood::maybe>);

  static_assert(upper_triangular_matrix<I21>);
  static_assert(upper_triangular_matrix<I20_1>);
  static_assert(upper_triangular_matrix<I01_2>);
  static_assert(upper_triangular_matrix<I00_21>);
  static_assert(upper_triangular_matrix<Cd21_2>);
  static_assert(upper_triangular_matrix<Cd20_1_2>);
  static_assert(upper_triangular_matrix<Cd01_2_2>);
  static_assert(upper_triangular_matrix<Cd00_21_2>);
  static_assert(upper_triangular_matrix<Md21>);
  static_assert(upper_triangular_matrix<Md20_1>);
  static_assert(upper_triangular_matrix<Md01_2>);
  static_assert(upper_triangular_matrix<Md00_21>);
  static_assert(upper_triangular_matrix<Tu22>);
  static_assert(upper_triangular_matrix<Tu20_2>);
  static_assert(upper_triangular_matrix<Tu02_2>);
  static_assert(upper_triangular_matrix<Tu00_22>);
  static_assert(not upper_triangular_matrix<Tl22>);
  static_assert(not upper_triangular_matrix<Tl22, Likelihood::maybe>);

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
  static_assert(square_matrix<Sal22, Likelihood::maybe>);
  static_assert(square_matrix<Sal20_2, Likelihood::maybe>);
  static_assert(square_matrix<Sal02_2, Likelihood::maybe>);
  static_assert(square_matrix<Sal00_22, Likelihood::maybe>);
  static_assert(square_matrix<Sau22, Likelihood::maybe>);
  static_assert(square_matrix<Sau20_2, Likelihood::maybe>);
  static_assert(square_matrix<Sau02_2, Likelihood::maybe>);
  static_assert(square_matrix<Sau00_22, Likelihood::maybe>);
  static_assert(square_matrix<Tl22, Likelihood::maybe>);
  static_assert(square_matrix<Tl20_2, Likelihood::maybe>);
  static_assert(square_matrix<Tl02_2, Likelihood::maybe>);
  static_assert(square_matrix<Tl00_22, Likelihood::maybe>);
  static_assert(square_matrix<Tu22, Likelihood::maybe>);
  static_assert(square_matrix<Tu20_2, Likelihood::maybe>);
  static_assert(square_matrix<Tu02_2, Likelihood::maybe>);
  static_assert(square_matrix<Tu00_22, Likelihood::maybe>);
  static_assert(square_matrix<DiagonalMatrix<M11>, Likelihood::maybe>);
  static_assert(square_matrix<DiagonalMatrix<M10>, Likelihood::maybe>);
  static_assert(square_matrix<DiagonalMatrix<M01>, Likelihood::maybe>);
  static_assert(square_matrix<DiagonalMatrix<M00>, Likelihood::maybe>);
  static_assert(square_matrix<SelfAdjointMatrix<M11, TriangleType::diagonal>, Likelihood::maybe>);
  static_assert(square_matrix<SelfAdjointMatrix<M10, TriangleType::lower>, Likelihood::maybe>);
  static_assert(square_matrix<SelfAdjointMatrix<M01, TriangleType::upper>, Likelihood::maybe>);
  static_assert(square_matrix<SelfAdjointMatrix<M00, TriangleType::lower>, Likelihood::maybe>);

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
  static_assert(square_matrix<Sal22>);
  static_assert(square_matrix<Sal20_2>);
  static_assert(square_matrix<Sal02_2>);
  static_assert(square_matrix<Sal00_22>);
  static_assert(square_matrix<Sau22>);
  static_assert(square_matrix<Sau20_2>);
  static_assert(square_matrix<Sau02_2>);
  static_assert(square_matrix<Sau00_22>);
  static_assert(square_matrix<Tl22>);
  static_assert(square_matrix<Tl20_2>);
  static_assert(square_matrix<Tl02_2>);
  static_assert(square_matrix<Tl00_22>);
  static_assert(square_matrix<Tu22>);
  static_assert(square_matrix<Tu20_2>);
  static_assert(square_matrix<Tu02_2>);
  static_assert(square_matrix<Tu00_22>);
  static_assert(square_matrix<DiagonalMatrix<M11>>);
  static_assert(square_matrix<DiagonalMatrix<M10>>);
  static_assert(square_matrix<DiagonalMatrix<M01>>);
  static_assert(square_matrix<DiagonalMatrix<M00>>);
  static_assert(square_matrix<SelfAdjointMatrix<M11, TriangleType::diagonal>>);
  static_assert(square_matrix<SelfAdjointMatrix<M10, TriangleType::lower>>);
  static_assert(square_matrix<SelfAdjointMatrix<M01, TriangleType::upper>>);
  static_assert(square_matrix<SelfAdjointMatrix<M00, TriangleType::lower>>);

  static_assert(one_by_one_matrix<Z01, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<Z01>);
  static_assert(one_by_one_matrix<C10_1, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<C10_1>);
  static_assert(one_by_one_matrix<C00_1, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<C00_1>);
  static_assert(one_by_one_matrix<C11_m1>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M11>>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M10>>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M01>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<DiagonalMatrix<M01>>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M01>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<DiagonalMatrix<M00>>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M00>, Likelihood::maybe>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M11, TriangleType::diagonal>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M10, TriangleType::lower>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M01, TriangleType::upper>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M00, TriangleType::lower>, Likelihood::maybe>);
  static_assert(not one_by_one_matrix<SelfAdjointMatrix<M00, TriangleType::lower>>);

  static_assert(maybe_has_same_shape_as<>);
  static_assert(maybe_has_same_shape_as<M32>);
  static_assert(maybe_has_same_shape_as<M32, M02, M30>);
  static_assert(maybe_has_same_shape_as<M20, M23, M03>);
  static_assert(maybe_has_same_shape_as<M20, Z23>);
  static_assert(maybe_has_same_shape_as<M20, M03>);

  static_assert(has_same_shape_as<M32, M32>);
  static_assert(not has_same_shape_as<M32, M02>);
  static_assert(not has_same_shape_as<M02, M32>);
  static_assert(has_same_shape_as<M22, Sal22, M22, Z22>);
}


TEST(eigen3, cwise_nullary_operations)
{
  static_assert(native_eigen_matrix<M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::IdentityReturnType>);
  static_assert(self_contained<const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>);

  static_assert(are_within_tolerance(constant_coefficient_v<C11_1>, 1));
  static_assert(constant_matrix<typename M00::ConstantReturnType, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(not constant_matrix<typename M00::ConstantReturnType, Likelihood::maybe, CompileTimeStatus::known>);
  static_assert(are_within_tolerance(constant_coefficient_v<Z11>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z22>, 0));

  EXPECT_EQ(constant_coefficient{M22::Constant(3)}(), 3);
  EXPECT_EQ(constant_coefficient{M20::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{M02::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{M00::Constant(2, 2, 3)}(), 3);
  EXPECT_EQ(constant_coefficient{M11::Identity()}(), 1);
  EXPECT_EQ(constant_coefficient{M10::Identity(1, 1)}(), 1);
  EXPECT_EQ(constant_coefficient{M01::Identity(1, 1)}(), 1);
  EXPECT_EQ(constant_coefficient{M00::Identity(1, 1)}(), 1);

  static_assert(not zero_matrix<typename M00::ConstantReturnType, Likelihood::maybe>);
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z22>);

  static_assert(constant_diagonal_coefficient_v<I11> == 1);
  static_assert(constant_diagonal_coefficient_v<I10> == 1);
  static_assert(constant_diagonal_coefficient_v<I01> == 1);
  static_assert(constant_diagonal_coefficient_v<I00> == 1);

  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I21> == 1);
  static_assert(not constant_diagonal_matrix<typename M00::ConstantReturnType, Likelihood::maybe, CompileTimeStatus::known>);
  static_assert(constant_diagonal_matrix<typename M00::ConstantReturnType, Likelihood::maybe, CompileTimeStatus::unknown>);
  static_assert(not constant_diagonal_matrix<typename M00::ConstantReturnType, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Z11>, 0));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Z22>, 0));

  static_assert(constant_diagonal_matrix<typename M33::IdentityReturnType>);
  static_assert(constant_diagonal_matrix<typename M30::IdentityReturnType, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<typename M03::IdentityReturnType, Likelihood::maybe>);
  static_assert(constant_diagonal_matrix<typename M00::IdentityReturnType, Likelihood::maybe>);
  static_assert(not constant_diagonal_matrix<typename M30::IdentityReturnType>);
  static_assert(not constant_diagonal_matrix<typename M03::IdentityReturnType>);
  static_assert(not constant_diagonal_matrix<typename M00::IdentityReturnType>);

  EXPECT_EQ(constant_diagonal_coefficient{M22::Identity()}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M20::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M02::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M00::Identity(2, 2)}(), 1);
  EXPECT_EQ(constant_diagonal_coefficient{M11::Constant(3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{M10::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{M01::Constant(1, 1, 3)}(), 3);
  EXPECT_EQ(constant_diagonal_coefficient{M00::Constant(1, 1, 3)}(), 3);

  static_assert(identity_matrix<typename M33::IdentityReturnType>);
  static_assert(identity_matrix<typename M30::IdentityReturnType, Likelihood::maybe>);
  static_assert(identity_matrix<typename M03::IdentityReturnType, Likelihood::maybe>);
  static_assert(identity_matrix<typename M00::IdentityReturnType, Likelihood::maybe>);
  static_assert(not identity_matrix<typename M30::IdentityReturnType>);
  static_assert(not identity_matrix<typename M03::IdentityReturnType>);
  static_assert(not identity_matrix<typename M00::IdentityReturnType>);
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<Z22>);

  static_assert(diagonal_matrix<typename M33::IdentityReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);

  static_assert(hermitian_matrix<M33::ConstantReturnType>);
  static_assert(not hermitian_matrix<M21::ConstantReturnType>);
  static_assert(hermitian_matrix<typename M33::IdentityReturnType>);
  static_assert(hermitian_matrix<Z22>);
  static_assert(hermitian_matrix<C11_2>);

  static_assert(lower_triangular_matrix<Z22>);

  static_assert(upper_triangular_matrix<Z22>);

  static_assert(square_matrix<Z11>);
  static_assert(square_matrix<C11_1>);

  static_assert(square_matrix<Z11, Likelihood::maybe>);
  static_assert(square_matrix<Z20, Likelihood::maybe>);
  static_assert(not square_matrix<Z21, Likelihood::maybe>);
  static_assert(square_matrix<C22_1, Likelihood::maybe>);
  static_assert(not square_matrix<C21_1, Likelihood::maybe>);

  static_assert(one_by_one_matrix<Z11>);
  static_assert(one_by_one_matrix<Z10, Likelihood::maybe>);
  static_assert(one_by_one_matrix<Z01, Likelihood::maybe>);
  static_assert(one_by_one_matrix<C11_1>);

  static_assert(not writable<M02::ConstantReturnType>);
  static_assert(not writable<M20::IdentityReturnType>);

  static_assert(not modifiable<M33::ConstantReturnType, M33>);
  static_assert(not modifiable<M33::IdentityReturnType, M33>);
  static_assert(not modifiable<M33::ConstantReturnType, M33::ConstantReturnType>);
  static_assert(not modifiable<M33::IdentityReturnType, M33::IdentityReturnType>);
}


TEST(eigen3, cwise_unary_operations)
{
  auto id = I21 {2, 2}; // Identity
  auto zero = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  static_assert(self_contained<const M22>);

  // scalar_opposite_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(-std::declval<C11_2>())>, -2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(-std::declval<C22_2>())>, -2));
  EXPECT_EQ((constant_coefficient{-cp2}()), -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<Cd21_2>())> == -2);
  EXPECT_EQ((constant_diagonal_coefficient{-cdm2}()), 2);
  static_assert(zero_matrix<decltype(-zero)>);
  static_assert(identity_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(diagonal_matrix<decltype(-zero)>);
  static_assert(diagonal_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<Cd21_2>())>);
  static_assert(lower_triangular_matrix<decltype(-std::declval<Tl22>())>);
  static_assert(upper_triangular_matrix<decltype(-std::declval<Tu22>())>);
  static_assert(hermitian_matrix<decltype(-std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(-std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(-zero)>);
  static_assert(not hermitian_matrix<decltype(cxa)>);
  static_assert(not hermitian_matrix<decltype(-cxb)>);
  static_assert(not writable<decltype(-std::declval<M22>())>);
  static_assert(not modifiable<decltype(-std::declval<M33>()), M33>);

  // scalar_abs_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().abs())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().abs())>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().abs())> == 1);
  static_assert(identity_matrix<decltype((-id).abs())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs())>);
  static_assert(zero_matrix<decltype(zero.abs())>);
  static_assert(diagonal_matrix<decltype((id * -3).abs())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().abs())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().abs())>);
  static_assert(hermitian_matrix<decltype(cxa.abs())>);

  // scalar_score_coeff_op (inherits from scalar_abs_op)
  static_assert(constant_diagonal_coefficient_v<Eigen::CwiseUnaryOp<Eigen::internal::scalar_score_coeff_op<double>, Cd21_m2>> == 2);

  // abs_knowing_score not implemented because it is not a true Eigen functor.

  // scalar_abs2_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().abs2())>, 4));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().abs2())> == 1);
  static_assert(identity_matrix<decltype((-id).abs2())>);
  static_assert(identity_matrix<decltype(std::declval<C11_m1>().abs2())>);
  static_assert(zero_matrix<decltype(zero.abs2())>);
  static_assert(diagonal_matrix<decltype((id * -3).abs2())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().abs2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().abs2())>);
  static_assert(hermitian_matrix<decltype(cxa.abs2())>);

  // scalar_conjugate_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().conjugate())>, 1));
  EXPECT_EQ((constant_coefficient{cxa.conjugate()}()), (std::complex<double>{1, -2}));
  EXPECT_EQ((constant_coefficient{cxb.conjugate()}()), (std::complex<double>{3, -4}));
  static_assert(constant_diagonal_coefficient_v<decltype(M11::Identity().conjugate())> == 1);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().conjugate())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().conjugate())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().conjugate())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().conjugate())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().conjugate())>);
  static_assert(hermitian_matrix<decltype(std::declval<I21>().conjugate())>);

  // scalar_arg_op
  EXPECT_EQ(constant_coefficient{cp2.arg()}(), std::arg(2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().arg())> == 0);
  EXPECT_EQ(constant_coefficient{cm2.arg()}(), std::arg(-2));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().arg())> == pi);
  EXPECT_TRUE(are_within_tolerance((constant_coefficient{cxa.arg()}()), (std::arg(std::complex<double>{1, 2}))));
  EXPECT_TRUE(are_within_tolerance((constant_coefficient{cxb.arg()}()), (std::arg(std::complex<double>{3, 4}))));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().arg())>);
  static_assert(not diagonal_matrix<decltype(std::declval<Cd21_2>().arg())>);
  static_assert(not lower_triangular_matrix<decltype(std::declval<Tl22>().arg())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().arg())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().arg())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().arg())>);
  static_assert(hermitian_matrix<decltype(cxa.arg())>);

  // scalar_cast_op
  using SCOp = Eigen::internal::scalar_cast_op<double, int>;
  static_assert(std::is_same_v<typename constant_coefficient<Eigen::CwiseUnaryOp<SCOp, C22_2>>::value_type, int>);
  static_assert(std::is_same_v<typename constant_coefficient<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cast_op<std::complex<double>, std::complex<int>>, decltype(cxa)>>::value_type, std::complex<int>>);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<SCOp, Cd21_2>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<SCOp, Cd21_2>>);
  static_assert(lower_triangular_matrix<Eigen::CwiseUnaryOp<SCOp, Tl22>>);
  static_assert(upper_triangular_matrix<Eigen::CwiseUnaryOp<SCOp, Tu22>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<SCOp, Sal22>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<SCOp, Sau22>>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_cast_op<std::complex<double>, std::complex<int>>, decltype(cxa)>>);

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  auto id1_int = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<int>, Eigen::Array<int, 1, 1>> {1, 1}; // Identity
  auto id2_int = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<int>, Eigen::Array<int, 2, 2>> {2, 2}; // Identity
  auto cp2_int = (id1_int + id1_int).replicate<2, 2>(); // Constant +2
  auto cdp2_int = id2_int * 2; // Constant diagonal +2

  // scalar_shift_right_op
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftRight<1>()}(), cp2_int.shiftRight<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftRight<1>())> == 1);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), cdp2_int.shiftRight<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftRight<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftRight<1>()}(), 1);
  static_assert(constant_diagonal_matrix<decltype(cdp2_int.shiftRight<1>()), Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().cast<int>().shiftRight<1>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().cast<int>().shiftRight<1>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().cast<int>().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cast<int>().shiftRight<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().cast<int>().shiftRight<1>())>);

  // scalar_shift_left_op
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(constant_coefficient{cp2_int.shiftLeft<1>()}(), cp2_int.shiftLeft<1>()(0, 1));
  static_assert(constant_coefficient_v<decltype(cp2_int.shiftLeft<1>())> == 4);
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), cdp2_int.shiftLeft<1>()(0, 0));
  EXPECT_EQ(0, cdp2_int.shiftLeft<1>()(0, 1));
  EXPECT_EQ(constant_diagonal_coefficient{cdp2_int.shiftLeft<1>()}(), 4);
  static_assert(constant_diagonal_matrix<decltype(cdp2_int.shiftLeft<1>()), Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().cast<int>().shiftLeft<1>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().cast<int>().shiftLeft<1>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().cast<int>().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cast<int>().shiftLeft<1>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().cast<int>().shiftLeft<1>())>);
#endif

  // scalar_real_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(real(C22_2 {std::declval<C22_2>()}))>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(real(Cd21_2 {std::declval<Cd21_2>()}))> == 2);
  EXPECT_EQ((constant_coefficient{real(cxa)}()), 1);
  EXPECT_EQ((constant_coefficient{real(cxb)}()), 3);
  static_assert(hermitian_matrix<decltype(real(cxa))>);

  // scalar_imag_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(imag(C22_2 {std::declval<C22_2>()}))>, 0));
  static_assert(constant_diagonal_coefficient_v<decltype(imag(Cd21_2 {std::declval<Cd21_2>()}))> == 0);
  EXPECT_EQ((constant_coefficient{imag(cxa)}()), 2);
  EXPECT_EQ((constant_coefficient{imag(cxb)}()), 4);
  static_assert(hermitian_matrix<decltype(imag(cxa))>);

  // scalar_real_ref_op -- Eigen::CwiseUnaryView
  static_assert(not self_contained<decltype(std::declval<CA22>().real())>);
  static_assert(not self_contained<decltype(std::declval<C22_2>().real())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().real())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>().real())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().real())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().real())>, 2));
  EXPECT_EQ(constant_coefficient{cxa.real()}(), 1);
  EXPECT_EQ(constant_coefficient{cxb.real()}(), 3);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().real())> == 2);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_m1>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Z22>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().real())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().real())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().real())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().real())>);
  static_assert(hermitian_matrix<decltype(cxa.real())>);

  // scalar_imag_ref_op -- Eigen::CwiseUnaryView
  static_assert(not self_contained<decltype(std::declval<CA22>().imag())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().imag())>, 0));
  EXPECT_EQ(constant_coefficient{cxa.imag()}(), 2);
  EXPECT_EQ(constant_coefficient{cxb.imag()}(), 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().imag())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().imag())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().imag())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().imag())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().imag())>);
  static_assert(hermitian_matrix<decltype(cxa.imag())>);

  // scalar_exp_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().exp())>, constexpr_exp(2)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().exp())>, constexpr_exp(-2)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.exp()}(), cp2.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cm2.exp()}(), cm2.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.exp()}(), cxa.exp()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.exp()}(), cxb.exp()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().exp())>);
  static_assert(not zero_matrix<decltype(zero.exp())>);
  static_assert(not diagonal_matrix<decltype((id * -3).exp())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().exp())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().exp())>);
  static_assert(not hermitian_matrix<decltype(cxa.exp())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_expm1_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().expm1())>, constexpr_expm1(2)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().expm1())>, constexpr_expm1(-2)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.expm1()}(), cp2.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cm2.expm1()}(), cm2.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.expm1()}(), cxa.expm1()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.expm1()}(), cxb.expm1()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().expm1())> == constexpr_expm1(2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().expm1())> == constexpr_expm1(-1));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.expm1()}(), cdp2.expm1()(0, 0)));
  EXPECT_EQ(0, cdp2.expm1()(0, 1));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdm2.expm1()}(), cdm2.expm1()(0, 0)));
  EXPECT_EQ(0, cdm2.expm1()(0, 1));
  static_assert(zero_matrix<decltype(zero.expm1())>);
  static_assert(diagonal_matrix<decltype((id * -3).expm1())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().expm1())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().expm1())>);
  static_assert(not hermitian_matrix<decltype(cxa.expm1())>); // because cxa is not necessarily hermitian
#endif

  // scalar_log_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log())>, constexpr_log(2)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log()}(), cp2.log()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.log()}(), cxa.log()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log()}(), cxb.log()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().log())>);
  static_assert(not diagonal_matrix<decltype((id * 3).log())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().log())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().log())>);
  static_assert(not hermitian_matrix<decltype(cxa.log())>); // because cxa is not necessarily hermitian

  // scalar_log1p_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log1p())>, constexpr_log(2+1)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log1p()}(), cp2.log1p()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.log1p()}(), cxa.log1p()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log1p()}(), cxb.log1p()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().log1p())> == constexpr_log(2+1));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.log1p()}(), cdp2.log1p()(0, 0)));
  EXPECT_EQ(0, cdp2.log1p()(0, 1));
  static_assert(zero_matrix<decltype(zero.log1p())>);
  static_assert(diagonal_matrix<decltype((id * -3).log1p())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().log1p())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().log1p())>);
  static_assert(not hermitian_matrix<decltype(cxa.log1p())>); // because cxa is not necessarily hermitian

  // scalar_log10_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log10())>, constexpr_log(2) / numbers::ln10_v<double>));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log10()}(), cp2.log10()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.log10()}(), cxa.log10()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log10()}(), cxb.log10()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log10())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().log10())>);
  static_assert(not diagonal_matrix<decltype((id * 3).log10())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().log10())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().log10())>);
  static_assert(not hermitian_matrix<decltype(cxa.log10())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_log2_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().log2())>, constexpr_log(2) / numbers::ln2_v<double>));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.log2()}(), cp2.log2()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.log2()}(), cxa.log2()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.log2()}(), cxb.log2()(0, 0)));
  static_assert(zero_matrix<decltype(std::declval<C22_1>().log2())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().log2())>);
  static_assert(not diagonal_matrix<decltype((id * 3).log2())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().log2())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().log2())>);
  static_assert(not hermitian_matrix<decltype(cxa.log2())>); // because cxa is not necessarily hermitian
#endif

  // scalar_sqrt_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sqrt())>, constexpr_sqrt(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sqrt()}(), cp2.sqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sqrt()}(), cxa.sqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.sqrt()}(), cxb.sqrt()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().sqrt())> == constexpr_sqrt(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sqrt()}(), cdp2.sqrt()(0, 0)));
  EXPECT_EQ(0, cdp2.sqrt()(0, 1));
  static_assert(zero_matrix<decltype(zero.sqrt())>);
  static_assert(diagonal_matrix<decltype((id * 3).sqrt())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().sqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().sqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.sqrt())>); // because cxa is not necessarily hermitian

  // scalar_rsqrt_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().rsqrt())>, 1./constexpr_sqrt(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.rsqrt()}(), cp2.rsqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.rsqrt()}(), cxa.rsqrt()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.rsqrt()}(), cxb.rsqrt()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().rsqrt())>);
  static_assert(not zero_matrix<decltype(zero.rsqrt())>);
  static_assert(not diagonal_matrix<decltype((id * 3).rsqrt())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().rsqrt())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().rsqrt())>);
  static_assert(not hermitian_matrix<decltype(cxa.rsqrt())>); // because cxa is not necessarily hermitian

  // scalar_cos_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().cos())>, constexpr_cos(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cos()}(), cp2.cos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cos()}(), cxa.cos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.cos()}(), cxb.cos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().cos())>);
  static_assert(not zero_matrix<decltype(zero.cos())>);
  static_assert(not diagonal_matrix<decltype((id * 3).cos())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().cos())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cos())>);
  static_assert(not hermitian_matrix<decltype(cxa.cos())>); // because cxa is not necessarily hermitian

  // scalar_sin_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sin())>, constexpr_sin(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sin()}(), cp2.sin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sin()}(), cxa.sin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.sin()}(), cxb.sin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().sin())> == constexpr_sin(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sin()}(), cdp2.sin()(0, 0)));
  EXPECT_EQ(0, cdp2.sin()(0, 1));
  static_assert(zero_matrix<decltype(zero.sin())>);
  static_assert(diagonal_matrix<decltype((id * 3).sin())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().sin())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().sin())>);
  static_assert(not hermitian_matrix<decltype(cxa.sin())>); // because cxa is not necessarily hermitian

  // scalar_tan_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().tan())>, constexpr_tan(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.tan()}(), cp2.tan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.tan()}(), cxa.tan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.tan()}(), cxb.tan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().tan())> == constexpr_tan(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.tan()}(), cdp2.tan()(0, 0)));
  EXPECT_EQ(0, cdp2.tan()(0, 1));
  static_assert(zero_matrix<decltype(zero.tan())>);
  static_assert(diagonal_matrix<decltype((id * 3).tan())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().tan())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().tan())>);
  static_assert(not hermitian_matrix<decltype(cxa.tan())>); // because cxa is not necessarily hermitian

  // scalar_acos_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_1>().acos())>, constexpr_acos(1.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{(cp2/4).acos()}(), (cp2/4).acos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.acos()}(), cxa.acos()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.acos()}(), cxb.acos()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_m1>().acos())>);
  static_assert(not zero_matrix<decltype(zero.acos())>);
  static_assert(not diagonal_matrix<decltype((id* 0.25).acos())>);
  static_assert(not upper_triangular_matrix<decltype((std::declval<Tu22>()* 0.25).acos())>);
  static_assert(hermitian_matrix<decltype((std::declval<Sal22>()* 0.25).acos())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).acos())>); // because cxa is not necessarily hermitian

  // scalar_asin_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_1>().asin())>, constexpr_asin(1.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{(cp2/4).asin()}(), (cp2/4).asin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.asin()}(), cxa.asin()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.asin()}(), cxb.asin()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().asin())> == constexpr_asin(-1.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{(cdp2*0.25).asin()}(), (cdp2*0.25).asin()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).asin()(0, 1));
  static_assert(zero_matrix<decltype(zero.asin())>);
  static_assert(diagonal_matrix<decltype((id * 0.25).asin())>);
  static_assert(upper_triangular_matrix<decltype((std::declval<Tu22>() * 0.25).asin())>);
  static_assert(hermitian_matrix<decltype((std::declval<Sal22>() * 0.25).asin())>);
  static_assert(not hermitian_matrix<decltype((cxa* 0.25).asin())>); // because cxa is not necessarily hermitian

  // scalar_atan_op
  //static_assert(constexpr_atan(11.) > 1.1);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().atan())>, constexpr_atan(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.atan()}(), cp2.atan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.atan()}(), cxa.atan()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.atan()}(), cxb.atan()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m2>().atan())> == constexpr_atan(-2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{(cdp2*0.25).atan()}(), (cdp2*0.25).atan()(0, 0)));
  EXPECT_EQ(0, (cdp2*0.25).atan()(0, 1));
  static_assert(zero_matrix<decltype(zero.atan())>);
  static_assert(diagonal_matrix<decltype((id * 0.25).atan())>);
  static_assert(upper_triangular_matrix<decltype((std::declval<Tu22>() * 0.25).atan())>);
  static_assert(hermitian_matrix<decltype((std::declval<Sal22>() * 0.25).atan())>);
  static_assert(not hermitian_matrix<decltype((cxa * 0.25).atan())>); // because cxa is not necessarily hermitian

  // scalar_tanh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().tanh())>, constexpr_tanh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.tanh()}(), cp2.tanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.tanh()}(), cxa.tanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.tanh()}(), cxb.tanh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().tanh())> == constexpr_tanh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.tanh()}(), cdp2.tanh()(0, 0)));
  EXPECT_EQ(0, cdp2.tanh()(0, 1));
  static_assert(zero_matrix<decltype(zero.tanh())>);
  static_assert(diagonal_matrix<decltype((id * 3).tanh())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().tanh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().tanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.tanh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_atanh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().atanh())>, constexpr_atanh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.atanh()}(), cp2.atanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.atanh()}(), cxa.atanh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.atanh()}(), cxb.atanh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().atanh())> == constexpr_atanh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.atanh()}(), cdp2.atanh()(0, 0)));
  EXPECT_EQ(0, cdp2.atanh()(0, 1));
  static_assert(zero_matrix<decltype(zero.atanh())>);
  static_assert(diagonal_matrix<decltype((id * 3).atanh())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().atanh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().atanh())>);
  static_assert(not hermitian_matrix<decltype(cxa.atanh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_sinh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().sinh())>, constexpr_sinh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.sinh()}(), cp2.sinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.sinh()}(), cxa.sinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.sinh()}(), cxb.sinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().sinh())> == constexpr_sinh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.sinh()}(), cdp2.sinh()(0, 0)));
  EXPECT_EQ(0, cdp2.sinh()(0, 1));
  static_assert(zero_matrix<decltype(zero.sinh())>);
  static_assert(diagonal_matrix<decltype((id * 3).sinh())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().sinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().sinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.sinh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_asinh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().asinh())>, constexpr_asinh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.asinh()}(), cp2.asinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.asinh()}(), cxa.asinh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.asinh()}(), cxb.asinh()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().asinh())> == constexpr_asinh(2.));
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.asinh()}(), cdp2.asinh()(0, 0)));
  EXPECT_EQ(0, cdp2.asinh()(0, 1));
  static_assert(zero_matrix<decltype(zero.asinh())>);
  static_assert(diagonal_matrix<decltype((id * 3).asinh())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().asinh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().asinh())>);
  static_assert(not hermitian_matrix<decltype(cxa.asinh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_cosh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().cosh())>, constexpr_cosh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cosh()}(), cp2.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cosh()}(), cxa.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.cosh()}(), cxb.cosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().cosh())>);
  static_assert(not zero_matrix<decltype(zero.cosh())>);
  static_assert(not diagonal_matrix<decltype((id * 3).cosh())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().cosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.cosh())>); // because cxa is not necessarily hermitian

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_acosh_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().cos())>, constexpr_cosh(2.)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cosh()}(), cp2.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cosh()}(), cxa.cosh()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.cosh()}(), cxb.cosh()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().cosh())>);
  static_assert(not zero_matrix<decltype(zero.cosh())>);
  static_assert(not diagonal_matrix<decltype((id * 3).cosh())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().cosh())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cosh())>);
  static_assert(not hermitian_matrix<decltype(cxa.cosh())>); // because cxa is not necessarily hermitian
#endif

  // scalar_inverse_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().inverse())>, 0.5));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.inverse()}(), cp2.inverse()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.inverse()}(), cxa.inverse()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().inverse())>);
  static_assert(not zero_matrix<decltype(zero.inverse())>);
  static_assert(not diagonal_matrix<decltype((id * 3).inverse())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().inverse())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().inverse())>);
  static_assert(not hermitian_matrix<decltype(cxa.inverse())>); // because cxa is not necessarily hermitian

  // scalar_square_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().square())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().square())>, 4));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.square()}(), cp2.square()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.square()}(), cxa.square()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.square()}(), cxb.square()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().square())> == 4);
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.square()}(), cdp2.square()(0, 0)));
  EXPECT_EQ(0, cdp2.square()(0, 1));
  static_assert(zero_matrix<decltype(zero.square())>);
  static_assert(diagonal_matrix<decltype((id * 3).square())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().square())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().square())>);
  static_assert(not hermitian_matrix<decltype(cxa.square())>); // because cxa is not necessarily hermitian

  // scalar_cube_op
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().cube())> == 8));
  static_assert(constant_coefficient_v<decltype(std::declval<C22_m2>().cube())> == -8));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.cube()}(), cp2.cube()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.cube()}(), cxa.cube()(0, 0)));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().cube())> == 4);
  EXPECT_TRUE(are_within_tolerance(constant_diagonal_coefficient{cdp2.cube()}(), cdp2.cube()(0, 0)));
  EXPECT_EQ(0, cdp2.cube()(0, 1));
  static_assert(zero_matrix<decltype(zero.cube())>);
  static_assert(diagonal_matrix<decltype((id * 3).cube())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().cube())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().cube())>);
  static_assert(not hermitian_matrix<decltype(cxa.cube())>); // because cxa is not necessarily hermitian

  // EGI::scalar_round_op not implemented
  // EGI::scalar_floor_op not implemented
  // EGI::scalar_rint_op not implemented (Eigen 3.4+)
  // EGI::scalar_ceil_op not implemented

  // EGI::scalar_isnan_op not implemented
  // EGI::scalar_isinf_op not implemented
  // EGI::scalar_isfinite_op not implemented

  // scalar_boolean_not_op
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_true>())> == false);
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_false>())> == true);
  static_assert(constant_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_coefficient_v<decltype(not std::declval<Z22>())> == true); // requires narrowing from 0 to false.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_true>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_false>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<C22_1>())> == false); // requires narrowing from 1 to true.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<Z11>())> == true); // requires narrowing from 0 to false.
  static_assert(zero_matrix<decltype(not std::declval<B22_true>())>);
  static_assert(not diagonal_matrix<decltype(not std::declval<B22_false>())>);
  static_assert(hermitian_matrix<decltype(not std::declval<B22_false>())>);

  // EGI::scalar_sign_op not implemented

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_logistic_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().logistic())>, 1 / (1 + (internal::constexpr_exp(-2)))));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().logistic())>, 1 / (1 + (internal::constexpr_exp(2)))));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cp2.logistic()}(), cp2.logistic()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxa.logistic()}(), cxa.logistic()(0, 0)));
  EXPECT_TRUE(are_within_tolerance(constant_coefficient{cxb.logistic()}(), cxb.logistic()(0, 0)));
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().logistic())>);
  static_assert(not zero_matrix<decltype(zero.logistic())>);
  static_assert(not diagonal_matrix<decltype((id * 3).logistic())>);
  static_assert(not upper_triangular_matrix<decltype(std::declval<Tu22>().logistic())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().logistic())>);
  static_assert(not hermitian_matrix<decltype(cxa.logistic())>); // because cxa is not necessarily hermitian
#endif

  // bind1st_op
  using CB1sum = Eigen::internal::bind1st_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>{cp2, CB1sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>{cm2, CB1sum{6}}}()), 4);
  static_assert(constant_matrix<Eigen::CwiseUnaryOp<CB1sum, Z22>, Likelihood::definitely, CompileTimeStatus::unknown>);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cdp2)>{cdp2, CB1sum{3}}}()), 5);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB1sum, decltype(cdm2)>{cdm2, CB1sum{6}}}()), 4);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1sum, decltype(cm2)>>);

  using CB1prod = Eigen::internal::bind1st_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>{cm2, CB1prod{6}}}()), -12);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB1prod, decltype(cdp2)>{cdp2, CB1prod{3}}}()), 6);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(id)>, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(zero_matrix<Eigen::CwiseUnaryOp<CB1prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB1prod, Md21>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cp2)>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB1prod, decltype(cm2)>>);

  // bind2nd_op
  using CB2sum = Eigen::internal::bind2nd_op<Eigen::internal::scalar_sum_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>{cp2, CB2sum{3}}}()), 5);
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>{cm2, CB2sum{6}}}()), 4);
  static_assert(constant_matrix<Eigen::CwiseUnaryOp<CB2sum, Z22>, Likelihood::definitely, CompileTimeStatus::unknown>);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cdp2)>{cdp2, CB2sum{3}}}()), 5);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB2sum, decltype(cdm2)>{cdm2, CB2sum{6}}}()), 4);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2sum, decltype(cm2)>>);

  using CB2prod = Eigen::internal::bind2nd_op<Eigen::internal::scalar_product_op<double, double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>{cp2, CB2prod{3}}}()), 6);
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseUnaryOp<CB2prod, decltype(cdm2)>{cdm2, CB2prod{6}}}()), -12);
  static_assert(constant_diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(id)>, Likelihood::definitely, CompileTimeStatus::unknown>);
  static_assert(zero_matrix<Eigen::CwiseUnaryOp<CB2prod, Z22>>);
  static_assert(diagonal_matrix<Eigen::CwiseUnaryOp<CB2prod, Md21>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>, Likelihood::maybe>);
  static_assert(not hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cp2)>>);
  static_assert(hermitian_matrix<Eigen::CwiseUnaryOp<CB2prod, decltype(cm2)>>);
}


TEST(eigen3, cwise_binary_operations)
{
  auto id = I21 {2, 2}; // Identity
  auto cid = Eigen::CwiseNullaryOp<EGI::scalar_identity_op<cdouble>, CA22> {2, 2};
  auto zero = id - id; // Zero
  auto cp2 = (I11 {1, 1} + I11 {1, 1}).replicate(2, 2); // Constant +2
  auto cm2 = (-(I11 {1, 1} + I11 {1, 1})).replicate<2, 2>(); // Constant -2
  auto cxa = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{1, 2}}; // Constant complex
  auto cxb = Eigen::CwiseNullaryOp<EGI::scalar_constant_op<cdouble>, CA22> {2, 2, std::complex<double>{3, 4}}; // Constant complex
  auto cdp2 = id * 2; // Constant diagonal +2
  auto cdm2 = id * -2; // Constant diagonal -2

  // general CwiseBinaryOp
  static_assert(self_contained<decltype(2 * std::declval<I21>() + std::declval<I21>())>);
  static_assert(not self_contained<decltype(2 * std::declval<I21>() + A22 {1, 2, 3, 4})>);
  static_assert(row_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I21>() + std::declval<I21>())>> == 2);
  static_assert(column_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I21>() + std::declval<I21>())>> == 2);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, A22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const A22, const A22>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const A22,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, A22>>>);

  // scalar_sum_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_3>())>, 5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>, 0));
  EXPECT_EQ((constant_coefficient{cxa + cxb}()), (std::complex<double>{4, 6}));
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + M22::Constant(3).array()}(), 5);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() + zero}(), 2);
  EXPECT_EQ(constant_coefficient {zero + M22::Constant(3).array()}(), 3);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() + std::declval<Cd21_3>())> == 5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() + std::declval<C22_m2>())> == 0);
  EXPECT_EQ(constant_diagonal_coefficient {id + id}(), 2);
  EXPECT_EQ((constant_diagonal_coefficient {I11{1, 1} + M11::Constant(3).array()}()), 4);

  static_assert(zero_matrix<decltype(std::declval<C22_2>() + std::declval<C22_m2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md20_1>() + std::declval<Md20_1>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md01_2>() + std::declval<Md01_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md00_21>() + std::declval<Md00_21>())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>() + std::declval<Tl22>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>() + std::declval<Tu22>())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tu22>() + std::declval<Tl22>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() + std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal20_2>() + std::declval<Sau02_2>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() + std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau20_2>() + std::declval<Sau02_2>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal20_2>() + std::declval<Sau02_2>())>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() + std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() + std::declval<Sau22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() + std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() + std::declval<Sau22>())> == TriangleType::none);

  static_assert(not writable<decltype(std::declval<M22>() + std::declval<M22>())>);
  static_assert(not modifiable<decltype(std::declval<M33>() + std::declval<M33>()), M33>);


  // scalar_product_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<C21_m2>())>, -4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() * std::declval<Z21>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<C21_2>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<M21>().array() * std::declval<Z21>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>() * std::declval<M21>().array())>, 0));
  EXPECT_EQ((constant_coefficient{cxa * cxb}()), (std::complex<double>{-5, 10}));
  EXPECT_EQ((constant_coefficient{cp2 * cm2}()), -4);
  EXPECT_EQ(constant_coefficient {M22::Constant(2).array() * M22::Constant(3).array()}(), 6);
  EXPECT_EQ((constant_coefficient{zero * M22::Constant(3).array()}()), 0);
  EXPECT_EQ((constant_coefficient{M22::Constant(2).array() * zero}()), 0);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<Cd21_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<A22>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<A22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() * std::declval<Cd21_3>())> == 6); // no conjugate-product test
  static_assert(constant_diagonal_matrix<decltype(cdp2), Likelihood::definitely, CompileTimeStatus::unknown>);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2}()), 2);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * cdm2}()), -4);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * zero}()), 0);
  EXPECT_EQ((constant_diagonal_coefficient{zero * cdm2}()), 0);
  EXPECT_EQ((constant_diagonal_coefficient{cdp2 * id}()), 2);
  EXPECT_EQ((constant_diagonal_coefficient{id * cdm2}()), -2);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * std::declval<M21>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * 2)>);
  static_assert(diagonal_matrix<decltype(3 * std::declval<Md21>())>);
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tl22>() * std::declval<Tu22>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu22>() * std::declval<Tl22>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu20_2>() * std::declval<Tl02_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu00_22>() * std::declval<Tl00_22>())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(lower_triangular_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>() * std::declval<Tl22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl20_2>() * std::declval<Tl02_2>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl00_22>() * std::declval<Tl00_22>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(upper_triangular_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>() * std::declval<Tu22>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu20_2>() * std::declval<Tu02_2>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu00_22>() * std::declval<Tu00_22>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Md21>() * std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(hermitian_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() * std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(hermitian_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() * std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() * std::declval<Sau22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() * std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() * std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() * std::declval<Sau22>())> == TriangleType::none);

  static_assert(not writable<decltype(std::declval<I21>() * 2)>);
  static_assert(not modifiable<decltype(M33::Identity() * 2), decltype(M33::Identity() * 2)>);

  // scalar_conj_product_op
  using CProd = Eigen::internal::scalar_conj_product_op<std::complex<double>, std::complex<double>>;
  EXPECT_EQ((constant_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cxa), decltype(cxb)>{cxa, cxb}}()), (std::complex<double>{11, -2}));
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cxa), decltype(cid)>{cxa, cid}}()), (std::complex<double>{1, -2}));
  EXPECT_EQ((constant_diagonal_coefficient{Eigen::CwiseBinaryOp<CProd, decltype(cid), decltype(cxb)>{cid, cxb}}()), (std::complex<double>{3, 4}));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<CProd, Md21, Md21>>);
  static_assert(OpenKalman::interface::TriangularTraits<Eigen::CwiseBinaryOp<CProd, M22, Md21>>::triangle_type == TriangleType::diagonal);
  static_assert(OpenKalman::interface::TriangularTraits<Eigen::CwiseBinaryOp<CProd, Md21, M22>>::triangle_type == TriangleType::diagonal);
  static_assert(OpenKalman::interface::TriangularTraits<Eigen::CwiseBinaryOp<CProd, Tl22, Tu22>>::triangle_type == TriangleType::diagonal);
  static_assert(OpenKalman::interface::TriangularTraits<Eigen::CwiseBinaryOp<CProd, Tu22, Tl22>>::triangle_type == TriangleType::diagonal);
  static_assert(lower_triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tl22, Tl22>>);
  static_assert(lower_triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tl22, M22>>);
  static_assert(upper_triangular_matrix<Eigen::CwiseBinaryOp<CProd, Tu22, Tu22>>);
  static_assert(upper_triangular_matrix<Eigen::CwiseBinaryOp<CProd, M22, Tu22>>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<CProd, Sau22, Sal22>>);
  static_assert(OpenKalman::interface::HermitianTraits<Eigen::CwiseBinaryOp<CProd, Sau22, Md21>>::is_hermitian == true);
  static_assert(OpenKalman::interface::HermitianTraits<Eigen::CwiseBinaryOp<CProd, Md21, Md21>>::is_hermitian == true);

  // scalar_min_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(cp2.min(cm2))>, -2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().min(std::declval<C21_m2>()))>, -2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().min(std::declval<Cd21_3>()))> == 2);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>().min(std::declval<Md21>()))>);
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>().min(std::declval<M21>().array()))>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().min(std::declval<Tl22>()))>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().min(std::declval<Tu22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().min(std::declval<Sal22>()))>);

  // scalar_max_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(cp2.max(cm2))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().max(std::declval<C21_m2>()))>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().max(std::declval<Cd21_3>()))> == 3);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>().max(std::declval<Md21>()))>);
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>().max(std::declval<M21>().array()))>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().max(std::declval<Tl22>()))>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().max(std::declval<Tu22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().max(std::declval<Sal22>()))>);

  // scalar_cmp_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, false));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, decltype(cm2), decltype(cp2)>{cm2, cp2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LE>, decltype(cp2), decltype(cp2)>{cp2, cp2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GT>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_GE>, decltype(cp2), decltype(cm2)>{cp2, cm2})>, true));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_NEQ>, decltype(cm2), decltype(cm2)>{cm2, cm2})>, false));
  // No test for Eigen::internal::ComparisonName::cmp_UNORD
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_EQ>, M22, M22>>);
  static_assert(not triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_cmp_op<double, double, Eigen::internal::ComparisonName::cmp_LT>, M22, M22>>);

  // scalar_hypot_op
  using CWHYP = Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, decltype(cp2), decltype(cm2)>;
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(CWHYP{cp2, cm2})>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, C21_2, C21_m2>>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Cd21_2, Cd21_3>>, OpenKalman::internal::constexpr_sqrt(13.)));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Md21, Md21>>);
  static_assert(not diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, M21, Md21>>);
  static_assert(lower_triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tl22, Tl22>>);
  static_assert(upper_triangular_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Tu22, Tu22>>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Sau22, Sal22>>);

  // scalar_pow_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(cp2.pow(cm2))>, 0.25));
  using M11_int = eigen_matrix_t<int, 1, 1>;
  using C11_3_int = decltype(M11_int::Identity() + M11_int::Identity() + M11_int::Identity());
  using C21_3_int = Eigen::Replicate<C11_3_int, 2, 1>;
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_3_int>().array().pow(std::declval<C21_3_int>().array()))>, 27));
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>().pow(std::declval<Md21>()))>);
  static_assert(not triangular_matrix<decltype(std::declval<Sau22>().pow(std::declval<Sau22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().pow(std::declval<Sal22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().pow(std::declval<Sau22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>().pow(std::declval<Sal22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().pow(std::declval<Md21>()))>);

  // scalar_difference_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_2>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_m2>())>, 4));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() - std::declval<C22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() - std::declval<Cd21_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() - std::declval<Cd21_3>())> == -1);
  static_assert(zero_matrix<decltype(std::declval<C22_2>() - std::declval<C22_2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() - std::declval<C21_2>())>);
  static_assert(identity_matrix<decltype(M33::Identity() - (M33::Identity() - M33::Identity()))>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md20_1>() - std::declval<Md01_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md01_2>() - std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md00_21>() - std::declval<Md00_21>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() - std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal00_22>() - std::declval<Sau00_22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() - std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau00_22>() - std::declval<Sau00_22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal00_22>() - std::declval<Sau00_22>())>);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() - std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() - std::declval<Sau22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sau22>() - std::declval<Sal22>())> == TriangleType::none);
  static_assert(hermitian_adapter_type_of_v<decltype(std::declval<Sal22>() - std::declval<Sau22>())> == TriangleType::none);
  EXPECT_EQ((constant_coefficient{cxa - cxb}()), (std::complex<double>{-2, -2}));

  // scalar_quotient_op
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())>, -1.5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_3>())>, 0));
  static_assert(not constant_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>())>); // divide by zero
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() / std::declval<C21_m2>())>, -1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>() / std::declval<C21_m2>())>, 0));
  static_assert(not constant_matrix<decltype(std::declval<C21_2>() / std::declval<Z21>())>); // divide by zero
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<decltype(std::declval<C11_3>() / std::declval<C11_m2>())>, -1.5));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z11>() / std::declval<C11_m2>())> == 0);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_3>() / std::declval<Z11>())>); // divide by zero
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);
  static_assert(not triangular_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);
  using CWQ = Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<std::complex<double>, std::complex<double>>, decltype(cxa), decltype(cxb)>;
  EXPECT_EQ((constant_coefficient{CWQ{cxa, cxb}}()), (std::complex<double>{11./25, 2./25}));

  // scalar_boolean_and_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() and std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() and std::declval<B22_false>())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_false>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<BI22>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_false>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() and std::declval<B22_true>())> == true);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() and std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() and std::declval<M21>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<M21>().array() and std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tl22>() and std::declval<Tu22>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu22>() and std::declval<Tl22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>() and std::declval<Tl22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>() and std::declval<A22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<A22>() and std::declval<Tl22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() and std::declval<Sal22>())>);

  // scalar_boolean_or_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() or std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() or std::declval<B22_false>())> == false);
  static_assert(constant_coefficient_v<decltype(std::declval<BI22>() or std::declval<B22_true>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() or std::declval<BI22>())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() or std::declval<B22_false>())> == true);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() or std::declval<Md21>())>);
  static_assert(not (diagonal_matrix<decltype(std::declval<Md21>() or std::declval<M21>().array())>));
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>() or std::declval<Tl22>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>() or std::declval<Tu22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() or std::declval<Sal22>())>);

  // scalar_boolean_xor_op
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>() xor std::declval<B22_true>())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>() xor std::declval<B22_true>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<BI22>() xor std::declval<BI22>())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>() xor std::declval<BI22>())> == true);
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>() xor std::declval<Md21>())>);
  static_assert(not triangular_matrix<decltype(std::declval<Tu22>() xor std::declval<Tu22>())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>() xor std::declval<Sal22>())>);

#if EIGEN_VERSION_AT_LEAST(3,4,0)
  // scalar_absolute_difference_op
  using AD = Eigen::internal::scalar_absolute_difference_op<double, double>;
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::CwiseBinaryOp<AD, C21_2, C21_m2>>, 4));
  static_assert(diagonal_matrix<Eigen::CwiseBinaryOp<AD, Md21, Md21>>);
  static_assert(lower_triangular_matrix<Eigen::CwiseBinaryOp<AD, Tl22, Tl22>>);
  static_assert(upper_triangular_matrix<Eigen::CwiseBinaryOp<AD, Tu22, Tu22>>);
  static_assert(hermitian_matrix<Eigen::CwiseBinaryOp<AD, Sau22, Sal22>>);
#endif

}


TEST(eigen3, cwise_ternary_operations)
{
  // No current tests for cwise ternary operations
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
  static_assert(self_contained<Eigen::ArrayWrapper<I21>>);
  static_assert(not self_contained<Eigen::ArrayWrapper<M32>>);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::ArrayWrapper<C22_2>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::ArrayWrapper<C20_2>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::ArrayWrapper<C02_2>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::ArrayWrapper<C00_2>>, 2));

  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<DiagonalMatrix<C21_2>>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<DiagonalMatrix<C20_2>>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<DiagonalMatrix<C01_2>>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::ArrayWrapper<DiagonalMatrix<C00_2>>> == 2);

  static_assert(zero_matrix<Eigen::ArrayWrapper<Z22>>);
  static_assert(zero_matrix<Eigen::ArrayWrapper<Z21>>);
  static_assert(zero_matrix<Eigen::ArrayWrapper<Z23>>);

  static_assert(diagonal_matrix<Md21>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<Z22>>);
  static_assert(not hermitian_adapter<Eigen::ArrayWrapper<C22_2>>);
  static_assert(lower_triangular_matrix<Md21>);
  static_assert(upper_triangular_matrix<Md21>);
}


TEST(eigen3, Eigen_Block)
{
  static_assert(native_eigen_matrix<decltype(std::declval<C22_2>().matrix().block<2,1>(0, 0))>);
  static_assert(native_eigen_array<decltype(std::declval<C22_2>().block<2,1>(0, 0))>);
  static_assert(not native_eigen_matrix<decltype(std::declval<C22_2>().block<2,1>(0, 0))>);

  static_assert(self_contained<decltype(std::declval<I21>().block<2,1>(0, 0))>);
  static_assert(not self_contained<decltype(std::declval<Eigen::ArrayWrapper<M32>>().block<2,1>(0, 0))>);
  static_assert(self_contained<decltype((2 * std::declval<I21>() + std::declval<I21>()).col(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I21>() + A22 {1, 2, 3, 4}).col(0))>);
  static_assert(self_contained<decltype((2 * std::declval<I21>() + std::declval<I21>()).row(0))>);
  static_assert(not self_contained<decltype((2 * std::declval<I21>() + A22 {1, 2, 3, 4}).row(0))>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().block<2, 1>(0, 0))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().block(2, 1, 0, 0))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 2>(0, 0))>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>().block<1, 1>(0, 0))>, 0));

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

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M22::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M20::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M02::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M00::Identity().diagonal())>, 1));

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M22::Identity().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M20::Identity().diagonal<-1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M02::Identity().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M00::Identity().diagonal<-1>())>, 0));
  static_assert(not constant_matrix<decltype(M22::Identity().diagonal<Eigen::DynamicIndex>())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())>, 2));

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<-1>())>, 0));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C10_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C01_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().matrix().diagonal())> == 2);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().matrix().diagonal<1>())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix().diagonal<-1>())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().matrix().diagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd20_1_2>().matrix().diagonal<-1>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd20_1_2>().matrix().diagonal<1>())> == 0);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd21_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd20_1_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd01_2_2>().matrix().diagonal())>);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<Cd00_21_2>().matrix().diagonal())>);
}


TEST(eigen3, Eigen_DiagonalMatrix)
{
  static_assert(row_dimension_of_v<DM2> == 2);
  static_assert(row_dimension_of_v<DM0> == dynamic_size);

  static_assert(column_dimension_of_v<DM2> == 2);
  static_assert(column_dimension_of_v<DM0> == dynamic_size);

  static_assert(self_contained<DM2>);
  static_assert(self_contained<DM0>);

  static_assert(diagonal_matrix<DM2>);
  static_assert(diagonal_matrix<DM0>);

  static_assert(lower_triangular_matrix<DM0>);

  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::DiagonalMatrix<double, 2>>, M21>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, 3>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>, M00>);

  static_assert(diagonal_matrix<Eigen::DiagonalMatrix<double, 3>>);

  static_assert(not writable<Eigen::DiagonalMatrix<double, 3>>);
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

  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::DiagonalWrapper<M21>>, const M21&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M31>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M22>>, M44>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M30>>, M00>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M01>>, M00>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::DiagonalWrapper<M00>>, M00>);

  static_assert(not self_contained<Eigen::DiagonalWrapper<M31>>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().matrix().asDiagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>().matrix().asDiagonal())>, 0));
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


// No current tests for Eigen::Inverse


// No current tests for Eigen::Map


TEST(eigen3, Eigen_MatrixWrapper)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix())>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix())> == 2);
  static_assert(zero_matrix<decltype(std::declval<Z23>().matrix())>);
  static_assert(identity_matrix<decltype(std::declval<I21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().matrix())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().matrix())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().matrix())>);
}


TEST(eigen3, Eigen_PartialReduxExpr)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<0>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<1>())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<2>())>, constexpr_sqrt(8.)));
  static_assert(are_within_tolerance<5>(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<3>())>, constexpr_pow(16., 1./3)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<Eigen::Infinity>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().squaredNorm())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().norm())>, constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().stableNorm())>, constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().hypotNorm())>, constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().rowwise().sum())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C01_2>().rowwise().sum())>, 2));
  static_assert(not constant_matrix<decltype(std::declval<C02_2>().colwise().sum())>);
#if not EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().mean())>, 2));
#endif
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().minCoeff())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().maxCoeff())>, 2));
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().all())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().colwise().any())> == true);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().count())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().count())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().count())>, 2));
  static_assert(not constant_matrix<decltype(std::declval<C02_2>().colwise().count())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().prod())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().prod())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().prod())>, 4));
  static_assert(not constant_matrix<decltype(std::declval<C02_2>().colwise().prod())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().reverse())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate<2>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().replicate(2))>, 2));

  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().lpNorm<0>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().lpNorm<1>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().lpNorm<2>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().lpNorm<3>())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().squaredNorm())> == 4);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().norm())> == 2);
#if not EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(constant_coefficient_v<decltype(std::declval<Cd01_2>().colwise().norm())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd10_2>().rowwise().norm())> == 2);
#endif
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().stableNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().hypotNorm())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().colwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_m2>().rowwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd20_1_m2>().colwise().sum())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd20_1_m2>().rowwise().sum())> == -2);
  static_assert(not constant_matrix<decltype(std::declval<Cd01_2_m2>().colwise().sum())>);
  static_assert(not constant_matrix<decltype(std::declval<Cd01_2_m2>().rowwise().sum())>);
#if not EIGEN_VERSION_AT_LEAST(3,4,0)
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_2>().colwise().mean())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd10_2>().rowwise().mean())> == 2);
  static_assert(not constant_matrix<decltype(std::declval<Cd01_2>().colwise().mean())>);
#endif
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_2>().colwise().minCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_2>().colwise().count())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd20_1_m2>().rowwise().count())> == 2);
  static_assert(not constant_matrix<decltype(std::declval<Cd01_2_m2>().colwise().count())>);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd21_2>().colwise().prod())> == 0);
  static_assert(constant_coefficient_v<decltype(std::declval<Cd20_1_m2>().rowwise().prod())> == 0);
  static_assert(not constant_matrix<decltype(std::declval<Cd01_2_m2>().colwise().prod())>);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().colwise().reverse())> == -2);
  static_assert(constant_coefficient_v<decltype(std::declval<C11_m2>().rowwise().reverse())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().colwise().replicate<1>())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<1>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().squaredNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().norm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().stableNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().hypotNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().sum())>);
#if EIGEN_VERSION_AT_LEAST(3,4,0)
  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{(M22::Identity() - M22::Identity()).colwise().mean()}), 0.);
  EXPECT_EQ(get_scalar_constant_value(constant_coefficient{M22::Zero().colwise().mean()}), 0.);
#else
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().mean())>);
#endif
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().minCoeff())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().maxCoeff())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().prod())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().reverse())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().replicate(2))>);
}


// No current tests for Eigen::PermutationWrapper.


TEST(eigen3, Eigen_Product)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_m2>().matrix())>, -8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C22_2>().matrix())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().matrix() * std::declval<C22_2>().matrix())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<C02_2>().matrix())>, 8));
  static_assert(not constant_matrix<decltype(std::declval<C20_2>().matrix() * std::declval<C02_2>().matrix())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<I21>().matrix())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<I21>().matrix() * std::declval<C22_2>().matrix())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())>, 0));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_2>().matrix())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_1>().matrix() * std::declval<C11_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().matrix() * std::declval<C11_1>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix() * std::declval<Cd21_3>().matrix())> == 6);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix() * std::declval<I21>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<I21>().matrix() * std::declval<Cd21_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().matrix() * std::declval<C21_2>().matrix())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<C22_2>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().matrix() * std::declval<Z22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().matrix() * std::declval<M22>().matrix())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>().matrix() * std::declval<Z22>().matrix())> == 0);

  static_assert(zero_matrix<decltype(std::declval<Z11>().matrix() * std::declval<Z11>().matrix())>);
  static_assert(zero_matrix<decltype(std::declval<Z12>().matrix() * std::declval<Z21>().matrix())>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Md21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Sal22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(hermitian_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Sau22>().matrix())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Tl22>().matrix())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(upper_triangular_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Tu22>().matrix())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().matrix() * std::declval<Cd21_2>().matrix())>);
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
  static_assert(interface::IndexibleObjectTraits<Z00>::max_indices == 2);
  static_assert(interface::IndexTraits<Z00, 0>::dimension == dynamic_size);
  static_assert(interface::IndexTraits<Z00, 1>::dimension == dynamic_size);
  EXPECT_EQ((interface::IndexTraits<Z00, 0>::dimension_at_runtime(z00_21)), 2);
  EXPECT_EQ((interface::IndexTraits<Z00, 1>::dimension_at_runtime(z00_21)), 1);
  static_assert(std::is_same_v<typename interface::IndexibleObjectTraits<Z00>::scalar_type, double>);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::Replicate<Z11, 1, 2>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(z20_1)>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(z01_2)>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::Replicate<C20_2, 1, 2>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::Replicate<C02_2, 1, 2>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().replicate<5,5>())>, 2));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().replicate<1,1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().replicate<5,5>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(z22)> == 0);
  static_assert(not constant_diagonal_matrix<decltype(z20_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z02_2)>);
  static_assert(not constant_diagonal_matrix<decltype(z00_22)>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().replicate<1,1>())> == 2);

  static_assert(identity_matrix<Eigen::Replicate<I21, 1, 1>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().replicate<1, 1>())>);
  static_assert(diagonal_matrix<decltype(z11.replicate<2, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<2, Eigen::Dynamic>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, 2>())>);
  static_assert(not diagonal_matrix<decltype(z11.replicate<Eigen::Dynamic, Eigen::Dynamic>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().replicate<1, 1>())>);

  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().replicate<1, 1>())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().replicate<1, 1>())>);

  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().replicate<1, 1>())>);
}


TEST(eigen3, Eigen_Reverse)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().reverse())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().reverse())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().reverse())>, 2));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<Z22, Eigen::Vertical>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::Reverse<M11::IdentityReturnType, Eigen::Horizontal>> == 1);

  static_assert(zero_matrix<decltype(std::declval<Z23>().reverse())>);

  static_assert(identity_matrix<Eigen::Reverse<I21, Eigen::BothDirections>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().reverse())>);
  static_assert(diagonal_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);
  static_assert(not diagonal_matrix<Eigen::Reverse<C00_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().reverse())>);
  static_assert(hermitian_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Tu22>().reverse())>);
  static_assert(lower_triangular_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(upper_triangular_matrix<decltype(std::declval<Tl22>().reverse())>);
  static_assert(upper_triangular_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);
}


TEST(eigen3, Eigen_Select)
{
  auto br = make_eigen_matrix<bool, 2, 2>(true, false, true, false);
  auto bsa = eigen_matrix_t<bool, 2, 2>::Identity();

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<Z22>()))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<C22_2>(), std::declval<M22>()))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(br.select(std::declval<C22_2>(), std::declval<C22_2>()))>, 2));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd21_2>(), std::declval<Z22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_true>().select(std::declval<Cd21_2>(), std::declval<M22>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<Cd21_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(br.select(std::declval<Cd21_2>(), std::declval<Cd21_2>()))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<C22_2>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Z22>()))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().select(std::declval<C11_2>(), std::declval<Z22>()))> == 2);

  static_assert(zero_matrix<decltype(std::declval<B22_true>().select(std::declval<Z22>(), M22::Identity()))>);
  static_assert(not zero_matrix<decltype(std::declval<B22_true>().select(std::declval<I21>(), std::declval<Z22>()))>);
  static_assert(zero_matrix<decltype(std::declval<B22_false>().select(std::declval<I21>(), std::declval<Z22>()))>);
  static_assert(zero_matrix<decltype(br.select(std::declval<Z22>(), std::declval<Z22>()))>);

  static_assert(diagonal_matrix<decltype(std::declval<B22_true>().select(std::declval<Md21>(), std::declval<M22>()))>);
  static_assert(diagonal_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Md21>()))>);

  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Sal22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_true>().select(std::declval<Sau22>(), std::declval<M22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sal22>()))>);
  static_assert(hermitian_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sau22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sal22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sau22>(), std::declval<Sau22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sau22>()))>);
  static_assert(hermitian_matrix<decltype(bsa.select(std::declval<Sau22>(), std::declval<Sal22>()))>);

  static_assert(lower_triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tl22>(), std::declval<M22>()))>);
  static_assert(lower_triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tl22>()))>);

  static_assert(upper_triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tu22>(), std::declval<M22>()))>);
  static_assert(upper_triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tu22>()))>);
}


TEST(eigen3, Eigen_SelfAdjointView)
{
  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>, M22&>);

  static_assert(not native_eigen_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(not writable<Eigen::SelfAdjointView<M33, Eigen::Lower>>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M30, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M03, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M00, Eigen::Lower>>, M00>);

  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<SelfAdjointMatrix<M02>>>);
  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<SelfAdjointMatrix<M20>>>);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C22_2>, Eigen::Upper>>, 2));

  static_assert(constant_matrix<C11_1_complex>);
  static_assert(std::real(constant_coefficient_v<C11_1_complex>) == 1);
  static_assert(std::imag(constant_coefficient_v<C11_1_complex>) == 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_1_complex>, Eigen::Lower>>);

  static_assert(constant_matrix<C11_2_complex, Likelihood::definitely, CompileTimeStatus::any>);
  EXPECT_EQ(std::real(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 2);
  EXPECT_EQ(std::imag(constant_coefficient{CM11::Identity() + CM11::Identity()}()), 0);
  static_assert(constant_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<C11_2_complex>, Eigen::Lower>, Likelihood::definitely, CompileTimeStatus::any>);

  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Cd21_2>, Eigen::Upper>>, 2));

  static_assert(zero_matrix<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Z22>, Eigen::Upper>>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M30::Identity(3, 3)), Eigen::Lower>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M03::Identity(3, 3)), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M00::Identity(3, 3)), Eigen::Lower>>);

  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md21>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md20_1>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md01_2>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md00_21>().matrix()), Eigen::Lower>>);

  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md21>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md20_1>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md01_2>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md00_21>().matrix()), Eigen::Upper>>);

  static_assert(lower_hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(not lower_hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Lower>>); // the diagonal must be real

  static_assert(upper_hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(not upper_hermitian_adapter<Eigen::SelfAdjointView<CM22, Eigen::Upper>>); // the diagonal must be real

  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);

  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(hermitian_adapter<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
}


TEST(eigen3, Eigen_Solve)
{
  static_assert(not self_contained<Eigen::Solve<Eigen::PartialPivLU<M31>, M31>>);
}


TEST(eigen3, Eigen_Transpose)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().transpose())>, 2));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().transpose())> == 2);

  static_assert(zero_matrix<decltype((std::declval<Z23>()).transpose())>);

  static_assert(identity_matrix<Eigen::Transpose<I21>>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sau22>().transpose())>);
  static_assert(hermitian_matrix<decltype(std::declval<Sal22>().transpose())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tu22>().transpose())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tl22>().transpose())>);
}


TEST(eigen3, Eigen_TriangularView)
{
  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::TriangularView<M22, Eigen::Upper>>, M22&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M33, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M30, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M03, Eigen::Upper>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::TriangularView<M00, Eigen::Upper>>, M00>);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<C11_2>, Eigen::Lower>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::UnitDiag>>, 1));

  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd21_2>, Eigen::Lower>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(std::declval<Tl22>().matrix()), Eigen::Upper | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(std::declval<Tu22>().matrix()), Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::UnitDiag>> == 1);

  static_assert(zero_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>);

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M30::Identity(3, 3)), Eigen::Lower>>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M03::Identity(3, 3)), Eigen::Upper>>);
  static_assert(identity_matrix<Eigen::TriangularView<decltype(M00::Identity(3, 3)), Eigen::Lower>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper | Eigen::ZeroDiag>>);
  static_assert(identity_matrix<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Upper | Eigen::UnitDiag>>);

  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md21>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md20_1>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md01_2>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md00_21>().matrix()), Eigen::Lower>>);

  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md21>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md20_1>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md01_2>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Md00_21>().matrix()), Eigen::Upper>>);

  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Tl22>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::TriangularView<decltype(std::declval<Tu22>().matrix()), Eigen::Lower>>);

  static_assert(upper_triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>>);
  static_assert(upper_triangular_matrix<Eigen::TriangularView<decltype(std::declval<Tl22>().matrix()), Eigen::Upper>>);

  static_assert(lower_triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>>);
  static_assert(upper_triangular_matrix<Eigen::TriangularView<decltype(std::declval<Tu22>().matrix()), Eigen::Lower>>);

  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Upper>>);
  static_assert(triangular_matrix<Eigen::TriangularView<M33, Eigen::Lower>>);

  static_assert(hermitian_matrix<Eigen::TriangularView<decltype(std::declval<Md21>().matrix()), Eigen::Lower>>);
  static_assert(hermitian_matrix<Eigen::TriangularView<decltype(std::declval<Md21>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_hermitian_adapter<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Lower>>);
  static_assert(diagonal_hermitian_adapter<Eigen::TriangularView<Eigen::MatrixWrapper<Z22>, Eigen::Upper>>);
}


TEST(eigen3, Eigen_VectorBlock)
{
  static_assert(native_eigen_matrix<Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 1>>);
  static_assert(native_eigen_array<Eigen::VectorBlock<Eigen::Array<double, 2, 1>, 1>>);
  static_assert(std::is_same_v<double, typename Eigen::VectorBlock<Eigen::Matrix<double, 2, 1>, 0>::Scalar>);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_2>().segment<1>(0))> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C21_m2>().segment(1, 0))> == -2);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().segment<1>(0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_m2>().segment<1>(0))> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z21>().segment<1>(0))>);
  static_assert(zero_matrix<decltype(std::declval<Z21>().segment(1, 0))>);

}


TEST(eigen3, Eigen_VectorWiseOp)
{
  static_assert(max_indices_of_v<decltype(std::declval<M34>().colwise())> == 2);
  static_assert(std::is_same_v<scalar_type_of_t<decltype(std::declval<M34>().colwise())>, double>);

  static_assert(index_dimension_of_v<decltype(std::declval<M34>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M30>().rowwise()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(std::declval<M34>().colwise()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(std::declval<M04>().colwise()), 1> == 4);

  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().colwise())> == 2);
  static_assert(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().rowwise())>);
}
