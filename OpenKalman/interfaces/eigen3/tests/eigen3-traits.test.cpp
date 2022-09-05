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

  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_size>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_size, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_size, dynamic_size>;
}


TEST(eigen3, Eigen_Matrix)
{
  static_assert(Eigen3::native_eigen_matrix<M11>);
  static_assert(Eigen3::native_eigen_matrix<M00>);
  static_assert(Eigen3::native_eigen_general<M11>);
  static_assert(Eigen3::native_eigen_general<M00>);
  static_assert(native_eigen_matrix<Eigen3::EigenWrapper<Eigen3::ConstantMatrix<M33, 1>>>);
  static_assert(native_eigen_matrix<Eigen3::EigenWrapper<Eigen3::ConstantMatrix<M00, 1>>>);
  static_assert(interface::IndexibleObjectTraits<M11>::max_indices == 2);
  static_assert(interface::IndexibleObjectTraits<M00>::max_indices == 2);
  static_assert(interface::IndexibleObjectTraits<M21>::max_indices == 2);
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
  static_assert(native_eigen_matrix<M22>);
  static_assert(native_eigen_matrix<M00>);
  static_assert(native_eigen_matrix<CM22>);
  static_assert(native_eigen_matrix<CM00>);
  static_assert(not native_eigen_matrix<double>);

  static_assert(modifiable<M33, M33>);
  static_assert(not modifiable<M33, M31>);
  static_assert(not modifiable<M33, eigen_matrix_t<int, 3, 3>>);
  static_assert(not modifiable<const M33, M33>);
  static_assert(modifiable<M33, IdentityMatrix<M33>>);
  static_assert(not modifiable<M33, int>);
}


namespace
{
  using Z11 = decltype(M11::Identity() - M11::Identity());
  using Z22 = decltype(M22::Identity() - M22::Identity());
  using Z21 = decltype((M22::Identity() - M22::Identity()).diagonal());
  using Z23 = Eigen::Replicate<Z11, 2, 3>;
  using Z12 = Eigen::Replicate<Z11, 1, 2>;
  using Z20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic>;
  using Z02 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>;
  using Z00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>;
  using Z01 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>;

  using C11_1 = M11::IdentityReturnType;
  using C22_1 = Eigen::Replicate<C11_1, 2, 2>;
  using C21_1 = Eigen::Replicate<C11_1, 2, 1>;
  using C12_1 = Eigen::Replicate<C11_1, 1, 2>;
  using C20_1 = Eigen::Replicate<C11_1, 2, Eigen::Dynamic>;
  using C10_1 = Eigen::Replicate<C11_1, 1, Eigen::Dynamic>;
  using C01_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 1>;
  using C02_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, 2>;
  using C00_1 = Eigen::Replicate<C11_1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_2 = decltype(M11::Identity() + M11::Identity());
  using C22_2 = Eigen::Replicate<C11_2, 2, 2>;
  using C21_2 = Eigen::Replicate<C11_2, 2, 1>;
  using C12_2 = Eigen::Replicate<C11_2, 1, 2>;
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C10_2 = Eigen::Replicate<C11_2, 1, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 1>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_3 = decltype(M11::Identity() + M11::Identity() + M11::Identity());
  using C21_3 = Eigen::Replicate<C11_3, 2, 1>;
  using C20_3 = Eigen::Replicate<C11_3, 2, Eigen::Dynamic>;
  using C01_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, 1>;
  using C00_3 = Eigen::Replicate<C11_3, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m1 = decltype(-M11::Identity());
  using C21_m1 = Eigen::Replicate<C11_m1, 2, 1>;
  using C20_m1 = Eigen::Replicate<C11_m1, 2, Eigen::Dynamic>;
  using C01_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, 1>;
  using C00_m1 = Eigen::Replicate<C11_m1, Eigen::Dynamic, Eigen::Dynamic>;

  using C11_m2 = decltype(-(M11::Identity() + M11::Identity()));
  using C22_m2 = Eigen::Replicate<C11_m2, 2, 2>;
  using C21_m2 = Eigen::Replicate<C11_m2, 2, 1>;

  using B11_true = decltype(eigen_matrix_t<bool, 1, 1>::Identity());
  using B11_false = decltype((not eigen_matrix_t<bool, 1, 1>::Identity().array()).matrix());
  using B22_true = decltype(eigen_matrix_t<bool, 1, 1>::Identity().replicate<2,2>());
  using B22_false = decltype((not eigen_matrix_t<bool, 1, 1>::Identity().array()).matrix().replicate<2,2>());

  using I21 = M22::IdentityReturnType;
  using I20_1 = Eigen::DiagonalWrapper<C20_1>;
  using I01_2 = Eigen::DiagonalWrapper<C01_1>;
  using I00_21 = Eigen::DiagonalWrapper<C00_1>;

  using D21_2 = Eigen::DiagonalWrapper<C21_2>;
  using D20_1_2 = Eigen::DiagonalWrapper<C20_2>;
  using D01_2_2 = Eigen::DiagonalWrapper<C01_2>;
  using D00_21_2 = Eigen::DiagonalWrapper<C00_2>;

  using D21_3 = decltype(C21_3 {M11::Identity() + M11::Identity() + M11::Identity(), 2, 1}.asDiagonal());
  using D20_1_3 = decltype(C20_3 {M11::Identity() + M11::Identity() + M11::Identity(), 2, 1}.asDiagonal());
  using D01_2_3 = decltype(C01_3 {M11::Identity() + M11::Identity() + M11::Identity(), 2, 1}.asDiagonal());
  using D00_21_3 = decltype(C00_3 {M11::Identity() + M11::Identity() + M11::Identity(), 2, 1}.asDiagonal());

  using DM2 = Eigen::DiagonalMatrix<double, 2>;
  using DM0 = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

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

  using Md21 = Eigen::ArrayWrapper<DiagonalMatrix<M21>>;
  using Md20_1 = Eigen::ArrayWrapper<DiagonalMatrix<M20>>;
  using Md01_2 = Eigen::ArrayWrapper<DiagonalMatrix<M01>>;
  using Md00_21 = Eigen::ArrayWrapper<DiagonalMatrix<M00>>;

  using Sal22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M22, TriangleType::lower>>;
  using Sal20_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M20, TriangleType::lower>>;
  using Sal02_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M02, TriangleType::lower>>;
  using Sal00_22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M00, TriangleType::lower>>;

  using Sau22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M22, TriangleType::upper>>;
  using Sau20_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M20, TriangleType::upper>>;
  using Sau02_2 = Eigen::ArrayWrapper<SelfAdjointMatrix<M02, TriangleType::upper>>;
  using Sau00_22 = Eigen::ArrayWrapper<SelfAdjointMatrix<M00, TriangleType::upper>>;

  using Tl22 = Eigen::ArrayWrapper<TriangularMatrix<M22, TriangleType::lower>>;
  using Tl20_2 = Eigen::ArrayWrapper<TriangularMatrix<M20, TriangleType::lower>>;
  using Tl02_2 = Eigen::ArrayWrapper<TriangularMatrix<M02, TriangleType::lower>>;
  using Tl00_22 = Eigen::ArrayWrapper<TriangularMatrix<M00, TriangleType::lower>>;

  using Tu22 = Eigen::ArrayWrapper<TriangularMatrix<M22, TriangleType::upper>>;
  using Tu20_2 = Eigen::ArrayWrapper<TriangularMatrix<M20, TriangleType::upper>>;
  using Tu02_2 = Eigen::ArrayWrapper<TriangularMatrix<M02, TriangleType::upper>>;
  using Tu00_22 = Eigen::ArrayWrapper<TriangularMatrix<M00, TriangleType::upper>>;
}


TEST(eigen3, Eigen_Array)
{
  static_assert(native_eigen_array<Eigen::Array<double, 3, 2>>);
  static_assert(not native_eigen_matrix<Eigen::Array<double, 3, 2>>);
  static_assert(native_eigen_array<Eigen3::EigenWrapper<Eigen3::ConstantMatrix<Eigen::Array<double, 3, 2>, 1>>>);
  static_assert(native_eigen_array<Eigen3::EigenWrapper<Eigen3::ConstantMatrix<Eigen::Array<double, 3, 2>, 1>>>);
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
  static_assert(lower_self_adjoint_matrix<Eigen::ArrayWrapper<C22_2>>);
  static_assert(upper_self_adjoint_matrix<Eigen::ArrayWrapper<C22_2>>);
  static_assert(lower_triangular_matrix<Md21>);
  static_assert(upper_triangular_matrix<Md21>);
}


TEST(eigen3, Eigen_Block)
{
  static_assert(native_eigen_matrix<decltype(std::declval<C22_2>().block<2,1>(0, 0))>);
  static_assert(native_eigen_array<decltype(std::declval<C22_2>().array().block<2,1>(0, 0))>);
  static_assert(not native_eigen_matrix<decltype(std::declval<C22_2>().array().block<2,1>(0, 0))>);

  static_assert(self_contained<decltype(std::declval<I21>().block<2,1>(0, 0))>);
  static_assert(not self_contained<decltype(std::declval<Eigen::ArrayWrapper<M32>>().block<2,1>(0, 0))>);
  static_assert(self_contained<decltype(column<0>(2 * std::declval<I21>() + std::declval<I21>()))>);
  static_assert(not self_contained<decltype(column<0>(2 * std::declval<I21>() + M22 {1, 2, 3, 4}))>);
  static_assert(self_contained<decltype(row<0>(2 * std::declval<I21>() + std::declval<I21>()))>);
  static_assert(not self_contained<decltype(row<0>(2 * std::declval<I21>() + M22 {1, 2, 3, 4}))>);

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


TEST(eigen3, Eigen_CwiseBinaryOp)
{
  static_assert(self_contained<decltype(2 * std::declval<I21>() + std::declval<I21>())>);
  static_assert(not self_contained<decltype(2 * std::declval<I21>() + M22 {1, 2, 3, 4})>);
  static_assert(row_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I21>() + std::declval<I21>())>> == 2);
  static_assert(column_dimension_of_v<std::remove_const_t<decltype(2 * std::declval<I21>() + std::declval<I21>())>> == 2);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);
  static_assert(self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<double>, M22>,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const M22, const M22>>);
  static_assert(not self_contained<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<double, double>, const M22,
    const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_3>())>, 5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>, 0));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() + std::declval<Cd21_3>())> == 5);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() + std::declval<C22_m2>())> == 0);
  static_assert(zero_matrix<decltype(std::declval<C22_2>() + std::declval<C22_m2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() + std::declval<C21_m2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md20_1>() + std::declval<Md20_1>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md01_2>() + std::declval<Md01_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md00_21>() + std::declval<Md00_21>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>() + std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sal22>() + std::declval<Sau22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sal20_2>() + std::declval<Sau02_2>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>() + std::declval<Sau22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sal22>() + std::declval<Sau22>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau20_2>() + std::declval<Sau02_2>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sau22>() + std::declval<Sal22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sal20_2>() + std::declval<Sau02_2>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Md21>() + std::declval<Md21>())>);
  static_assert(not writable<decltype(std::declval<M22>() + std::declval<M22>())>);
  static_assert(not modifiable<decltype(std::declval<M33>() + std::declval<M33>()), M33>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_2>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>() - std::declval<C21_m2>())>, 4));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() - std::declval<C22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() - std::declval<Cd21_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() - std::declval<Cd21_3>())> == -1);
  static_assert(zero_matrix<decltype(std::declval<C22_2>() - std::declval<C22_2>())>);
  static_assert(zero_matrix<decltype(std::declval<C21_2>() - std::declval<C21_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md20_1>() - std::declval<Md01_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md01_2>() - std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md00_21>() - std::declval<Md00_21>())>);
  static_assert(identity_matrix<decltype(M33::Identity() - (M33::Identity() - M33::Identity()))>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>() - std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sal00_22>() - std::declval<Sau00_22>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>() - std::declval<Sau22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau00_22>() - std::declval<Sau00_22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sau22>() - std::declval<Sal22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sal22>() - std::declval<Sau22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sal00_22>() - std::declval<Sau00_22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Md21>() - std::declval<Md21>())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().array() * std::declval<C21_m2>().array())>, -4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().array() * std::declval<Z21>().array())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>().array() * std::declval<C21_2>().array())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<M21>().array() * std::declval<Z21>().array())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>().array() * std::declval<M21>().array())>, 0));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() * std::declval<Z22>().array())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().array() * std::declval<Cd21_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>().array() * std::declval<Z22>().array())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>().array() * std::declval<M22>().array())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>() * std::declval<Cd21_3>())> == 6); // no conjugate-product test
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * std::declval<M21>().array())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>() * 2)>);
  static_assert(diagonal_matrix<decltype(3 * std::declval<Md21>())>);
  static_assert(not diagonal_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tl22>() * std::declval<Tu22>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu22>() * std::declval<Tl22>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu20_2>() * std::declval<Tl02_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Tu00_22>() * std::declval<Tl00_22>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Md21>() * std::declval<Md21>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(lower_self_adjoint_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>() * std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Md21>() * 3)>);
  static_assert(upper_self_adjoint_matrix<decltype(2 * std::declval<Md21>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>() * std::declval<Sau22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(not upper_self_adjoint_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sau22>() * std::declval<Sal22>())>);
  static_assert(self_adjoint_matrix<decltype(std::declval<Sal22>() * std::declval<Sau22>())>);
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
  static_assert(not writable<decltype(std::declval<I21>() * 2)>);
  static_assert(not modifiable<decltype(M33::Identity() * 2), decltype(M33::Identity() * 2)>);

  // no conjugate-product test

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_3>().array() / std::declval<C11_m2>().array())>, -1.5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>().array() / std::declval<C11_3>().array())>, 0));
  static_assert(not constant_matrix<decltype(std::declval<C11_3>().array() / std::declval<Z11>().array())>); // divide by zero
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().array() / std::declval<C21_m2>().array())>, -1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>().array() / std::declval<C21_m2>().array())>, 0));
  static_assert(not constant_matrix<decltype(std::declval<C21_2>().array() / std::declval<Z21>().array())>); // divide by zero
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<decltype(std::declval<C11_3>().array() / std::declval<C11_m2>().array())>, -1.5));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z11>().array() / std::declval<C11_m2>().array())> == 0);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_3>().array() / std::declval<Z11>().array())>); // divide by zero
  static_assert(not lower_self_adjoint_matrix<decltype(std::declval<Md21>() / std::declval<Md21>())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().array().min(std::declval<C21_m2>().array()))>, -2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().min(std::declval<Cd21_3>()))> == 2);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().array().max(std::declval<C21_m2>().array()))>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().max(std::declval<Cd21_3>()))> == 3);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, C21_2, C21_m2>>, OpenKalman::internal::constexpr_sqrt(8.)));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Eigen::CwiseBinaryOp<Eigen::internal::scalar_hypot_op<double, double>, Cd21_2, Cd21_3>>, OpenKalman::internal::constexpr_sqrt(13.)));

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_m2>().array().pow(std::declval<C21_3>().array()))>, -8));

  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().array() and std::declval<B22_true>().array())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().array() and std::declval<B22_false>().array())> == false);

  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().array() or std::declval<B22_true>().array())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().array() or std::declval<B22_false>().array())> == false);

  static_assert(constant_coefficient_v<decltype(std::declval<B22_false>().array() xor std::declval<B22_true>().array())> == true);
  static_assert(constant_coefficient_v<decltype(std::declval<B22_true>().array() xor std::declval<B22_true>().array())> == false);
}


TEST(eigen3, Eigen_CwiseNullaryOp)
{
  static_assert(native_eigen_matrix<M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::ConstantReturnType>);
  static_assert(self_contained<typename M33::IdentityReturnType>);
  static_assert(self_contained<const Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<double>, M22>>);

  static_assert(are_within_tolerance(constant_coefficient_v<C11_1>, 1));
  static_assert(not constant_matrix<typename M00::ConstantReturnType>); // because the constant is not known at compile time
  static_assert(are_within_tolerance(constant_coefficient_v<Z11>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Z22>, 0));

  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I21> == 1);
  static_assert(not constant_diagonal_matrix<typename M00::ConstantReturnType>); // because the constant is not known at compile time
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Z11>, 0));
  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Z22>, 0));

  static_assert(not zero_matrix<typename M00::ConstantReturnType>); // because the constant is not known at compile time
  static_assert(zero_matrix<Z11>);
  static_assert(zero_matrix<Z22>);

  static_assert(identity_matrix<typename M33::IdentityReturnType>);
  static_assert(not identity_matrix<typename M30::IdentityReturnType>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<typename M03::IdentityReturnType>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<typename M00::IdentityReturnType>); // We can't tell if it's a square matrix at compile time
  static_assert(identity_matrix<C11_1>);
  static_assert(not identity_matrix<Z22>);

  static_assert(diagonal_matrix<typename M33::IdentityReturnType>);

  static_assert(self_adjoint_matrix<M33::ConstantReturnType>);

  static_assert(not lower_self_adjoint_matrix<M33::ConstantReturnType>);
  static_assert(not lower_self_adjoint_matrix<M21::ConstantReturnType>);
  static_assert(lower_self_adjoint_matrix<typename M33::IdentityReturnType>);
  static_assert(lower_self_adjoint_matrix<Z22>);
  static_assert(lower_self_adjoint_matrix<C11_2>);

  static_assert(not upper_self_adjoint_matrix<M33::ConstantReturnType>);
  static_assert(upper_self_adjoint_matrix<typename M33::IdentityReturnType>);
  static_assert(upper_self_adjoint_matrix<Z22>);

  static_assert(lower_triangular_matrix<Z22>);

  static_assert(upper_triangular_matrix<Z22>);

  static_assert(square_matrix<Z11>);
  static_assert(square_matrix<C11_1>);

  static_assert(one_by_one_matrix<Z11>);
  static_assert(one_by_one_matrix<C11_1>);

  static_assert(not writable<M02::ConstantReturnType>);
  static_assert(not writable<M20::IdentityReturnType>);

  static_assert(not modifiable<M33::ConstantReturnType, M33>);
  static_assert(not modifiable<M33::IdentityReturnType, M33>);
  static_assert(not modifiable<M33::ConstantReturnType, M33::ConstantReturnType>);
  static_assert(not modifiable<M33::IdentityReturnType, M33::IdentityReturnType>);
}


// No current tests for CwiseTernaryOp


TEST(eigen3, Eigen_CwiseUnaryOp)
{
  static_assert(self_contained<const M22>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(-std::declval<C11_2>())>, -2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(-std::declval<C22_2>())>, -2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().conjugate())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(real(C22_2 {std::declval<C22_2>()}.array()))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(imag(C22_2 {std::declval<C22_2>()}.array()))>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().array().abs())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().array().abs())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().array().abs2())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().array().sqrt())>, OpenKalman::internal::constexpr_sqrt(2.)));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_1>().array().inverse())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().array().inverse())>, 0.5));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().array().square())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().array().cube())>, -8));
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_true>().array())> == false);
  static_assert(constant_coefficient_v<decltype(not std::declval<B22_false>().array())> == true);
  static_assert(constant_coefficient_v<decltype(not std::declval<C22_1>().array())> == false); // requires narrowing from 1 to true.
  static_assert(constant_coefficient_v<decltype(not std::declval<Z22>().array())> == true); // requires narrowing from 0 to false.

  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<C11_2>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(-std::declval<Cd21_2>())> == -2);
  static_assert(constant_diagonal_coefficient_v<decltype(M11::Identity().conjugate())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(real(Cd21_2 {std::declval<Cd21_2>()}))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(imag(Cd21_2 {std::declval<Cd21_2>()}))> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().abs())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().abs())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().abs2())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().abs2())> == 1);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().sqrt())> == OpenKalman::internal::constexpr_sqrt(2.));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_m1>().inverse())> == -1);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().square())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().cube())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_true>().array())> == false);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<B11_false>().array())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<C22_1>().array())> == false); // requires narrowing from 1 to true.
  static_assert(constant_diagonal_coefficient_v<decltype(not std::declval<Z11>().array())> == true); // requires narrowing from 0 to false.

  static_assert(identity_matrix<decltype(-std::declval<C11_m1>())>);

  static_assert(diagonal_matrix<decltype(-std::declval<Z22>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<C11_m1>())>);
  static_assert(diagonal_matrix<decltype(-std::declval<Cd21_2>())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().conjugate())>);

  static_assert(lower_self_adjoint_matrix<decltype(-std::declval<Sal22>())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().conjugate())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().cube())>);

  static_assert(upper_self_adjoint_matrix<decltype(-std::declval<Sau22>())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().conjugate())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().cube())>);

  static_assert(lower_triangular_matrix<decltype(-std::declval<Tl22>())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().conjugate())>);

  static_assert(upper_triangular_matrix<decltype(-std::declval<Tu22>())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().conjugate())>);

  static_assert(not writable<decltype(-std::declval<M22>())>);

  static_assert(not modifiable<decltype(-std::declval<M33>()), M33>);
}


TEST(eigen3, Eigen_CwiseUnaryView)
{
  static_assert(not self_contained<decltype(std::declval<CM22>().real())>);
  static_assert(not self_contained<decltype(std::declval<CM22>().imag())>);
  static_assert(not self_contained<decltype(std::declval<C22_2>().real())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().real())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M11::Identity().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>().real())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z11>().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().real())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().imag())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().real())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().imag())>, 0));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().real())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().imag())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().imag())> == 0);

  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().real())>);
  static_assert(diagonal_matrix<decltype(eigen_matrix_t<cdouble, 2, 2>::Identity().imag())>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_m1>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Z22>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().real())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().real())>);

  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().real())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().imag())>);

  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().real())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().imag())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().real())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().imag())>);

  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().real())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().imag())>);
}


TEST(eigen3, Eigen_Diagonal)
{
  static_assert(not self_contained<decltype(std::declval<M22>().diagonal())>);
  static_assert(self_contained<decltype(std::declval<C22_2>().diagonal())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M22::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M20::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M02::Identity().diagonal())>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M00::Identity().diagonal())>, 1));

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M22::Identity().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M20::Identity().diagonal<-1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M02::Identity().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(M00::Identity().diagonal<-1>())>, 0));
  static_assert(not constant_matrix<decltype(M22::Identity().diagonal<Eigen::DynamicIndex>())>);

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().diagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().diagonal<1>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().diagonal<-1>())>, 2));

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<1>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Cd21_2>().matrix().diagonal<-1>())>, 0));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C10_2>().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C01_2>().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().diagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().diagonal())> == 2);

  static_assert(not constant_diagonal_matrix<decltype(std::declval<C11_2>().diagonal<1>())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().diagonal<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>().diagonal<-1>())> == 2);
  static_assert(not constant_diagonal_matrix<decltype(std::declval<C22_2>().diagonal())>);

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

  static_assert(writable<Eigen::DiagonalMatrix<double, 3>>);
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

  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>().asDiagonal())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z21>().asDiagonal())>, 0));
  static_assert(not constant_matrix<decltype(std::declval<C21_2>().asDiagonal())>);

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().asDiagonal())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z21>().asDiagonal())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().asDiagonal())> == 2);

  static_assert(not zero_matrix<decltype(std::declval<C11_1>())>);
  static_assert(not zero_matrix<decltype(std::declval<C11_2>())>);
  static_assert(not zero_matrix<decltype(std::declval<C21_2>())>);

  static_assert(zero_matrix<decltype(std::declval<Z11>().asDiagonal())>);
  static_assert(zero_matrix<decltype(std::declval<Z21>().asDiagonal())>);

  static_assert(identity_matrix<decltype(std::declval<C11_1>().asDiagonal())>);
  static_assert(identity_matrix<decltype(std::declval<C21_1>().asDiagonal())>);

  static_assert(diagonal_matrix<Eigen::DiagonalWrapper<M31>>);
  static_assert(diagonal_matrix<decltype(std::declval<C11_2>().asDiagonal())>);
  static_assert(diagonal_matrix<decltype(std::declval<C21_2>().asDiagonal())>);

  static_assert(not writable<Eigen::DiagonalWrapper<M31>>);
}


// No current tests for Eigen::Inverse


// No current tests for Eigen::Map


TEST(eigen3, Eigen_MatrixWrapper)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().array().matrix())>, 2));
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix())> == 2);
  static_assert(zero_matrix<decltype(std::declval<Z23>().array().matrix())>);
  static_assert(identity_matrix<decltype(std::declval<I21>().array().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Cd21_2>().matrix())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().matrix())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().matrix())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().matrix())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().matrix())>);
}


TEST(eigen3, Eigen_PartialReduxExpr)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<1>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().lpNorm<2>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().squaredNorm())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().norm())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().stableNorm())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_m2>().colwise().hypotNorm())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C02_2>().rowwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().rowwise().sum())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>().colwise().sum())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C01_2>().rowwise().sum())>, 2));
  static_assert(not constant_matrix<decltype(std::declval<C02_2>().colwise().sum())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise().mean())>, 2));
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

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().lpNorm<1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().lpNorm<2>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().squaredNorm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().norm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C01_2>().colwise().norm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C10_2>().rowwise().norm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().stableNorm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_m2>().colwise().hypotNorm())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().sum())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>().rowwise().sum())> == 4);
  static_assert(not constant_matrix<decltype(std::declval<C01_2>().colwise().sum())>);
  static_assert(not constant_matrix<decltype(std::declval<C10_2>().rowwise().sum())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().mean())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C01_2>().colwise().mean())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C10_2>().rowwise().mean())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().minCoeff())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().maxCoeff())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().colwise().all())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<B11_true>().colwise().any())> == true);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().count())> == 2);
  static_assert(not constant_matrix<decltype(std::declval<C01_2>().colwise().count())>);
  static_assert(not constant_matrix<decltype(std::declval<C10_2>().rowwise().count())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C21_2>().colwise().prod())> == 4);
  static_assert(not constant_matrix<decltype(std::declval<C01_2>().colwise().prod())>);
  static_assert(not constant_matrix<decltype(std::declval<C10_2>().rowwise().prod())>);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().colwise().reverse())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().colwise().replicate<1>())> == 2);

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<1>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().lpNorm<2>())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().squaredNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().norm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().stableNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().hypotNorm())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().sum())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise().mean())>);
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
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>() * std::declval<C11_2>())>, 4));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C12_2>() * std::declval<C21_2>())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_1>() * std::declval<C11_2>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C11_2>() * std::declval<C11_1>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>() * std::declval<C22_m2>())>, -8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>() * std::declval<C22_2>())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C20_2>() * std::declval<C22_2>())>, 8));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>() * std::declval<C02_2>())>, 8));
  static_assert(not constant_matrix<decltype(std::declval<C20_2>() * std::declval<C02_2>())>);
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>() * std::declval<I21>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<I21>() * std::declval<C22_2>())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>() * std::declval<Z22>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>() * std::declval<Z22>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>() * std::declval<C22_2>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<M22>() * std::declval<Z22>())>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<Z22>() * std::declval<M22>())>, 0));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>() * std::declval<C11_2>())> == 4);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_1>() * std::declval<C11_2>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>() * std::declval<C11_1>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix() * std::declval<Cd21_3>().matrix())> == 6);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Cd21_2>().matrix() * std::declval<I21>())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<I21>() * std::declval<Cd21_2>().matrix())> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C12_2>() * std::declval<C21_2>())> == 8);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<C22_2>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C22_2>() * std::declval<Z22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<Z22>() * std::declval<M22>())> == 0);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<M22>() * std::declval<Z22>())> == 0);

  static_assert(zero_matrix<decltype(std::declval<Z11>() * std::declval<Z11>())>);
  static_assert(zero_matrix<decltype(std::declval<Z12>() * std::declval<Z21>())>);

  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Md21>().matrix())>);
  static_assert(diagonal_matrix<decltype(std::declval<Md21>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Sal22>().matrix())>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Sau22>().matrix())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(lower_triangular_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Tl22>().matrix())>);
  static_assert(lower_triangular_matrix<decltype(std::declval<Tl22>().matrix() * std::declval<Cd21_2>().matrix())>);

  static_assert(upper_triangular_matrix<decltype(std::declval<Cd21_2>().matrix() * std::declval<Tu22>().matrix())>);
  static_assert(upper_triangular_matrix<decltype(std::declval<Tu22>().matrix() * std::declval<Cd21_2>().matrix())>);
}


// No current tests for Eigen::Ref.


TEST(eigen3, Eigen_Replicate)
{
  auto z11 = M11::Identity() - M11::Identity();
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

  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sal22>().replicate<1, 1>())>);

  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sau22>().replicate<1, 1>())>);

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

  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sau22>().reverse())>);
  static_assert(lower_self_adjoint_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sal22>().reverse())>);
  static_assert(upper_self_adjoint_matrix<Eigen::Reverse<C11_2, Eigen::Vertical>>);

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

  static_assert(lower_self_adjoint_matrix<decltype(std::declval<B22_true>().select(std::declval<Sal22>(), std::declval<M22>()))>);
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sal22>()))>);
  static_assert(lower_self_adjoint_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sal22>()))>);
  static_assert(not lower_self_adjoint_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sau22>()))>);

  static_assert(upper_self_adjoint_matrix<decltype(std::declval<B22_true>().select(std::declval<Sau22>(), std::declval<M22>()))>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Sau22>()))>);
  static_assert(upper_self_adjoint_matrix<decltype(bsa.select(std::declval<Sau22>(), std::declval<Sau22>()))>);
  static_assert(not upper_self_adjoint_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sau22>()))>);

  static_assert(self_adjoint_matrix<decltype(bsa.select(std::declval<Sal22>(), std::declval<Sau22>()))>);

  static_assert(lower_triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tl22>(), std::declval<M22>()))>);
  static_assert(lower_triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tl22>()))>);

  static_assert(upper_triangular_matrix<decltype(std::declval<B22_true>().select(std::declval<Tu22>(), std::declval<M22>()))>);
  static_assert(upper_triangular_matrix<decltype(std::declval<B22_false>().select(std::declval<M22>(), std::declval<Tu22>()))>);
}


TEST(eigen3, Eigen_SelfAdjointView)
{
  static_assert(std::is_same_v<nested_matrix_of_t<Eigen::SelfAdjointView<M22, Eigen::Lower>>, M22&>);

  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M33, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M30, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M03, Eigen::Lower>>, M33>);
  static_assert(std::is_same_v<dense_writable_matrix_t<Eigen::SelfAdjointView<M00, Eigen::Lower>>, M00>);

  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<SelfAdjointMatrix<M02>>>);
  static_assert(not has_dynamic_dimensions<dense_writable_matrix_t<SelfAdjointMatrix<M20>>>);

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::SelfAdjointView<C22_2, Eigen::Upper>>, 2));

  static_assert(are_within_tolerance(constant_diagonal_coefficient_v<Eigen::SelfAdjointView<Eigen::MatrixWrapper<Cd21_2>, Eigen::Upper>>, 2));

  static_assert(zero_matrix<Eigen::SelfAdjointView<Z22, Eigen::Upper>>);

  static_assert(identity_matrix<Eigen::SelfAdjointView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M30::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M03::Identity(3, 3)), Eigen::Upper>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::SelfAdjointView<decltype(M00::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time

  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md21>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md20_1>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md01_2>().matrix()), Eigen::Lower>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md00_21>().matrix()), Eigen::Lower>>);

  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md21>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md20_1>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md01_2>().matrix()), Eigen::Upper>>);
  static_assert(diagonal_matrix<Eigen::SelfAdjointView<decltype(std::declval<Md00_21>().matrix()), Eigen::Upper>>);

  static_assert(lower_self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(not lower_self_adjoint_matrix<Eigen::SelfAdjointView<CM22, Eigen::Lower>>); // the diagonal must be real

  static_assert(upper_self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
  static_assert(not upper_self_adjoint_matrix<Eigen::SelfAdjointView<CM22, Eigen::Upper>>); // the diagonal must be real

  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Lower>>);
  static_assert(self_adjoint_matrix<Eigen::SelfAdjointView<M33, Eigen::Upper>>);
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
  static_assert(lower_self_adjoint_matrix<decltype(std::declval<Sau22>().transpose())>);
  static_assert(upper_self_adjoint_matrix<decltype(std::declval<Sal22>().transpose())>);
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

  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<C11_2, Eigen::Lower>>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Z22, Eigen::Lower>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Z22, Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<Z22, Eigen::Lower | Eigen::UnitDiag>>, 1));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::ZeroDiag>>, 0));
  static_assert(are_within_tolerance(constant_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::UnitDiag>>, 1));

  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Eigen::MatrixWrapper<Cd21_2>, Eigen::Lower>> == 2);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Z22, Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<Z22, Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(M22::Identity()), Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(std::declval<Tl22>().matrix()), Eigen::Upper | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<decltype(std::declval<Tu22>().matrix()), Eigen::Lower | Eigen::UnitDiag>> == 1);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::ZeroDiag>> == 0);
  static_assert(constant_diagonal_coefficient_v<Eigen::TriangularView<M11, Eigen::Lower | Eigen::UnitDiag>> == 1);

  static_assert(zero_matrix<Eigen::TriangularView<Z22, Eigen::Lower>>);

  static_assert(identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper>>);
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M30::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M03::Identity(3, 3)), Eigen::Upper>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M00::Identity(3, 3)), Eigen::Lower>>); // We can't tell if it's a square matrix at compile time
  static_assert(not identity_matrix<Eigen::TriangularView<decltype(M33::Identity()), Eigen::Upper | Eigen::ZeroDiag>>);
  static_assert(identity_matrix<Eigen::TriangularView<Z22, Eigen::Upper | Eigen::UnitDiag>>);

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
}


TEST(eigen3, Eigen_VectorBlock)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_2>().segment<1>(0))>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C21_m2>().segment(1, 0))>, -2));

  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_2>().segment<1>(0))> == 2);
  static_assert(constant_diagonal_coefficient_v<decltype(std::declval<C11_m2>().segment<1>(0))> == -2);

  static_assert(zero_matrix<decltype(std::declval<Z21>().segment<1>(0))>);
  static_assert(zero_matrix<decltype(std::declval<Z21>().segment(1, 0))>);

}


TEST(eigen3, Eigen_VectorWiseOp)
{
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().colwise())>, 2));
  static_assert(are_within_tolerance(constant_coefficient_v<decltype(std::declval<C22_2>().rowwise())>, 2));

  static_assert(zero_matrix<decltype(std::declval<Z22>().colwise())>);
  static_assert(zero_matrix<decltype(std::declval<Z22>().rowwise())>);
}


TEST(eigen3, Eigen_check_test_classes)
{
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

  static_assert(not constant_diagonal_matrix<Z21>);
  static_assert(not constant_diagonal_matrix<Z12>);
  static_assert(not constant_diagonal_matrix<Z23>);
  static_assert(not constant_diagonal_matrix<Z20>);
  static_assert(not constant_diagonal_matrix<Z02>);
  static_assert(not constant_diagonal_matrix<Z00>);
  static_assert(not constant_diagonal_matrix<Z01>);
  static_assert(constant_diagonal_coefficient_v<C11_1> == 1);
  static_assert(constant_diagonal_coefficient_v<C11_m1> == -1);
  static_assert(constant_diagonal_coefficient_v<C11_2> == 2);
  static_assert(constant_diagonal_coefficient_v<C11_m2> == -2);
  static_assert(not constant_diagonal_matrix<C21_1>);
  static_assert(not constant_diagonal_matrix<C20_1>);
  static_assert(not constant_diagonal_matrix<C01_1>);
  static_assert(not constant_diagonal_matrix<C00_1>);
  static_assert(constant_diagonal_coefficient_v<I21> == 1);
  static_assert(constant_diagonal_coefficient_v<I20_1> == 1);
  static_assert(constant_diagonal_coefficient_v<I01_2> == 1);
  static_assert(constant_diagonal_coefficient_v<I00_21> == 1);
  static_assert(constant_diagonal_coefficient_v<Cd21_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd20_1_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd01_2_2> == 2);
  static_assert(constant_diagonal_coefficient_v<Cd00_21_2> == 2);

  static_assert(zero_matrix<Z21>);
  static_assert(zero_matrix<Z23>);
  static_assert(zero_matrix<Z20>);
  static_assert(zero_matrix<Z02>);
  static_assert(zero_matrix<B22_false>);
  static_assert(not zero_matrix<Cd21_2>);

  static_assert(not identity_matrix<C21_1>);
  static_assert(identity_matrix<I21>);
  static_assert(identity_matrix<I20_1>);
  static_assert(identity_matrix<I01_2>);
  static_assert(identity_matrix<I00_21>);
  static_assert(not identity_matrix<Cd21_2>);
  static_assert(not identity_matrix<Cd21_3>);

  static_assert(diagonal_matrix<Z22>);
  static_assert(not diagonal_matrix<Z20>);
  static_assert(not diagonal_matrix<Z02>);
  static_assert(not diagonal_matrix<Z00>);
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

  static_assert(lower_self_adjoint_matrix<Z22>);
  static_assert(not lower_self_adjoint_matrix<Z20>);
  static_assert(not lower_self_adjoint_matrix<Z02>);
  static_assert(not lower_self_adjoint_matrix<Z00>);
  static_assert(lower_self_adjoint_matrix<C22_2>);
  static_assert(lower_self_adjoint_matrix<I21>);
  static_assert(lower_self_adjoint_matrix<I20_1>);
  static_assert(lower_self_adjoint_matrix<I01_2>);
  static_assert(lower_self_adjoint_matrix<I00_21>);
  static_assert(lower_self_adjoint_matrix<Cd21_2>);
  static_assert(lower_self_adjoint_matrix<Cd20_1_2>);
  static_assert(lower_self_adjoint_matrix<Cd01_2_2>);
  static_assert(lower_self_adjoint_matrix<Cd00_21_2>);
  static_assert(lower_self_adjoint_matrix<Md21>);
  static_assert(lower_self_adjoint_matrix<Md20_1>);
  static_assert(lower_self_adjoint_matrix<Md01_2>);
  static_assert(lower_self_adjoint_matrix<Md00_21>);
  static_assert(lower_self_adjoint_matrix<Sal22>);
  static_assert(lower_self_adjoint_matrix<Sal20_2>);
  static_assert(lower_self_adjoint_matrix<Sal02_2>);
  static_assert(lower_self_adjoint_matrix<Sal00_22>);
  static_assert(not lower_self_adjoint_matrix<Sau22>);

  static_assert(upper_self_adjoint_matrix<Z22>);
  static_assert(not upper_self_adjoint_matrix<Z20>);
  static_assert(not upper_self_adjoint_matrix<Z02>);
  static_assert(not upper_self_adjoint_matrix<Z00>);
  static_assert(upper_self_adjoint_matrix<C11_2>);
  static_assert(upper_self_adjoint_matrix<C22_2>);
  static_assert(upper_self_adjoint_matrix<I21>);
  static_assert(upper_self_adjoint_matrix<I20_1>);
  static_assert(upper_self_adjoint_matrix<I01_2>);
  static_assert(upper_self_adjoint_matrix<I00_21>);
  static_assert(upper_self_adjoint_matrix<Cd21_2>);
  static_assert(upper_self_adjoint_matrix<Cd20_1_2>);
  static_assert(upper_self_adjoint_matrix<Cd01_2_2>);
  static_assert(upper_self_adjoint_matrix<Cd00_21_2>);
  static_assert(upper_self_adjoint_matrix<Md21>);
  static_assert(upper_self_adjoint_matrix<Md20_1>);
  static_assert(upper_self_adjoint_matrix<Md01_2>);
  static_assert(upper_self_adjoint_matrix<Md00_21>);
  static_assert(upper_self_adjoint_matrix<Sau22>);
  static_assert(upper_self_adjoint_matrix<Sau20_2>);
  static_assert(upper_self_adjoint_matrix<Sau02_2>);
  static_assert(upper_self_adjoint_matrix<Sau00_22>);
  static_assert(not upper_self_adjoint_matrix<Sal22>);

  static_assert(lower_triangular_matrix<Z22>);
  static_assert(not lower_triangular_matrix<Z20>);
  static_assert(not lower_triangular_matrix<Z02>);
  static_assert(not lower_triangular_matrix<Z00>);
  static_assert(lower_triangular_matrix<C11_2>);
  static_assert(not lower_triangular_matrix<C22_2>);
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

  static_assert(upper_triangular_matrix<Z22>);
  static_assert(not upper_triangular_matrix<Z20>);
  static_assert(not upper_triangular_matrix<Z02>);
  static_assert(not upper_triangular_matrix<Z00>);
  static_assert(upper_triangular_matrix<C11_2>);
  static_assert(not upper_triangular_matrix<C22_2>);
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

  static_assert(not one_by_one_matrix<Z01>);
  static_assert(not one_by_one_matrix<C10_1>);
  static_assert(not one_by_one_matrix<C00_1>);
  static_assert(one_by_one_matrix<C11_m1>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M11>>);
  static_assert(one_by_one_matrix<DiagonalMatrix<M10>>);
  static_assert(not one_by_one_matrix<DiagonalMatrix<M01>>);
  static_assert(not one_by_one_matrix<DiagonalMatrix<M00>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M11, TriangleType::diagonal>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M10, TriangleType::lower>>);
  static_assert(one_by_one_matrix<SelfAdjointMatrix<M01, TriangleType::upper>>);
  static_assert(not one_by_one_matrix<SelfAdjointMatrix<M00, TriangleType::lower>>);
}
