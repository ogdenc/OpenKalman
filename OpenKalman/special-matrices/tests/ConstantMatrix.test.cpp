/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "special-matrices.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

using std::numbers::pi;

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

  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;
  using M10 = eigen_matrix_t<double, 1, dynamic_extent>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M04 = eigen_matrix_t<double, dynamic_extent, 4>;
  using M40 = eigen_matrix_t<double, 4, dynamic_extent>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M05 = eigen_matrix_t<double, dynamic_extent, 5>;

  using cdouble = std::complex<double>;

  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;

  using CM20 = eigen_matrix_t<cdouble, 2, dynamic_extent>;
  using CM02 = eigen_matrix_t<cdouble, dynamic_extent, 2>;
  using CM00 = eigen_matrix_t<cdouble, dynamic_extent, dynamic_extent>;

  using Axis2 = Coefficients<Axis, Axis>;

  auto z11 = M11::Identity() - M11::Identity(); using Z11 = decltype(z11);
  auto z22 = M22::Identity() - M22::Identity(); using Z22 = decltype(z22);
  auto z21 = z22.diagonal(); using Z21 = decltype(z21);
  auto z12 = z11.replicate<1,2>(); using Z12 = decltype(z12);
  auto z23 = z11.replicate<2,3>(); using Z23 = decltype(z23);
  using Z20 = Eigen::Replicate<Z11, 2, Eigen::Dynamic>; 
  using Z02 = Eigen::Replicate<Z11, Eigen::Dynamic, 2>; 
  using Z00 = Eigen::Replicate<Z11, Eigen::Dynamic, Eigen::Dynamic>; 
  using Z01 = Eigen::Replicate<Z11, Eigen::Dynamic, 1>; 
  auto z20_2 = Z20 {z11, 2, 2};
  auto z02_2 = Z02 {z11, 2, 2};
  auto z00_22 = Z00 {z11, 2, 2};
  auto z01_2 = Z01 {z11, 2, 1};
  auto z20_1 = Z20 {z11, 2, 1};
  auto z00_21 = Z00 {z11, 2, 1};

  auto c11_1 = M11::Identity(); using C11_1 = decltype(c11_1);
  auto c11_m1 = -M11::Identity(); using C11_m1 = decltype(c11_m1);
  auto c11_2 = c11_1 + c11_1; using C11_2 = decltype(c11_2);
  auto c11_m2 = -c11_2; using C11_m2 = decltype(c11_m2);
  auto c21_1 = c11_1.replicate<2,1>(); using C21_1 = decltype(c21_1);
  auto c21_2 = c11_2.replicate<2,1>(); using C21_2 = decltype(c21_2);
  auto c21_m2 = c11_m2.replicate<2,1>(); using C21_m2 = decltype(c21_m2);
  auto c12_2 = c11_2.replicate<1,2>(); using C12_2 = decltype(c12_2);
  auto c22_2 = c11_2.replicate<2,2>(); using C22_2 = decltype(c22_2);
  auto c22_m2 = c11_m2.replicate<2,2>(); using C22_m2 = decltype(c22_m2);
  using C20_2 = Eigen::Replicate<C11_2, 2, Eigen::Dynamic>;
  using C02_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  using C00_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, Eigen::Dynamic>;
  using C01_2 = Eigen::Replicate<C11_2, Eigen::Dynamic, 2>;
  auto c20_2_2 = C20_2 {c11_2, 2, 2};
  auto c02_2_2 = C02_2 {c11_2, 2, 2};
  auto c00_22_2 = C00_2 {c11_2, 2, 2};
  auto c20_1_2 = C20_2 {c11_2, 2, 1};
  auto c01_2_2 = C01_2 {c11_2, 2, 1};
  auto c00_21_2 = C00_2 {c11_2, 2, 1};

  auto b22_true = eigen_matrix_t<bool, 1, 1>::Identity().replicate<2,2>(); using B22_true = decltype(b22_true);
}


TEST(eigen3, ConstantMatrix_traits)
{
  static_assert(eigen_matrix<ConstantMatrix<double, 1, 2, 2>>);

  static_assert(zero_matrix<ConstantMatrix<double, 0, 2, 2>>);
  static_assert(zero_matrix<ConstantMatrix<double, 0, dynamic_extent, dynamic_extent>>);
  static_assert(not zero_matrix<ConstantMatrix<double, 1, 2, 2>>);

  static_assert(diagonal_matrix<ConstantMatrix<double, 0, 2, 2>>);
  static_assert(diagonal_matrix<ConstantMatrix<double, 5, 1, 1>>);
  static_assert(not diagonal_matrix<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);
  static_assert(not diagonal_matrix<ConstantMatrix<cdouble, 5, 2, 2>>);

  static_assert(self_adjoint_matrix<ConstantMatrix<double, 0, 2, 2>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<double, 5, 1, 1>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<cdouble, 5, 1, 1>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<double, 5, 2, 2>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<cdouble, 5, 2, 2>>);
  static_assert(not self_adjoint_matrix<ConstantMatrix<double, 5, 3, 4>>);
  static_assert(not self_adjoint_matrix<ConstantMatrix<cdouble, 5, 3, 4>>);

  static_assert(triangular_matrix<ConstantMatrix<double, 0, 2, 2>>);
  static_assert(triangular_matrix<ConstantMatrix<double, 5, 1, 1>>);
  static_assert(not triangular_matrix<ConstantMatrix<double, 5, 2, 2>>);
  static_assert(not triangular_matrix<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);
  static_assert(not triangular_matrix<ConstantMatrix<double, 5, 3, 4>>);
  static_assert(not triangular_matrix<ConstantMatrix<double, 0, 3, 4>>);

  static_assert(square_matrix<ConstantMatrix<double, 0, 2, 2>>);
  static_assert(square_matrix<ConstantMatrix<double, 5, 2, 2>>);
  static_assert(not square_matrix<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);
  static_assert(not square_matrix<ConstantMatrix<double, 5, 3, 4>>);

  static_assert(one_by_one_matrix<ConstantMatrix<double, 5, 1, 1>>);
  static_assert(not one_by_one_matrix<ConstantMatrix<double, 5, 1, dynamic_extent>>);
  static_assert(not one_by_one_matrix<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);

  static_assert(element_gettable<ConstantMatrix<double, 3, 2, 2>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<double, 3, 2, dynamic_extent>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<double, 3, dynamic_extent, 2>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<double, 3, dynamic_extent, dynamic_extent>, std::size_t, std::size_t>);

  static_assert(not element_settable<ConstantMatrix<double, 3, 2, 2>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<double, 3, 2, dynamic_extent>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<double, 3, dynamic_extent, 2>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<double, 3, 2, dynamic_extent>&, std::size_t, std::size_t>);

  static_assert(dynamic_rows<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);
  static_assert(dynamic_columns<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>);
  static_assert(dynamic_rows<ConstantMatrix<double, 5, dynamic_extent, 1>>);
  static_assert(not dynamic_columns<ConstantMatrix<double, 5, dynamic_extent, 1>>);
  static_assert(not dynamic_rows<ConstantMatrix<double, 5, 1, dynamic_extent>>);
  static_assert(dynamic_columns<ConstantMatrix<double, 5, 1, dynamic_extent>>);

  static_assert(zero_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 2, 3>>::zero())>);
  static_assert(zero_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 2, dynamic_extent>>::zero(3))>);
  static_assert(zero_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, 3>>::zero(2))>);
  static_assert(zero_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>::zero(2, 3))>);

  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, 2>>::identity())>);
  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, dynamic_extent>>::identity())>);
  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, 2>>::identity())>);

  static_assert(column_extent_of_v<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, 2>>::identity())> == 3);
  static_assert(column_extent_of_v<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, dynamic_extent>>::identity())> == 3);
  static_assert(column_extent_of_v<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, 2>>::identity())> == 2);

  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, 2>>::identity(4))>);
  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, 3, dynamic_extent>>::identity(4))>);
  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, 2>>::identity(4))>);
  static_assert(identity_matrix<decltype(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>::identity(4))>);

  EXPECT_EQ(column_count(MatrixTraits<ConstantMatrix<double, 5, 3, 2>>::identity(4)), 4);
  EXPECT_EQ(column_count(MatrixTraits<ConstantMatrix<double, 5, 3, dynamic_extent>>::identity(4)), 4);
  EXPECT_EQ(column_count(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, 2>>::identity(4)), 4);
  EXPECT_EQ(column_count(MatrixTraits<ConstantMatrix<double, 5, dynamic_extent, dynamic_extent>>::identity(4)), 4);

  static_assert(not writable<ConstantMatrix<double, 7, 3, 3>>);
  static_assert(modifiable<M31, ConstantMatrix<double, 7, 3, 1>>);
}


TEST(eigen3, ConstantMatrix_class)
{
  ConstantMatrix<double, 3, 2, 3> c323 {};
  ConstantMatrix<double, 3, 2, dynamic_extent> c320 {3};
  ConstantMatrix<double, 3, dynamic_extent, 3> c303 {2};
  ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> c300 {2,3};

  EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR(std::real(ConstantMatrix<cdouble, 3, 2, 2> {}(0, 1)), 3, 1e-6);

  EXPECT_TRUE(is_near(ConstantMatrix {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c300}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<double, 3, 2, 3> {}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<double, 3, 2, dynamic_extent> {3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<double, 3, dynamic_extent, 3> {2}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {2,3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<double, 2, 3> {}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<double, 2, dynamic_extent> {3}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<double, dynamic_extent, 3> {2}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<double, dynamic_extent, dynamic_extent> {2,3}}, M23::Zero()));

  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {ConstantMatrix {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {ConstantMatrix {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, dynamic_extent> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, 3> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {ConstantMatrix {c303}}, M23::Constant(3)));

  c320 = c323; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = c323; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = c323; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c320; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c303 = c320; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = c320; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c303; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = c303; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c300 = c303; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = c300; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = c300; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = c300; EXPECT_TRUE(is_near(c303, M23::Constant(3)));

  c323 = ConstantMatrix {c323}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantMatrix {c323}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantMatrix {c323}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantMatrix {c323}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantMatrix {c320}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantMatrix {c320}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantMatrix {c320}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantMatrix {c320}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantMatrix {c303}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantMatrix {c303}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantMatrix {c303}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantMatrix {c303}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));
  c323 = ConstantMatrix {c300}; EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  c320 = ConstantMatrix {c300}; EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  c303 = ConstantMatrix {c300}; EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  c300 = ConstantMatrix {c300}; EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR((ConstantMatrix<double, 3, 2, 2> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 2, 2> {}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 2, dynamic_extent> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, dynamic_extent, 2> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((ConstantMatrix<double, 3, 3, 1> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, dynamic_extent, 1> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 1, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 1, dynamic_extent> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 3, 1> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, dynamic_extent, 1> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 1, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<double, 3, 1, dynamic_extent> {3}[1]), 3, 1e-6);
}


TEST(eigen3, ConstantMatrix_overloads)
{
  ConstantMatrix<double, 5, 3, 4> c534 {};
  ConstantMatrix<double, 5, 3, dynamic_extent> c530_4 {4};
  ConstantMatrix<double, 5, dynamic_extent, 4> c504_3 {3};
  ConstantMatrix<double, 5, dynamic_extent, dynamic_extent> c500_34 {3, 4};

  ConstantMatrix<double, 5, 3, 3> c533 {};
  ConstantMatrix<double, 5, 3, dynamic_extent> c530_3 {3};
  ConstantMatrix<double, 5, dynamic_extent, 3> c503_3 {3};
  ConstantMatrix<double, 5, dynamic_extent, dynamic_extent> c500_33 {3, 3};

  ConstantMatrix<double, 5, 3, 1> c531 {};
  ConstantMatrix<double, 5, 3, dynamic_extent> c530_1 {1};
  ConstantMatrix<double, 5, dynamic_extent, 1> c501_3 {3};
  ConstantMatrix<double, 5, dynamic_extent, dynamic_extent> c500_31 {3, 1};

  EXPECT_TRUE(is_near(make_native_matrix(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_native_matrix(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_native_matrix(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_native_matrix(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_native_matrix(ConstantMatrix<cdouble, 5, 3, 4> {}), CM34::Constant(cdouble(5,0))));

  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c534}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c530_4}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c504_3}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c500_34}), M34::Constant(5)));

  EXPECT_EQ(row_count(c534), 3);
  EXPECT_EQ(row_count(c530_4), 3);
  EXPECT_EQ(row_count(c504_3), 3);
  EXPECT_EQ(row_count(c500_34), 3);

  EXPECT_EQ(column_count(c534), 4);
  EXPECT_EQ(column_count(c530_4), 4);
  EXPECT_EQ(column_count(c504_3), 4);
  EXPECT_EQ(column_count(c500_34), 4);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  auto m33_d5 = make_native_matrix<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5);

  EXPECT_TRUE(is_near(to_diagonal(c531), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c530_1), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c501_3), m33_d5));
  EXPECT_TRUE(is_near(to_diagonal(c500_31), m33_d5));

  EXPECT_TRUE(is_near(diagonal_of(c533), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c530_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c503_3), M31::Constant(5)));
  EXPECT_TRUE(is_near(diagonal_of(c500_33), M31::Constant(5)));
  static_assert(eigen_constant_expr<decltype(diagonal_of(c500_34))>);

  EXPECT_TRUE(is_near(transpose(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(transpose(c500_34), M43::Constant(5)));
  static_assert(eigen_constant_expr<decltype(transpose(c500_34))>);

  EXPECT_TRUE(is_near(adjoint(c534), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c530_4), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c504_3), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(c500_34), M43::Constant(5)));
  EXPECT_TRUE(is_near(adjoint(ConstantMatrix<cdouble, 5, 3, 4> {}), CM43::Constant(cdouble(5,0))));
  static_assert(eigen_constant_expr<decltype(adjoint(c500_34))>);

  EXPECT_NEAR(determinant(c533), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c530_3), 0, 1e-6);
  EXPECT_NEAR(determinant(c500_33), 0, 1e-6);

  EXPECT_NEAR(trace(c533), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c530_3), 15, 1e-6);
  EXPECT_NEAR(trace(c500_33), 15, 1e-6);

  // \todo rank_update

  M23 m23_66 = make_eigen_matrix<double, 2, 3>(6, 14, 22, 6, 14, 22);
  M20 m20_3_66 {2,3}; m20_3_66 = m23_66;
  M03 m03_2_66 {2,3}; m03_2_66 = m23_66;
  M00 m00_23_66 {2,3}; m00_23_66 = m23_66;
  auto m23_12 = make_eigen_matrix<double, 2, 3>(1.5, 3.5, 5.5, 1.5, 3.5, 5.5);

  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, m00_23_66), m23_12));

  ConstantMatrix<double, 8, 2, 3> c23_8;
  ConstantMatrix<double, 8, 2, dynamic_extent> c20_3_8 {3};
  ConstantMatrix<double, 8, dynamic_extent, 3> c03_2_8 {2};
  ConstantMatrix<double, 8, dynamic_extent, dynamic_extent> c00_23_8 {2, 3};
  ConstantMatrix<double, 2, 2, 3> c23_2;

  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c00_23_8), c23_2));

  ConstantMatrix<double, 6, 2, 3> c23_6;
  ConstantMatrix<double, 6, 2, dynamic_extent> c20_3_6 {3};
  ConstantMatrix<double, 6, dynamic_extent, 3> c03_2_6 {2};
  ConstantMatrix<double, 6, dynamic_extent, dynamic_extent> c00_23_6 {2, 3};
  auto m23_15 = make_eigen_matrix<double, 2, 3>(1.5, 1.5, 1.5, 1.5, 1.5, 1.5);

  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, 2> {}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 2, dynamic_extent> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {2, 2}, c00_23_6), m23_15));

  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, 1> {}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, 1> {}, make_eigen_matrix<double, 1, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, 1> {}, make_eigen_matrix<double, dynamic_extent, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, 1> {}, make_eigen_matrix<double, dynamic_extent, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, dynamic_extent> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, dynamic_extent> {1}, make_eigen_matrix<double, 1, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, dynamic_extent> {1}, make_eigen_matrix<double, dynamic_extent, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, 1, dynamic_extent> {1}, make_eigen_matrix<double, dynamic_extent, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 1> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 1> {1}, make_eigen_matrix<double, 1, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 1> {1}, make_eigen_matrix<double, dynamic_extent, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, 1> {1}, make_eigen_matrix<double, dynamic_extent, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {1, 1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {1, 1}, make_eigen_matrix<double, 1, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {1, 1}, make_eigen_matrix<double, dynamic_extent, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<double, 2, dynamic_extent, dynamic_extent> {1, 1}, make_eigen_matrix<double, dynamic_extent, dynamic_extent>(8)), make_eigen_matrix<double, 1, 1>(4)));

  EXPECT_TRUE(is_near(solve(M11::Identity(), make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));

  auto colzc34 = reduce_columns(c500_34);
  EXPECT_TRUE(is_near(reduce_columns(ConstantMatrix<double, 3, 2, 3> ()), (eigen_matrix_t<double, 2, 1>::Constant(3))));
  EXPECT_EQ(colzc34, (ConstantMatrix<double, 5, 3, 1> {}));
  EXPECT_EQ(row_count(colzc34), 3);
  EXPECT_EQ(column_count(colzc34), 1);
  static_assert(eigen_constant_expr<decltype(colzc34)>);

  auto rowzc34 = reduce_rows(c500_34);
  EXPECT_TRUE(is_near(reduce_rows(ConstantMatrix<double, 3, 2, 3> ()), (eigen_matrix_t<double, 1, 3>::Constant(3))));
  EXPECT_EQ(rowzc34, (ConstantMatrix<double, 5, 1, 4> {}));
  EXPECT_EQ(column_count(rowzc34), 4);
  EXPECT_EQ(row_count(rowzc34), 1);
  static_assert(eigen_constant_expr<decltype(rowzc34)>);

  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<double, 7, 5, 3> ()), LQ_decomposition(make_eigen_matrix<double, 5, 3>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto lq332 = make_self_contained(LQ_decomposition(make_eigen_matrix<double, 3, 2>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<double, 3, 3, 2> ()), lq332));
  auto lqzc30_2 = LQ_decomposition(ConstantMatrix<double, 3, 3, dynamic_extent> {2});
  EXPECT_TRUE(is_near(lqzc30_2, lq332));
  EXPECT_EQ(row_count(lqzc30_2), 3);
  EXPECT_EQ(column_count(lqzc30_2), 3);
  auto lqzc02_3 = LQ_decomposition(ConstantMatrix<double, 3, dynamic_extent, 2> {3});
  EXPECT_TRUE(is_near(lqzc02_3, lq332));
  EXPECT_EQ(row_count(lqzc02_3), 3);
  EXPECT_EQ(column_count(lqzc02_3), 3);
  auto lqzc00_32 = LQ_decomposition(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {3, 2});
  EXPECT_TRUE(is_near(lqzc00_32, lq332));
  EXPECT_EQ(row_count(lqzc00_32), 3);
  EXPECT_EQ(column_count(lqzc00_32), 3);

  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<double, 7, 3, 5> ()), QR_decomposition(make_eigen_matrix<double, 3, 5>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto qr323 = make_self_contained(QR_decomposition(make_eigen_matrix<double, 2, 3>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<double, 3, 2, 3> ()), qr323));
  auto qrzc20_3 = QR_decomposition(ConstantMatrix<double, 3, 2, dynamic_extent> {3});
  EXPECT_TRUE(is_near(qrzc20_3, qr323));
  EXPECT_EQ(row_count(qrzc20_3), 3);
  EXPECT_EQ(column_count(qrzc20_3), 3);
  auto qrzc03_2 = QR_decomposition(ConstantMatrix<double, 3, dynamic_extent, 3> {2});
  EXPECT_TRUE(is_near(qrzc03_2, qr323));
  EXPECT_EQ(row_count(qrzc03_2), 3);
  EXPECT_EQ(column_count(qrzc03_2), 3);
  auto qrzc00_23 = QR_decomposition(ConstantMatrix<double, 3, dynamic_extent, dynamic_extent> {2, 3});
  EXPECT_TRUE(is_near(qrzc00_23, qr323));
  EXPECT_EQ(row_count(qrzc00_23), 3);
  EXPECT_EQ(column_count(qrzc00_23), 3);

  // \todo concatenate_vertical
  // \todo concatenate_horizontal
  // \todo concatenate_vertical
  // \todo split_vertical
  // \todo split_horizontal
  // \todo vertical_vertical

  EXPECT_NEAR(get_element(ConstantMatrix<double, 5, 2, 2> {}, 1, 0), 5, 1e-8);

  ConstantMatrix<double, 5, dynamic_extent, dynamic_extent> c00 {2, 2};

  EXPECT_NEAR((get_element(c00, 0, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 0, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(ConstantMatrix<double, 7, 0, 1> {3}, 0)), 7, 1e-6);

  EXPECT_TRUE(is_near(column(ConstantMatrix<double, 6, 2, 3> {}, 1), (eigen_matrix_t<double, 2, 1>::Constant(6))));
  EXPECT_TRUE(is_near(column<1>(ConstantMatrix<double, 7, 2, 3> {}), (eigen_matrix_t<double, 2, 1>::Constant(7))));
  auto czc34 = column(c500_34, 1);
  EXPECT_EQ(row_count(czc34), 3);
  static_assert(column_count(czc34) == 1);
  static_assert(eigen_constant_expr<decltype(czc34)>);
  auto czv34 = column<1>(ConstantMatrix<double, 5, dynamic_extent, 4> {3});
  EXPECT_EQ(row_count(czv34), 3);
  static_assert(column_count(czv34) == 1);
  static_assert(eigen_constant_expr<decltype(czv34)>);

  EXPECT_TRUE(is_near(row(ConstantMatrix<double, 6, 3, 2> {}, 1), (eigen_matrix_t<double, 1, 2>::Constant(6))));
  EXPECT_TRUE(is_near(row<1>(ConstantMatrix<double, 7, 3, 2> {}), (eigen_matrix_t<double, 1, 2>::Constant(7))));
  auto rzc34 = row(c500_34, 1);
  EXPECT_EQ(column_count(rzc34), 4);
  static_assert(row_count(rzc34) == 1);
  static_assert(eigen_constant_expr<decltype(rzc34)>);
  auto rzv34 = row<1>(ConstantMatrix<double, 5, 3, dynamic_extent> {4});
  EXPECT_EQ(column_count(rzv34), 4);
  static_assert(row_count(rzv34) == 1);
  static_assert(eigen_constant_expr<decltype(rzv34)>);

  // \todo apply_columnwise
  // \todo apply_rowwise
  // \todo apply_coefficientwise
}


TEST(eigen3, ConstantMatrix_arithmetic)
{
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} + ConstantMatrix<double, 5, 2, 2> {}, ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} - ConstantMatrix<double, 5, 2, 2> {}, ConstantMatrix<double, -2, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {} * ConstantMatrix<double, 5, 3, 2> {}, ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 4, 3, 4> {} * ConstantMatrix<double, 7, 4, 2> {}, ConstantMatrix<double, 112, 3, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 0, 2, 2> {} * 2.0, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} * -2.0, ConstantMatrix<double, -6, 2, 2> {}));
  EXPECT_TRUE(is_near(3.0 * ConstantMatrix<double, 0, 2, 2> {}, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, -9, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 0, 2, 2> {} / 2.0, ConstantMatrix<double, 0, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 8, 2, 2> {} / -2.0, ConstantMatrix<double, -4, 2, 2> {}));
  EXPECT_TRUE(is_near(-ConstantMatrix<double, 7, 2, 2> {}, ConstantMatrix<double, -7, 2, 2> {}));

  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} + eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) + ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, 8, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 2> {} - eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<double, -2, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) - ConstantMatrix<double, 3, 2, 2> {}, ConstantMatrix<double, 2, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3> {} * eigen_matrix_t<double, 3, 2>::Constant(5), ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 4, 3, 4> {} * eigen_matrix_t<double, 4, 2>::Constant(7), ConstantMatrix<double, 112, 3, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 3>::Constant(3) * ConstantMatrix<double, 5, 3, 2> {}, ConstantMatrix<double, 45, 2, 2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 3, 4>::Constant(4) * ConstantMatrix<double, 7, 4, 2> {}, ConstantMatrix<double, 112, 3, 2> {}));

  EXPECT_EQ((ConstantMatrix<double, 3, 4, 3>::rows()), 4);
  EXPECT_EQ((ConstantMatrix<double, 3, 4, 3>::cols()), 3);
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3>::zero(), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix<double, 3, 2, 3>::identity(), M22::Identity()));
}

