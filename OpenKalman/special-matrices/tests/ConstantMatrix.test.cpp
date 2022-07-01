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

  using Axis2 = TypedIndex<Axis, Axis>;

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
  static_assert(eigen_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 1>>);

  static_assert(zero_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0>>);
  static_assert(zero_matrix<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 0>>);
  static_assert(not zero_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 1>>);

  static_assert(diagonal_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0>>);
  static_assert(diagonal_matrix<ConstantMatrix<eigen_matrix_t<double, 1, 1>, 5>>);
  static_assert(not diagonal_matrix<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);
  static_assert(not diagonal_matrix<ConstantMatrix<eigen_matrix_t<cdouble, 2, 2>, 5>>);

  static_assert(self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<double, 1, 1>, 5>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<cdouble, 1, 1>, 5>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5>>);
  static_assert(self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<cdouble, 2, 2>, 5>>);
  static_assert(not self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<double, 3, 4>, 5>>);
  static_assert(not self_adjoint_matrix<ConstantMatrix<eigen_matrix_t<cdouble, 3, 4>, 5>>);

  static_assert(triangular_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0>>);
  static_assert(triangular_matrix<ConstantMatrix<eigen_matrix_t<double, 1, 1>, 5>>);
  static_assert(not triangular_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5>>);
  static_assert(not triangular_matrix<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);
  static_assert(not triangular_matrix<ConstantMatrix<eigen_matrix_t<double, 3, 4>, 5>>);
  static_assert(not triangular_matrix<ConstantMatrix<eigen_matrix_t<double, 3, 4>, 0>>);

  static_assert(square_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0>>);
  static_assert(square_matrix<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5>>);
  static_assert(not square_matrix<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);
  static_assert(not square_matrix<ConstantMatrix<eigen_matrix_t<double, 3, 4>, 5>>);

  static_assert(one_by_one_matrix<ConstantMatrix<eigen_matrix_t<double, 1, 1>, 5>>);
  static_assert(not one_by_one_matrix<ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 5>>);
  static_assert(not one_by_one_matrix<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);

  static_assert(element_gettable<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 3>, std::size_t, std::size_t>);
  static_assert(element_gettable<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3>, std::size_t, std::size_t>);

  static_assert(not element_settable<ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 3>&, std::size_t, std::size_t>);
  static_assert(not element_settable<ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3>&, std::size_t, std::size_t>);

  static_assert(dynamic_rows<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);
  static_assert(dynamic_columns<ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>>);
  static_assert(dynamic_rows<ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 5>>);
  static_assert(not dynamic_columns<ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 5>>);
  static_assert(not dynamic_rows<ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 5>>);
  static_assert(dynamic_columns<ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 5>>);

  static_assert(not writable<ConstantMatrix<eigen_matrix_t<double, 3, 3>, 7>>);
  static_assert(modifiable<M31, ConstantMatrix<eigen_matrix_t<double, 3, 1>, 7>>);
}


TEST(eigen3, ConstantMatrix_class)
{
  ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> c323 {};
  ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> c320 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> c303 {2};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> c300 {2,3};

  EXPECT_TRUE(is_near(c323, M23::Constant(3)));
  EXPECT_TRUE(is_near(c320, M23::Constant(3)));
  EXPECT_TRUE(is_near(c303, M23::Constant(3)));
  EXPECT_TRUE(is_near(c300, M23::Constant(3)));

  EXPECT_NEAR(std::real(ConstantMatrix<eigen_matrix_t<cdouble, 2, 2>, 3> {}(0, 1)), 3, 1e-6);

  EXPECT_TRUE(is_near(ConstantMatrix {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {c300}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {3}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {2}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix {ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {2,3}}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<eigen_matrix_t<double, 2, 3>> {}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<eigen_matrix_t<double, 2, dynamic_size>> {3}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<eigen_matrix_t<double, dynamic_size, 3>> {2}}, M23::Zero()));
  EXPECT_TRUE(is_near(ConstantMatrix {ZeroMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>> {2,3}}, M23::Zero()));

  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {c303}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {c300}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {c323}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {c320}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {c303}, M23::Constant(3)));

  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {ConstantMatrix {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {ConstantMatrix {c303}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {ConstantMatrix {c300}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {ConstantMatrix {c323}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {ConstantMatrix {c320}}, M23::Constant(3)));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {ConstantMatrix {c303}}, M23::Constant(3)));

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

  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {}(0, 0)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 3> {2}(0, 1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {2,2}(0, 1)), 3, 1e-6);

  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 3, 1>, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 1, 3>, 3> {}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 3> {3}(1)), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 3, 1>, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 3> {3}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 1, 3>, 3> {}[1]), 3, 1e-6);
  EXPECT_NEAR((ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 3> {3}[1]), 3, 1e-6);
}


TEST(eigen3, ConstantMatrix_functions)
{
  ConstantMatrix<eigen_matrix_t<double, 3, 4>, 5> c534 {};
  ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 5> c530_4 {4};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 4>, 5> c504_3 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5> c500_34 {3, 4};

  ConstantMatrix<eigen_matrix_t<double, 3, 3>, 5> c533 {};
  ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 5> c530_3 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 5> c503_3 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5> c500_33 {3, 3};

  ConstantMatrix<eigen_matrix_t<double, 3, 1>, 5> c531 {};
  ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 5> c530_1 {1};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 5> c501_3 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5> c500_31 {3, 1};

  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c534), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c530_4), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c504_3), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(c500_34), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_dense_writable_matrix_from(ConstantMatrix<eigen_matrix_t<cdouble, 3, 4>, 5> {}), CM34::Constant(cdouble(5,0))));

  using C534 = decltype(c534);
  using C500 = decltype(c500_34);

  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>())> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), Dimensions<4>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), 4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(3, Dimensions<4>()))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<C534, 5>(3, 4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c534))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c530_4))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c504_3))> == 5);
  static_assert(constant_coefficient_v<decltype(make_constant_matrix_like<5>(c500_34))> == 5);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), Dimensions<4>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(Dimensions<3>(), 4)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(3, Dimensions<4>())), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<C500>(3, Dimensions<4>())), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534, 5>(3, 4)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like<C500>(3, 4)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c534)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c530_4)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c503_4)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z04_3)), 3);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(c500_34)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_zero_matrix_like(z00_34)), 3);

  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>()), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(Dimensions<3>(), Dimensions<4>())), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(Dimensions<3>(), 4)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<C500>(3, 4)), 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(3, Dimensions<4>())), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like<C534>(3, 4)), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like<C500>(3, 4)), 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(z34)), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(z30_4)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z30_4)), 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(z04_3)), 1> == 4);
  static_assert(index_dimension_of_v<decltype(make_constant_matrix_like(z00_34)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_zero_matrix_like(z00_34)), 4);

  static_assert(identity_matrix<decltype(make_identity_matrix_like<C533>())>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(Dimensions<3>()))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like<C500>(3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c533))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c530_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c503_3))>);
  static_assert(identity_matrix<decltype(make_identity_matrix_like(c500_33))>);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like<C500>(3)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 0> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(c503_2)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 0> == dynamic_size); EXPECT_EQ(get_dimensions_of<0>(make_identity_matrix_like(c500_32)), 3);

  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C533>()), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(Dimensions<3>())), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like<C500>(3), 1> == dynamic_size);  EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like<C500>(2)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c533)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c530_3)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(c520_2)), 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c503_3)), 1> == 3);
  static_assert(index_dimension_of_v<decltype(make_identity_matrix_like(c500_33)), 1> == dynamic_size); EXPECT_EQ(get_dimensions_of<1>(make_identity_matrix_like(c500_22)), 3);

  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c534}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c530_4}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c504_3}), M34::Constant(5)));
  EXPECT_TRUE(is_near(make_self_contained(ConstantMatrix {c500_34}), M34::Constant(5)));

  EXPECT_EQ(get_dimensions_of<0>(c534), 3);
  EXPECT_EQ(get_dimensions_of<0>(c530_4), 3);
  EXPECT_EQ(get_dimensions_of<0>(c504_3), 3);
  EXPECT_EQ(get_dimensions_of<0>(c500_34), 3);

  EXPECT_EQ(get_dimensions_of<1>(c534), 4);
  EXPECT_EQ(get_dimensions_of<1>(c530_4), 4);
  EXPECT_EQ(get_dimensions_of<1>(c504_3), 4);
  EXPECT_EQ(get_dimensions_of<1>(c500_34), 4);

  // to_euclidean is tested in ToEuclideanExpr.test.cpp.
  // from_euclidean is tested in FromEuclideanExpr.test.cpp.
  // wrap_angles is tested in FromEuclideanExpr.test.cpp.

  auto m33_d5 = make_dense_writable_matrix_from<M33>(5, 0, 0, 0, 5, 0, 0, 0, 5);

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
  EXPECT_TRUE(is_near(adjoint(ConstantMatrix<eigen_matrix_t<cdouble, 3, 4>, 5> {}), CM43::Constant(cdouble(5,0))));
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

  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, m00_23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, m23_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, m20_3_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, m03_2_66), m23_12));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, m00_23_66), m23_12));

  ConstantMatrix<eigen_matrix_t<double, 2, 3>, 8> c23_8;
  ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 8> c20_3_8 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 8> c03_2_8 {2};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 8> c00_23_8 {2, 3};
  ConstantMatrix<eigen_matrix_t<double, 2, 3>, 2> c23_2;

  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c00_23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c23_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c20_3_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c03_2_8), c23_2));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c00_23_8), c23_2));

  ConstantMatrix<eigen_matrix_t<double, 2, 3>, 6> c23_6;
  ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 6> c20_3_6 {3};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 6> c03_2_6 {2};
  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 6> c00_23_6 {2, 3};
  auto m23_15 = make_eigen_matrix<double, 2, 3>(1.5, 1.5, 1.5, 1.5, 1.5, 1.5);

  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 2> {2}, c00_23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c23_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c20_3_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c03_2_6), m23_15));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {2, 2}, c00_23_6), m23_15));

  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, 1>, 2> {}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, 1>, 2> {}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, 1>, 2> {}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, 1>, 2> {}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, 1, dynamic_size>, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 2> {1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 2> {1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 2> {1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 2> {1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {1, 1}, make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {1, 1}, make_eigen_matrix<double, 1, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));
  EXPECT_TRUE(is_near(solve(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 2> {1, 1}, make_eigen_matrix<double, dynamic_size, dynamic_size>(8)), make_eigen_matrix<double, 1, 1>(4)));

  EXPECT_TRUE(is_near(solve(M11::Identity(), make_eigen_matrix<double, 1, 1>(8)), make_eigen_matrix<double, 1, 1>(4)));

  auto colzc34 = average_reduce<1>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<1>(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> ()), (eigen_matrix_t<double, 2, 1>::Constant(3))));
  EXPECT_EQ(colzc34, (ConstantMatrix<eigen_matrix_t<double, 3, 1>, 5> {}));
  EXPECT_EQ(get_dimensions_of<0>(colzc34), 3);
  EXPECT_EQ(get_dimensions_of<1>(colzc34), 1);
  static_assert(eigen_constant_expr<decltype(colzc34)>);

  auto rowzc34 = average_reduce<0>(c500_34);
  EXPECT_TRUE(is_near(average_reduce<0>(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> ()), (eigen_matrix_t<double, 1, 3>::Constant(3))));
  EXPECT_EQ(rowzc34, (ConstantMatrix<eigen_matrix_t<double, 1, 4>, 5> {}));
  EXPECT_EQ(get_dimensions_of<1>(rowzc34), 4);
  EXPECT_EQ(get_dimensions_of<0>(rowzc34), 1);
  static_assert(eigen_constant_expr<decltype(rowzc34)>);

  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<eigen_matrix_t<double, 5, 3>, 7> ()), LQ_decomposition(make_eigen_matrix<double, 5, 3>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto lq332 = make_self_contained(LQ_decomposition(make_eigen_matrix<double, 3, 2>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(LQ_decomposition(ConstantMatrix<eigen_matrix_t<double, 3, 2>, 3> ()), lq332));
  auto lqzc30_2 = LQ_decomposition(ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 3> {2});
  EXPECT_TRUE(is_near(lqzc30_2, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc30_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc30_2), 3);
  auto lqzc02_3 = LQ_decomposition(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 2>, 3> {3});
  EXPECT_TRUE(is_near(lqzc02_3, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc02_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc02_3), 3);
  auto lqzc00_32 = LQ_decomposition(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {3, 2});
  EXPECT_TRUE(is_near(lqzc00_32, lq332));
  EXPECT_EQ(get_dimensions_of<0>(lqzc00_32), 3);
  EXPECT_EQ(get_dimensions_of<1>(lqzc00_32), 3);

  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<eigen_matrix_t<double, 3, 5>, 7> ()), QR_decomposition(make_eigen_matrix<double, 3, 5>(7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7)).cwiseAbs()));
  auto qr323 = make_self_contained(QR_decomposition(make_eigen_matrix<double, 2, 3>(3, 3, 3, 3, 3, 3)).cwiseAbs());
  EXPECT_TRUE(is_near(QR_decomposition(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> ()), qr323));
  auto qrzc20_3 = QR_decomposition(ConstantMatrix<eigen_matrix_t<double, 2, dynamic_size>, 3> {3});
  EXPECT_TRUE(is_near(qrzc20_3, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc20_3), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc20_3), 3);
  auto qrzc03_2 = QR_decomposition(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 3>, 3> {2});
  EXPECT_TRUE(is_near(qrzc03_2, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc03_2), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc03_2), 3);
  auto qrzc00_23 = QR_decomposition(ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 3> {2, 3});
  EXPECT_TRUE(is_near(qrzc00_23, qr323));
  EXPECT_EQ(get_dimensions_of<0>(qrzc00_23), 3);
  EXPECT_EQ(get_dimensions_of<1>(qrzc00_23), 3);

  // \todo concatenate_vertical
  // \todo concatenate_horizontal
  // \todo concatenate_vertical
  // \todo split_vertical
  // \todo split_horizontal
  // \todo vertical_vertical

  EXPECT_NEAR(get_element(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5> {}, 1, 0), 5, 1e-8);

  ConstantMatrix<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5> c00 {2, 2};

  EXPECT_NEAR((get_element(c00, 0, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 0, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 0)), 5, 1e-6);
  EXPECT_NEAR((get_element(c00, 1, 1)), 5, 1e-6);
  EXPECT_NEAR((get_element(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 1>, 7> {3}, 0)), 7, 1e-6);

  EXPECT_TRUE(is_near(column(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 6> {}, 1), (eigen_matrix_t<double, 2, 1>::Constant(6))));
  EXPECT_TRUE(is_near(column<1>(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 7> {}), (eigen_matrix_t<double, 2, 1>::Constant(7))));
  auto czc34 = column(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<0>(czc34), 3);
  static_assert(get_dimensions_of<1>(czc34) == 1);
  static_assert(eigen_constant_expr<decltype(czc34)>);
  auto czv34 = column<1>(ConstantMatrix<eigen_matrix_t<double, dynamic_size, 4>, 5> {3});
  EXPECT_EQ(get_dimensions_of<0>(czv34), 3);
  static_assert(get_dimensions_of<1>(czv34) == 1);
  static_assert(eigen_constant_expr<decltype(czv34)>);

  EXPECT_TRUE(is_near(row(ConstantMatrix<eigen_matrix_t<double, 3, 2>, 6> {}, 1), (eigen_matrix_t<double, 1, 2>::Constant(6))));
  EXPECT_TRUE(is_near(row<1>(ConstantMatrix<eigen_matrix_t<double, 3, 2>, 7> {}), (eigen_matrix_t<double, 1, 2>::Constant(7))));
  auto rzc34 = row(c500_34, 1);
  EXPECT_EQ(get_dimensions_of<1>(rzc34), 4);
  static_assert(get_dimensions_of<0>(rzc34) == 1);
  static_assert(eigen_constant_expr<decltype(rzc34)>);
  auto rzv34 = row<1>(ConstantMatrix<eigen_matrix_t<double, 3, dynamic_size>, 5> {4});
  EXPECT_EQ(get_dimensions_of<1>(rzv34), 4);
  static_assert(get_dimensions_of<0>(rzv34) == 1);
  static_assert(eigen_constant_expr<decltype(rzv34)>);

  // \todo apply_columnwise
  // \todo apply_rowwise
  // \todo apply_coefficientwise
}


TEST(eigen3, ConstantMatrix_arithmetic)
{
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {} + ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 8> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {} - ConstantMatrix<eigen_matrix_t<double, 2, 2>, 5> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, -2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {} * ConstantMatrix<eigen_matrix_t<double, 3, 2>, 5> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 45> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 3, 4>, 4> {} * ConstantMatrix<eigen_matrix_t<double, 4, 2>, 7> {}, ConstantMatrix<eigen_matrix_t<double, 3, 2>, 112> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {} * 2.0, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {} * -2.0, ConstantMatrix<eigen_matrix_t<double, 2, 2>, -6> {}));
  EXPECT_TRUE(is_near(3.0 * ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {}));
  EXPECT_TRUE(is_near(-3.0 * ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, -9> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {} / 2.0, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 0> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 8> {} / -2.0, ConstantMatrix<eigen_matrix_t<double, 2, 2>, -4> {}));
  EXPECT_TRUE(is_near(-ConstantMatrix<eigen_matrix_t<double, 2, 2>, 7> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, -7> {}));

  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {} + eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<eigen_matrix_t<double, 2, 2>, 8> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) + ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 8> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {} - eigen_matrix_t<double, 2, 2>::Constant(5), ConstantMatrix<eigen_matrix_t<double, 2, 2>, -2> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 2>::Constant(5) - ConstantMatrix<eigen_matrix_t<double, 2, 2>, 3> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 2> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3> {} * eigen_matrix_t<double, 3, 2>::Constant(5), ConstantMatrix<eigen_matrix_t<double, 2, 2>, 45> {}));
  EXPECT_TRUE(is_near(ConstantMatrix<eigen_matrix_t<double, 3, 4>, 4> {} * eigen_matrix_t<double, 4, 2>::Constant(7), ConstantMatrix<eigen_matrix_t<double, 3, 2>, 112> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 2, 3>::Constant(3) * ConstantMatrix<eigen_matrix_t<double, 3, 2>, 5> {}, ConstantMatrix<eigen_matrix_t<double, 2, 2>, 45> {}));
  EXPECT_TRUE(is_near(eigen_matrix_t<double, 3, 4>::Constant(4) * ConstantMatrix<eigen_matrix_t<double, 4, 2>, 7> {}, ConstantMatrix<eigen_matrix_t<double, 3, 2>, 112> {}));

  EXPECT_EQ((ConstantMatrix<eigen_matrix_t<double, 4, 3>, 3>::rows()), 4);
  EXPECT_EQ((ConstantMatrix<eigen_matrix_t<double, 4, 3>, 3>::cols()), 3);
  EXPECT_TRUE(is_near(make_zero_matrix_like<ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3>>(), eigen_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(make_identity_matrix_like<ConstantMatrix<eigen_matrix_t<double, 2, 3>, 3>>(), M22::Identity()));
}

